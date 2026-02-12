import requests
import gzip, io
import os, sys, time, ast
import tarfile
import pickle
import tenseal as ts
import torch
import struct
import hashlib
import numpy as np

from eth_account.messages import encode_defunct
from eth_keys import keys
from eth_account._utils.legacy_transactions import serializable_unsigned_transaction_from_dict
from eth_account._utils.signing import to_standard_v
from eth_account.datastructures import SignedMessage

from Crypto.Util.number import bytes_to_long, long_to_bytes
from Crypto.Hash import SHAKE128, SHA384
from Crypto.Cipher import AES
from collections import defaultdict

# -----------------------------
# IPFS CONFIG & HELPERS
# -----------------------------
# IPFS API typically runs on 5001 for control/add, Gateway on 8080 for retrieval
IPFS_API = "http://127.0.0.1:5001/api/v0"

def ipfs_version():
    """Check IPFS daemon connectivity."""
    try:
        r = requests.post(f"{IPFS_API}/version")
        return r.json()
    except Exception as e:
        print(f"⚠️ IPFS daemon not reachable: {e}")
        return None

def upload_to_Ipfs(wrapped_data):
    """
    Compresses data and uploads it to IPFS.
    Returns the CID (Content Identifier).
    """
    try:
        compressed_data = gzip.compress(wrapped_data)
        files = {"file": ("model_update.bin", compressed_data)}
        # API endpoint /add handles file uploads
        r = requests.post(f"{IPFS_API}/add", files=files)
        r.raise_for_status()
        cid = r.json()["Hash"]
        print(f"✅ IPFS Upload Successful. CID: {cid}")
        return cid
    except Exception as e:
        print(f"⚠️ Upload to IPFS failed: {e}")
        return None

def verify_sign(signed_data, msg, pubkey):
    msg_hash = signed_data[:32]
    r_sign = bytes_to_long(signed_data[32:64])
    s_sign = bytes_to_long(signed_data[64:96])
    v_sign = bytes_to_long(signed_data[96:97])
    sign_bytes = signed_data[97:]
    signature = SignedMessage(messageHash=msg_hash, r=r_sign, s=s_sign, v=v_sign, signature=sign_bytes)
    if not signature:
        raise ValueError("Invalid signed message data structure.")
    return True



def get_from_Ipfs(cid):
    """
    Fetches data from IPFS using a CID.
    Automatically decompresses the fetched gzip content.
    """
    try:
        # API endpoint /cat retrieves file content
        r = requests.post(f"{IPFS_API}/cat", params={"arg": cid})
        r.raise_for_status()
        # Data was compressed before upload, so we unzip it upon retrieval
        return unzip(r.content)
    except Exception as e:
        print(f"⚠️ Fetch from IPFS failed for CID {cid}: {e}")
        return None

# -----------------------------
# CRYPTO & SERIALIZATION HELPERS
# -----------------------------
def kdf(x):
    return SHAKE128.new(x).read(32)

def wrapfiles(*files):   # Input: ('A.bin', data_A), ('B.enc', data_B)
    """Packages multiple files into a single TAR buffer for transfer."""
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
        for file_name, file_data in files:
            file_info = tarfile.TarInfo(name=file_name)
            file_info.size = len(file_data)
            tar.addfile(file_info, io.BytesIO(file_data))
    return tar_buffer.getvalue()

def unwrap_files(tar_data):
    """Extracts files from a TAR buffer."""
    extracted_files = {}
    tar_buffer = io.BytesIO(tar_data)
    with tarfile.open(fileobj=tar_buffer, mode='r') as tar:
        for member in tar.getmembers():
            file = tar.extractfile(member)
            if file is not None:
                extracted_files[member.name] = file.read()
    return extracted_files

def unzip(gzip_data):
    """Decompress gzip bytes."""
    with gzip.GzipFile(fileobj=io.BytesIO(gzip_data)) as gz_file:
        return gz_file.read()

def serialize_data(encrypted_model, metadata, HE_algorithm):
    """Serializes encrypted TenSEAL vectors for storage/IPFS."""
    data_package = {
        'weights': {name: enc_weight.serialize() for name, enc_weight in encrypted_model.items()},
        'metadata': metadata,
        'algorithm': HE_algorithm
    }
    return pickle.dumps(data_package)

def deserialize_data(serialized_data, context):
    """Reconstructs TenSEAL vectors from serialized bytes."""
    data_package = pickle.loads(serialized_data)
    algorithm = data_package['algorithm']
    deserialized_weights = {}
    for name, weight_bytes in data_package['weights'].items():
        if algorithm == 'BFV':
            deserialized_weights[name] = ts.bfv_vector_from(context, weight_bytes)
        else:
            deserialized_weights[name] = ts.ckks_vector_from(context, weight_bytes)
    return deserialized_weights, data_package['metadata']

# -----------------------------
# HOMOMORPHIC ENCRYPTION HELPERS
# -----------------------------
def HE_encrypt_model(model, context, HE_algorithm):
    """Encrypts PyTorch model parameters using BFV or CKKS."""
    context.generate_galois_keys()
    context.generate_relin_keys()
    
    encrypted_weights = {}
    metadata = {'scaling_factors': {}, 'norms': {}, 'num_clients': 1}

    for name, param in model.named_parameters():
        param_data = param.detach().cpu().numpy().flatten()
        
        if HE_algorithm == "BFV":
            norm = np.linalg.norm(param_data)
            if norm != 0: param_data = param_data / norm
            metadata['norms'][name] = norm
            scale = 1e5
            metadata['scaling_factors'][name] = scale
            param_int = np.round(param_data * scale).astype(int).tolist()
            encrypted_weights[name] = ts.bfv_vector(context, param_int)
        else: # CKKS
            scale = 2**40
            metadata['scaling_factors'][name] = scale
            encrypted_weights[name] = ts.ckks_vector(context, param_data.tolist(), scale=scale)

    return encrypted_weights, metadata

def HE_decrypt_model(encrypted_weights, model, context, HE_algorithm, metadata):
    """Decrypts and updates model weights."""
    state_dict = model.state_dict()
    num_clients = metadata.get('num_clients', 1)

    for name, encrypted_weight in encrypted_weights.items():
        dec_weight = np.array(encrypted_weight.decrypt())
        if HE_algorithm == "BFV":
            scale = metadata['scaling_factors'].get(name, 1.0)
            norm = metadata['norms'].get(name, 1.0)
            dec_weight = (dec_weight / (scale * num_clients)) * norm
        else: # CKKS
            dec_weight = dec_weight / num_clients
            
        tensor_weight = torch.tensor(dec_weight).view(state_dict[name].shape)
        state_dict[name].copy_(tensor_weight.to(state_dict[name].dtype))
    return model

# -----------------------------
# BLOCKCHAIN SIGNATURE HELPERS
# -----------------------------
def sign_data(msg, Eth_private_key, web3):
    """Signs data using the Ethereum private key."""
    msg_hex = msg.hex() if isinstance(msg, bytes) else msg
    encoded_ct = encode_defunct(text=msg_hex)
    signed = web3.eth.account.sign_message(encoded_ct, private_key=Eth_private_key)
    # Pack v, r, s for full signature reconstruction
    return long_to_bytes(signed.v, 1) + long_to_bytes(signed.r, 32) + long_to_bytes(signed.s, 32) + signed.signature

def hash_data(data):
    """SHA-256 Hashing."""
    if isinstance(data, str): data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def AES_encrypt_data(key, msg):
    nonce = os.urandom(8)
    crypto = AES.new(key, AES.MODE_CTR, nonce=nonce)
    return nonce + crypto.encrypt(msg)

def pubKey_from_tx(tx_hash, web3):
    tx = web3.eth.get_transaction(tx_hash)
    v = tx['v']
    r = int(tx['r'].hex(), 16)
    s = int(tx['s'].hex(), 16)
    unsigned_tx = serializable_unsigned_transaction_from_dict({
        'nonce': tx['nonce'],
        'gasPrice': tx['gasPrice'],
        'gas': tx['gas'],
        'to': tx['to'],
        'value': tx['value'],
        'data': tx['input']
    })
    tx_hash_bytes = unsigned_tx.hash()
    standard_v = to_standard_v(v)
    signature = keys.Signature(vrs=(standard_v, r, s))
    public_key = signature.recover_public_key_from_msg_hash(tx_hash_bytes)
    return public_key



def AES_decrypt_data(key, cipher):
    nonce = cipher[:8]
    crypto = AES.new(key, AES.MODE_CTR, nonce=nonce)
    return crypto.decrypt(cipher[8:])