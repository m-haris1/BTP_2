from web3 import Web3
from eth_account import Account
from eth_account.messages import *
from eth_keys import keys

from pqcrypto.kem import ml_kem_768

from Crypto.Protocol.DH import key_agreement
from Crypto.Protocol.KDF import HKDF
from Crypto.PublicKey import ECC
from Crypto.Hash import SHA384
from Crypto.Util.number import *  # Imports bytes_to_long, long_to_bytes

import tenseal as ts
import socket
import pickle
import json
import os
import sys
import time
import train_model

# --- FIX: Path manipulation must occur before external imports to resolve ModuleNotFoundError ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ! THis is the correct one
# from utils import (
#     wrapfiles, unwrap_files, receive_Model, send_model,
#     AES_encrypt_data, AES_decrypt_data, sign_data, verify_sign,
#     hash_data, HE_encrypt_model, serialize_data, deserialize_data,
#     HE_decrypt_model, kdf, pubKey_from_tx
# )

from utils import (
    wrapfiles, unwrap_files, get_from_Ipfs, upload_to_Ipfs, unzip,
    AES_encrypt_data, AES_decrypt_data, sign_data,
    hash_data, HE_encrypt_model, serialize_data, deserialize_data,
    HE_decrypt_model, kdf
)

# --- END FIX ---

# Global variables will be defined in __main__
w3 = None
contract = None
contract_address = None
ETH_address = None
Eth_private_key = None
registered_id_p = 0  # To be set after registration
last_seen_task_id = 0  # Track the highest task ID we've seen to avoid reprocessing old tasks



def register_client(hash_epk, project_id):
    """
    Register client on-chain.
    Contract signature: registerClient(string,uint256)

    Args:
        hash_epk (str or bytes): Hex string (with or without '0x') or raw bytes
        project_id (int): The project ID

    Returns:
        tuple: (initial_score, tx_registration, project_id)
    """
    global w3, contract, ETH_address, registered_id_p

    # Always normalize to hex string with '0x' prefix
    if isinstance(hash_epk, (bytes, bytearray)):
        abi_value = "0x" + hash_epk.hex()
    else:
        s = str(hash_epk)
        if not (s.startswith("0x") or s.startswith("0X")):
            abi_value = "0x" + s
        else:
            abi_value = s

    # Debug print
    print(f"DEBUG: Entering register_client. abi_value type={type(abi_value)} value start={abi_value[:10]}...")

    try:
        # Call the contract with a string (hex with 0x prefix)
        Call_reg = contract.functions.registerClient(abi_value, int(project_id)).transact({'from': ETH_address})
        receipt = w3.eth.wait_for_transaction_receipt(Call_reg)
        gas_used = receipt['gasUsed']
        tx_registration = receipt['transactionHash'].hex()

        # No need to manually decode logs if ABI events are set up, but leave defensive
        logs = receipt.get('logs', [])
        if not logs:
            print("Warning: no logs found in registration receipt.")
            registered_id_p = int(project_id)
            return 0, tx_registration, registered_id_p

        # Simplify: just print receipt info
        print("✔ Registration successful")
        print(f"    Project ID: {project_id}")
        print(f"    Tx: {tx_registration}")
        print(f"    Your Address: {ETH_address}")
        print(f"    Gas: {gas_used} Wei")
        print('-' * 75)

        registered_id_p = int(project_id)
        return 0, tx_registration, registered_id_p

    except Exception as e:
        print(f"An unexpected error occurred during registration: {e}")
        sys.exit(1)






def task_completed(task_id, project_id):
    global contract
    return contract.functions.isTaskDone(task_id, project_id).call()


def listen_for_projcet():
    global contract
    print("Listen for project...")
    # Diagnostic: print chain and block so you can verify all processes use same RPC
    try:
        print(f"DEBUG: chain_id={w3.eth.chain_id}, block_number={w3.eth.block_number}")
    except Exception:
        pass

    # First, try a short historical scan to catch recently emitted events the client may have missed
    try:
        latest = w3.eth.block_number
        from_block = max(0, latest - 200)
        print(f"DEBUG CLIENT: Scanning ProjectRegistered events from blocks {from_block}..{latest}")
        try:
            events = contract.events.ProjectRegistered().getLogs(fromBlock=from_block, toBlock=latest)
        except Exception:
            # Fallback: create filter and fetch entries
            f = contract.events.ProjectRegistered.create_filter(fromBlock=from_block, toBlock=latest)
            events = f.get_all_entries()
            try:
                w3.eth.uninstall_filter(f.filter_id)
            except Exception:
                pass

        if events:
            # Prefer the most recent ProjectRegistered event in the scanned range.
            try:
                events_sorted = sorted(events, key=lambda e: (e.get('blockNumber', 0), e['args'].get('project_id', 0)))
            except Exception:
                events_sorted = events
            ev = events_sorted[-1]
            try:
                found_ids = [e['args'].get('project_id') for e in events_sorted]
                print(f"DEBUG CLIENT: Historic ProjectRegistered events found (project_ids={found_ids}), selecting latest {ev['args'].get('project_id')}")
            except Exception:
                pass
            project_id = ev['args']['project_id']
            cnt_clients = ev['args']['cnt_clients']
            server_address = ev['args']['serverAddress']
            creation_time = time.gmtime(int(ev['args']['transactionTime'])) if 'transactionTime' in ev['args'] else time.gmtime()
            initial_model_hash = ev['args'].get('hash_init_model')
            server_hash_pubkeys = ev['args'].get('hash_keys')
            tx_hash = ev['transactionHash']
            print('Received Project Info (historic):')
            print(f'    Project ID: {project_id}')
            print(f'    Server address: {server_address}')
            print(f'    required client count: {cnt_clients}')
            print(f'    Time: {time.strftime("%Y-%m-%d %H:%M:%S (UTC)", creation_time)}')
            print(f'    Hash_pubkeys: {server_hash_pubkeys}')
            print('-' * 75)
            return tx_hash, project_id, server_address, cnt_clients, initial_model_hash, server_hash_pubkeys
    except Exception as e:
        print(f"DEBUG CLIENT: Historic scan failed: {e}")

    # If historic scan didn't find anything, fall back to live event listening
    try:
        task_event_filter = contract.events.ProjectRegistered.create_filter(fromBlock="latest")
        print("DEBUG CLIENT: No historic ProjectRegistered found; listening for new events...")
        while True:
            events = task_event_filter.get_new_entries()
            if events:
                events = sorted(events, key=lambda e: e['args']['project_id'])
                ev = events[-1]
                project_id = ev['args']['project_id']
                cnt_clients = ev['args']['cnt_clients']
                server_address = ev['args']['serverAddress']
                creation_time = time.gmtime(int(ev['args']['transactionTime'])) if 'transactionTime' in ev['args'] else time.gmtime()
                initial_model_hash = ev['args'].get('hash_init_model')
                server_hash_pubkeys = ev['args'].get('hash_keys')
                tx_hash = ev['transactionHash']
                print('Received Project Info:')
                print(f'    Project ID: {project_id}')
                print(f'    Server address: {server_address}')
                print(f'    required client count: {cnt_clients}')
                print(f'    Time: {time.strftime("%Y-%m-%d %H:%M:%S (UTC)", creation_time)}')
                print(f'    Hash_pubkeys: {server_hash_pubkeys}')
                print('-' * 75)
                return tx_hash, project_id, server_address, cnt_clients, initial_model_hash, server_hash_pubkeys
            time.sleep(1)
    except Exception as e:
        print(f"Error fetching project events: {e}")
    return None, None, None, None, None, None


def listen_for_task(timeout):
    global contract, registered_id_p, last_seen_task_id
    print("Listen for task...")
    start_time = time.time()
    Task_id = Hashed_model = round_num = hash_keys = project_id_received = server_address = D_t = 0
    while True:
        try:
            latest_block = w3.eth.block_number
            start_block = max(0, latest_block - 20)   # look back 20 blocks
            task_event_filter = contract.events.TaskPublished.create_filter(fromBlock=start_block)

            events = task_event_filter.get_all_entries()
            if events:
                # Filter events for our registered project and find the next unprocessed task
                matching_events = []
                for event in events:
                    project_id_received = event['args']['project_id']
                    Task_id = event['args']['taskId']
                    if registered_id_p == project_id_received and Task_id > last_seen_task_id:
                        matching_events.append((Task_id, event))
                
                # Process the smallest task ID first (to process rounds in order)
                if matching_events:
                    matching_events.sort(key=lambda x: x[0])  # Sort by Task_id
                    Task_id, event = matching_events[0]
                    
                    round_num = event['args']['round']
                    server_address = event['args']['serverAddress']
                    Hashed_model = event['args']['HashModel']
                    hash_keys = event['args']['hash_keys']
                    project_id_received = event['args']['project_id']
                    tx_hash = event['transactionHash'].hex()
                    creation_time = time.gmtime(int(event['args']['creationTime']))
                    D_t = time.gmtime(int(event['args']['DeadlineTask']))

                    print('Published Task Info:')
                    print(f'    Task ID: {Task_id}')
                    print(f'    Project ID: {project_id_received}')
                    print(f'    Server address: {server_address}')
                    print(f'    Transaction Hash: {tx_hash}')
                    print(f'    Time: {time.strftime("%Y-%m-%d %H:%M:%S (UTC)", creation_time)}')
                    print(f'    Deadline: {time.strftime("%Y-%m-%d %H:%M:%S (UTC)", D_t)}')
                    print('-' * 75)
                    last_seen_task_id = Task_id
                    return round_num, Task_id, Hashed_model, hash_keys, project_id_received, server_address, D_t
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                break
        except Exception as e:
            print(f"Error while fetching tasks: {e}")
            break
        time.sleep(1)
    return 0, 0, None, None, 0, None, 0


# ! THis is the original one
# def update_model_Tx(r, Hash_model, hash_ct_epk, Task_id, project_id):
#     global w3, ETH_address, Eth_private_key, contract
#     print(f"DEBUG: Updating Model Tx. R: {r}, Model Hash: {Hash_model[:10]}..., ct_epk_hash: {hash_ct_epk[:10]}...")
#     try:
#         nonce = w3.eth.get_transaction_count(ETH_address)  # Fetch the latest nonce for the account
#         transaction = contract.functions.updateModel(r, Hash_model, hash_ct_epk, Task_id, project_id).build_transaction({
#             'from': ETH_address,
#             'nonce': nonce,
#             'gas': 2000000,  # Adjust gas limit if necessary
#             'gasPrice': w3.to_wei('50', 'gwei')
#         })
#         signed_tx = w3.eth.account.sign_transaction(transaction, private_key=Eth_private_key)
#         tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
#         tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)  # Wait for transaction receipt
#         gas_used = tx_receipt['gasUsed']
#         tx_update = tx_receipt['transactionHash'].hex()
#         print(' ')
#         print('Train completed, model update Info:')
#         print(f'    Tx: {tx_update}')
#         print(f'    Gas: {gas_used} Wei')
#         print('-' * 75)
#         return tx_update
#     except ValueError as e:
#         print(f"Error occurred: {e}")
#         if "nonce" in str(e):
#             print("Retrying transaction with updated nonce...")
#             return update_model_Tx(r, Hash_model, hash_ct_epk, Task_id, project_id)  # Recursive retry
#         raise e


def update_model_Tx(r, CID, hash_ct_epk, Task_id, project_id):
    """Submits the IPFS CID to the blockchain"""
    nonce = w3.eth.get_transaction_count(ETH_address)
    transaction = contract.functions.updateModel(r, CID, hash_ct_epk, Task_id, project_id).build_transaction({
        'from': ETH_address, 'nonce': nonce, 'gas': 2000000, 'gasPrice': w3.to_wei('50', 'gwei')
    })
    signed_tx = w3.eth.account.sign_transaction(transaction, Eth_private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    return w3.eth.wait_for_transaction_receipt(tx_hash)['transactionHash'].hex()

def listen_for_feedback(current_round, client_address, blocks_lookback=10):
    global w3, contract
    print(f"DEBUG: Listening for Feedback (R: {current_round})")
    latest_block = w3.eth.block_number
    start_block = max(0, latest_block - blocks_lookback)
    feedback_filter = contract.events.FeedbackProvided.create_filter(fromBlock=start_block)
    while True:
        feedback_events = feedback_filter.get_all_entries()  # Fetch events from the filter
        for feedback in feedback_events:
            event_client_address = feedback['args']['clientAddress']
            event_round = feedback['args']['round']
            if event_client_address == client_address and event_round == current_round:
                accepted = feedback['args']['accepted']
                task_id = feedback['args']['taskId']
                tx_hash = feedback['transactionHash'].hex()
                project_id = feedback['args']['project_id']
                T = feedback['args']['terminate']
                score_change = feedback['args']['scoreChange']
                server_addr = feedback['args']['serverId']
                print('Feedback Info:')
                print(f'    Tx: {tx_hash}')
                print(f'    Status: {accepted}')
                print(f'    Round: {event_round}')
                print(f'    Score: {score_change}')
                print(f'    Time: {time.strftime("%Y-%m-%d %H:%M:%S (UTC)", time.gmtime())}')
                print(f'    Server address: {server_addr}')
                print('-' * 75)
                return project_id, T, score_change
        time.sleep(1)
    return None, False, 0

if __name__ == "__main__":
    start_time = time.time()
    print("--- CLIENT START ---")
    
    # 1. Connect to Ganache
    try:
        ganache_url = "http://127.0.0.1:7545"
        w3 = Web3(Web3.HTTPProvider(ganache_url))
        print("Client connected to blockchain (Ganache) successfully\n")
    except Exception as e:
        print("Exception connecting to Ganache:", e)
        sys.exit(1)

    # 2. Parse CLI Arguments
    if len(sys.argv) < 6:
        print("Usage: python client.py <private_key> <contract_address> <num_epochs> <dataset_type> <HE_algorithm>")
        sys.exit(1)

    Eth_private_key, contract_address = sys.argv[1], sys.argv[2]
    num_epochs, dataset_type, HE_algorithm = int(sys.argv[3]), sys.argv[4], sys.argv[5]

    account = Account.from_key(Eth_private_key)
    ETH_address = account.address
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(script_dir)

    with open(main_dir + "/contract/contract-abi.json", "r") as f:
        contract_abi = json.load(f)
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)

    # 3. Load HE Configuration if applicable
    HE_config_with_key = None
    if HE_algorithm in ['CKKS', 'BFV']:
        with open(main_dir + f'/participant/keys/{HE_algorithm}_with_priv_key.pkl', "rb") as f:
            HE_config_with_key = ts.context_from(pickle.load(f))

    # 4. Listen for Project Registration and Join
    Tx_r, project_id, server_address, cnt_clients, initial_model_cid, hash_pubkeys = listen_for_projcet()
    
    esk_a = ECC.generate(curve='p256')
    epk_a_bytes = bytes(esk_a.public_key().export_key(format='PEM'), 'utf-8')
    hash_epk = hash_data(epk_a_bytes)
    ini_score, Tx_reg, registered_id_p = register_client(hash_epk, project_id)

    # 5. Establish Off-chain Connection for Control Signaling
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Server off-chain listener binds to localhost:65432 (see server/server.py)
    client_socket.connect(('127.0.0.1', 65432))
    client_socket.send(json.dumps({"msg_type": "Hello!", "Data": ETH_address}).encode('utf-8'))

    # Fetch Session ID
    session_id = client_socket.recv(4096).decode('utf-8').split(':')[-1].strip()

    # Perform Key Exchange
    client_socket.send(json.dumps({"msg_type": "pubkeys please", "Data": session_id}).encode('utf-8'))
    received_keys = json.loads(client_socket.recv(4096).decode('utf-8'))
    epk_b_pem, kpk_b = bytes.fromhex(received_keys['epk_b_pem']), bytes.fromhex(received_keys['kpk_b'])
    
    ct, ss_k = ml_kem_768.encrypt(kpk_b)
    hash_ct_epk_a = hash_data(ct + epk_a_bytes)
    client_socket.send(json.dumps({"msg_type": "none", "epk_a_pem": epk_a_bytes.hex(), "ciphertext": ct.hex()}).encode('utf-8'))

    # Derive Symmetric Keys
    SS = ss_k + key_agreement(eph_priv=esk_a, eph_pub=ECC.import_key(epk_b_pem), kdf=kdf)
    salt_a = salt_s = b'\0' * 32
    Root_key = HKDF(SS, 32, salt_a, SHA384, 1)
    chain_key, Model_key = HKDF(Root_key, 32, salt_s, SHA384, 2)

    # 6. Main Federated Learning Rounds Loop
    while True:
        r, Task_id, global_cid_onchain, hash_keys, project_id_received, server_eth_addr, D_t = listen_for_task(240)
        if Task_id == 0 or task_completed(Task_id, project_id_received): break

        # --- IPFS PAYLOAD RETRIEVAL ---
        # Request global model FIRST (it's encrypted with current/old key)
        # Server may respond with "not ready" if it hasn't prepared the wrapped model yet.
        global_cid = None
        for _ in range(40):  # ~20s total with sleep(0.5)
            client_socket.send(json.dumps({"msg_type": "Global model please", "Data": session_id}).encode('utf-8'))
            cid_data = client_socket.recv(4096)
            if not cid_data:
                print("DEBUG CLIENT: Empty response while requesting Global model CID; retrying...")
                time.sleep(0.5)
                continue
            try:
                resp = json.loads(cid_data.decode('utf-8'))
            except json.JSONDecodeError:
                print(f"DEBUG CLIENT: Non-JSON response while requesting Global model CID: {cid_data[:80]!r} ... retrying")
                time.sleep(0.5)
                continue

            if resp.get("msg_type") == "Global model CID" and resp.get("CID"):
                global_cid = resp["CID"]
                break
            if resp.get("msg_type") == "Global model not ready":
                time.sleep(0.5)
                continue
            print(f"DEBUG CLIENT: Unexpected response while requesting Global model CID: {resp}; retrying...")
            time.sleep(0.5)

        if not global_cid:
            print("DEBUG CLIENT: Failed to obtain Global model CID after retries; aborting this round.")
            continue
        
        print(f"DEBUG: Fetching Global Model from IPFS CID: {global_cid}")
        # Fetch, Decompress, and Unwrap
        model_blob = get_from_Ipfs(global_cid)
        unwrapped_pkg = unwrap_files(model_blob)
        
        # Safely extract AES ciphertext and decrypt/unwrap it.
        global_model_ct = unwrapped_pkg.get('global_model.enc')
        if global_model_ct is None:
            # Defensive: missing expected entry — log and skip this task iteration
            print(f"DEBUG: 'global_model.enc' not found in server package for CID {global_cid}. Keys in package: {list(unwrapped_pkg.keys())}")
            continue

        # DEBUG: show model key fingerprint so you can compare with server logs
        try:
            print(f"DEBUG: Using Model_key len={len(Model_key)} hex-prefix={Model_key.hex()[:16]} hash={hash_data(Model_key)[:16]}")
        except Exception:
            print("DEBUG: Unable to print Model_key fingerprint")

        # Attempt decryption + tar-unpack with robust error handling to avoid tarfile.ReadError crash
        try:
            decrypted_wrapper = AES_decrypt_data(Model_key, global_model_ct)

            # Quick sanity check: tar "ustar" magic located at offset 257 (standard tar). If missing, warn.
            if len(decrypted_wrapper) > 264 and decrypted_wrapper[257:262] != b'ustar':
                print("DEBUG: Decrypted data does not contain tar ustar magic -> likely wrong key or corrupted data")
                # write raw decrypted bytes for inspection
                debug_dir = os.path.join(main_dir, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                open(os.path.join(debug_dir, f"decrypted_non_tar_r{r}_{ETH_address[:8]}.bin"), "wb").write(decrypted_wrapper)
                raise ValueError("Decrypted payload missing tar header")

            dec_payload = unwrap_files(decrypted_wrapper)
        except Exception as e:
            # Save debugging artifacts for offline inspection and continue instead of crashing
            print(f"DEBUG: Failed to decrypt/unpack global model (CID={global_cid}) — error: {e}")
            try:
                debug_dir = os.path.join(main_dir, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                dump_path = os.path.join(debug_dir, f"failed_global_r{r}_{ETH_address[:8]}.bin")
                open(dump_path, "wb").write(global_model_ct or b"")
                print(f"DEBUG: Wrote raw ciphertext to {dump_path}")
            except Exception:
                pass
            continue

        # Asymmetric Ratcheting AFTER decrypting global model
        # (Global model was encrypted with old key, so we decrypt first, then ratchet)
        if hash_keys != 'None':
            print(f"DEBUG CLIENT: Asymmetric ratcheting triggered for round {r}")
            # Generate new ephemeral ECDH key
            esk_a_new = ECC.generate(curve='p256')
            epk_a_new_bytes = bytes(esk_a_new.public_key().export_key(format='PEM'), 'utf-8')
            
            # Request new server public keys
            client_socket.send(json.dumps({"msg_type": "update pubkeys", "Data": session_id}).encode('utf-8'))
            
            # Receive new server public keys
            received_keys_new = json.loads(client_socket.recv(4096).decode('utf-8'))
            epk_b_pem_new, kpk_b_new = bytes.fromhex(received_keys_new['epk_b_pem']), bytes.fromhex(received_keys_new['kpk_b'])
            
            # Perform new key exchange
            ct_new, ss_k_new = ml_kem_768.encrypt(kpk_b_new)
            hash_ct_epk_a_new = hash_data(ct_new + epk_a_new_bytes)
            client_socket.send(json.dumps({
                "msg_type": "none", 
                "epk_a_pem": epk_a_new_bytes.hex(), 
                "ciphertext": ct_new.hex()
            }).encode('utf-8'))
            
            # Derive new Root Key
            SS_new = ss_k_new + key_agreement(eph_priv=esk_a_new, eph_pub=ECC.import_key(epk_b_pem_new), kdf=kdf)
            Root_key = HKDF(SS_new, 32, salt_a, SHA384, 1)
            
            # Update keys: derive new chain_key and Model_key from new Root_key
            chain_key, Model_key = HKDF(Root_key, 32, salt_s, SHA384, 2)
            
            # Update hash_ct_epk_a for next blockchain transaction
            hash_ct_epk_a = hash_ct_epk_a_new
            
            print(f"DEBUG CLIENT: Asymmetric ratcheting complete. New Root Key derived.")
        
        if r != 1 and HE_algorithm != 'None':
            print("DEBUG dec_payload keys:", dec_payload.keys())
            global_model_data, meta = deserialize_data(dec_payload['global_HE_model.bin'], HE_config_with_key)
            global_model = HE_decrypt_model(global_model_data, Local_model, HE_config_with_key, HE_algorithm, meta)
        else:
            global_model = dec_payload.get('global_model.pth')

        # 7. Local Training
        round_start = time.time()
        # Ensure we have a valid global_model before training
        if 'global_model' not in locals() or global_model is None:
            print(f"DEBUG: No global_model available for round {r}; skipping training and continuing.")
            continue        
        Local_model = train_model.train(global_model, num_epochs, dataset_type, mu=0.1)
        
        # 8. IPFS PAYLOAD UPLOAD ---
        if HE_algorithm != 'None':
            enc_m, meta = HE_encrypt_model(Local_model, HE_config_with_key, HE_algorithm)
            local_bytes = serialize_data(enc_m, meta, HE_algorithm)
            m_filename = f'local_HE_model_{ETH_address}.bin'
        else:
            local_bytes = pickle.dumps(Local_model.state_dict())
            m_filename = f'local_model_{ETH_address}.pth'

        local_hash = hash_data(local_bytes)
        info_json = json.dumps({'Model hash': local_hash, 'Round': r, 'Task': Task_id}, indent=4).encode('utf-8')
        
        # Wrap, Encrypt, and Upload to IPFS
        wrapped_update = wrapfiles(('Local_model_info.json', info_json), (m_filename, local_bytes))
        update_ct = AES_encrypt_data(Model_key, wrapped_update)
        signed_ct = sign_data(update_ct, Eth_private_key, w3)
        final_pkg = wrapfiles(('signature.bin', signed_ct), ('Local_model.enc', update_ct))
        
        local_cid = upload_to_Ipfs(final_pkg) #

        # 9. Update Blockchain and Notify Server with CID
        update_model_Tx(r, local_cid, hash_ct_epk_a, Task_id, project_id_received)
        client_socket.send(json.dumps({
            "msg_type": "local model update CID", 
            "Data": session_id, 
            "CID": local_cid
        }).encode('utf-8'))

        # 10. Handle Feedback and Ratchet
        proj_fb, T, score = listen_for_feedback(r, ETH_address)
        if T: break

        # NOTE: Do NOT perform symmetric ratcheting here.
        # The server encrypts the next global model with the client's current Model_key.
        # Performing symmetric ratcheting at the end of the round causes the client to
        # use a different key than the server when fetching the next global model,
        # producing decryption failures. Perform symmetric ratcheting only after
        # successfully decrypting the next global model (or when explicitly signalled).
        print("--------------------- ROUND END ---------------------")
        print("Client finished. Total Runtime:", time.time() - start_time)