"""
Created on Tue Jan 2 14:18:41 2024
@author: HIGHer
"""
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account

from collections import namedtuple
from pqcrypto.kem import ml_kem_768

from Crypto.Protocol.DH import key_agreement
from Crypto.Protocol.KDF import HKDF
from Crypto.PublicKey import ECC
from Crypto.Hash import SHA384
from Crypto.Util.number import *

import socket, pickle
import tenseal as ts
import os, sys, time, json
import csv # New import for metrics logging
from threading import Thread, Event
from queue import Queue, Empty
from subprocess import Popen # New import for launching the HE test script
import torch

import aggregate
from simple_cnn_config import SimpleCNN
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
import runpy, io, contextlib, threading, json


def log_server_metrics(metrics: dict, project_id: int, tag: str = "server"):
    """
    Append a server-side metrics row to metrics/server_<project_id>.csv.
    Call this after you finish aggregation/publish task/upload to IPFS etc.
    metrics: dict of metric_name -> value
    tag: short string to indicate the phase (e.g. 'aggregate', 'publish', 'ipfs_upload')
    """
    try:
        # Determine the project root directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_dir = os.path.dirname(script_dir)
        metrics_dir = os.path.join(main_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_file = os.path.join(metrics_dir, f'server_{project_id}.csv')
        write_header = not os.path.exists(metrics_file)
        
        # Prepare data row
        row_data = [int(time.time()), project_id, tag]
        metric_keys = list(metrics.keys())
        metric_values = [metrics[k] for k in metric_keys]

        with open(metrics_file, 'a', newline='') as sf:
            writer = csv.writer(sf)
            if write_header:
                header = ['timestamp', 'project_id', 'phase'] + metric_keys
                writer.writerow(header)
            
            row = row_data + metric_values
            writer.writerow(row)
    except Exception as e:
        print(f"DEBUG SERVER: Failed to write server metrics: {e}")

# Global Variables (Initialized in __main__)
w3 = None
contract_address = None
contract_abi = None
Eth_address = None
Eth_private_key = None
ecdh = None
kyber = None
salt_a = None
salt_s = None
clients_dict = {}
wraped_global_model = None
model_info = {}


def generate_keys():
    KeyPair = namedtuple('KeyPair', ['pk', 'sk'])
    ecdh_priv = ECC.generate(curve='p256')
    ecdh_pub = bytes(ecdh_priv.public_key().export_key(format='PEM'), 'utf-8')
    kyber_pub, kyebr_priv = ml_kem_768.generate_keypair() 
    ecdh_keys = KeyPair(pk =ecdh_pub, sk =ecdh_priv)
    kyber_keys = KeyPair(pk=kyber_pub, sk=kyebr_priv)
    print(f"DEBUG SERVER: New ECDH/Kyber keys generated.") # DEBUG
    print(f"DEBUG SERVER: ECDH Pub (PEM bytes) length: {len(ecdh_pub)}") # DEBUG
    return ecdh_keys, kyber_keys


# ! This is the orignal one
# def register_project(project_id, cnt_clients_req, hash_init_model, hash_keys):
#     contract = w3.eth.contract(address=contract_address, abi=contract_abi)
#     if not contract.functions.isProjectTerminated(project_id).call():
#         print(f"DEBUG SERVER: Registering project {project_id} on-chain. Model Hash: {hash_init_model[:10]}...") # DEBUG
#         for attempt in range(3):
#             try:
#                 nonce = w3.eth.get_transaction_count(Eth_address, 'pending')
#                 transaction = contract.functions.registerProject(
#                     project_id, cnt_clients_req, hash_init_model, hash_keys
#                 ).build_transaction({
#                     'from': Eth_address,
#                     'gas': 2000000,
#                     'gasPrice': w3.to_wei('50', 'gwei'),
#                     'nonce': nonce,
#                 })
#                 signed_transaction = w3.eth.account.sign_transaction(transaction, Eth_private_key)
#                 tx_sent = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)
#                 receipt = w3.eth.wait_for_transaction_receipt(tx_sent)
#                 gas_used=receipt['gasUsed']
#                 tx_registration = receipt['transactionHash'].hex()
#                 print(f'Project Registeration on contract:')
#                 print(f'     Tx_hash: {tx_registration}')
#                 print(f'     Gas: {gas_used} Wei')
#                 print(f'     Project ID: {project_id}')
#                 print(f'     required client count: {cnt_clients_req}') 
#                 print(f'     Initial model hash: {hash_init_model}')
#                 print(f'     Pubic keys hash: {hash_keys}')
#                 print('-'*75)
#                 return tx_registration
#             except ValueError as e:
#                 print(f"Error: {e}. Retrying transaction...")
#                 time.sleep(2)
#         raise Exception("Transaction failed after retries.")
#     else:
#         print(f"Project {project_id} is already completed.")
#         sys.exit()

def register_project(project_id, cnt_clients_req, model_bytes, hash_keys):
    """
    Uploads the initial model to IPFS and registers the project on-chain 
    using the resulting CID as the model identifier.
    """
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)
    
    if contract.functions.isProjectTerminated(project_id).call():
        print(f"Project {project_id} is already completed.")
        sys.exit()
    
    # --- IPFS STEP ---
    print(f"DEBUG SERVER: Uploading initial model to IPFS...")
    model_cid = upload_to_Ipfs(model_bytes)
    if not model_cid:
        raise Exception("Failed to upload initial model to IPFS. Ensure IPFS daemon is running.")
    print(f"DEBUG SERVER: IPFS Upload Successful. CID: {model_cid}")
    
    # --- On-chain registration ---
    print(f"DEBUG SERVER: Registering project {project_id} on-chain with CID.")
    for attempt in range(3):
        try:
            nonce = w3.eth.get_transaction_count(Eth_address, 'pending')
            transaction = contract.functions.registerProject(
                project_id, cnt_clients_req, model_cid, hash_keys
            ).build_transaction({
                'from': Eth_address,
                'gas': 2000000,
                'gasPrice': w3.to_wei('50', 'gwei'),
                'nonce': nonce,
            })
            
            signed_transaction = w3.eth.account.sign_transaction(transaction, Eth_private_key)
            tx_sent = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_sent)
            
            tx_registration = receipt['transactionHash'].hex()
            print(f'Project Registration on contract (IPFS Mode):')
            print(f'     Tx_hash: {tx_registration}')
            print(f'     Project ID: {project_id}')
            print(f'     Model IPFS CID: {model_cid}') 
            print(f'     Public keys hash: {hash_keys}')
            print('-'*75)
            return tx_registration, model_cid
            
        except ValueError as e:
            print(f"Error: {e}. Retrying transaction...")
            time.sleep(2)
    
    raise Exception("Transaction failed after retries.")




def wait_for_clients(event_queue, stop_event, poll_interval=1):
    """
    Robust on-chain event poller for ClientRegistered events.
    Uses an inclusive from_block and dedup set (txHash+logIndex) so events
    emitted in the same block are not missed.
    """
    global w3, contract_address, contract_abi
    print('DEBUG SERVER: Event listener thread started (ClientRegistered events)')

    if geth_poa_middleware not in w3.middleware_onion:
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)

    contract = w3.eth.contract(address=contract_address, abi=contract_abi)

    # Start slightly before tip to reduce race windows
    last_processed_block = max(0, w3.eth.block_number - 1)

    # Keep a set of processed event identifiers to avoid duplicates
    processed_events = set()

    while not stop_event.is_set():
        try:
            current_block = w3.eth.block_number
            print(f"DEBUG SERVER: wait_for_clients: last_processed={last_processed_block}, current={current_block}, queue_size={event_queue.qsize()}")

            # Query inclusive range [last_processed_block, current_block] so we do not miss events in the same block
            from_block = last_processed_block
            to_block = current_block

            events = []
            try:
                # web3.py's .getLogs() is usually the preferred, more efficient way
                events = contract.events.ClientRegistered().getLogs(fromBlock=from_block, toBlock=to_block)
            except Exception as ex_getlogs:
                # fallback to filter-based retrieval if getLogs fails
                try:
                    event_filter = contract.events.ClientRegistered.create_filter(fromBlock=from_block, toBlock=to_block)
                    events = event_filter.get_all_entries()
                    try:
                        w3.eth.uninstall_filter(event_filter.filter_id)
                    except Exception:
                        pass
                except Exception as ex_filter:
                    print(f"DEBUG SERVER: getLogs/filter failed: getLogs_err={ex_getlogs} filter_err={ex_filter}")
                    events = []

            if events:
                print(f"DEBUG SERVER: wait_for_clients: found {len(events)} ClientRegistered events in blocks {from_block}-{to_block}")
                for event in events:
                    # build a stable id for deduplication (txHash + logIndex)
                    txh = event.get('transactionHash')
                    if isinstance(txh, bytes):
                        txh = txh.hex()
                    log_index = event.get('logIndex', None)
                    ev_id = (txh, log_index)
                    if ev_id in processed_events:
                        continue
                    processed_events.add(ev_id)

                    try:
                        event_queue.put(event)
                        addr = event['args'].get('clientAddress') if 'args' in event and event['args'] else event['args']
                        print(f"DEBUG SERVER: Client registration event queued at block {event.get('blockNumber', to_block)}: {addr}")
                    except Exception as q_ex:
                        print(f"DEBUG SERVER: Failed to put event into queue: {q_ex}")
            else:
                print(f"DEBUG SERVER: No ClientRegistered events in blocks {from_block}-{to_block}")

            # Advance pointer to current tip (we used inclusive query)
            last_processed_block = to_block

            time.sleep(poll_interval)

        except Exception as e:
            print(f"DEBUG SERVER: Error in registration event listener: {e}")
            time.sleep(poll_interval)


def finish_tash(task_id, project_id):
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)
    print(f"DEBUG SERVER: Finishing Task {task_id}") # DEBUG
    nonce = w3.eth.get_transaction_count(Eth_address)
    
    # Build the transaction with task_id and project_id
    transaction = contract.functions.finishTask(task_id, project_id).build_transaction({
        'from': Eth_address,
        'gas': 2000000,
        'gasPrice': w3.to_wei('50', 'gwei'),
        'nonce': nonce,
    })
    # Sign and send the transaction
    signed_transaction = w3.eth.account.sign_transaction(transaction, Eth_private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)
    # Wait for the receipt
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    gas_used = receipt['gasUsed']
    tx_publish = receipt['transactionHash'].hex()
    print(f'Task terminated:')
    print(f'     Tx_hash: {tx_publish}')
    print(f'     Gas: {gas_used} Wei')
    print(f'     Task ID: {task_id}')
    print(f'     Project ID: {project_id}')
    print('-' * 75)

def finish_project(project_id):
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)
    print(f"DEBUG SERVER: Finishing Project {project_id}") # DEBUG
    nonce = w3.eth.get_transaction_count(Eth_address)
    # Build the transaction with project_id
    transaction = contract.functions.finishProject(project_id).build_transaction({
        'from': Eth_address,
        'gas': 2000000, 
        'gasPrice': w3.to_wei('50', 'gwei'),
        'nonce': nonce,
    })
    # Sign and send the transaction
    signed_transaction = w3.eth.account.sign_transaction(transaction, Eth_private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)
    # Wait for the receipt
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    gas_used = receipt['gasUsed']
    tx_publish = receipt['transactionHash'].hex()
    print(f'Project terminated:')
    print(f'     Tx_hash: {tx_publish}')
    print(f'     Gas: {gas_used} Wei')
    print(f'     Project ID: {project_id}')
    print('-' * 75)


def publish_task(r, Model_CID, hash_keys, Task_id, project_id, D_t):
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)
    print(f"DEBUG SERVER: Publishing Task R:{r}. Model Hash: {Model_CID[:10]}..., Keys Hash: {hash_keys}") # DEBUG
    nonce = w3.eth.get_transaction_count(Eth_address)
    transaction = contract.functions.publishTask(r,Model_CID, hash_keys, Task_id, project_id, D_t).build_transaction({
        'from': Eth_address,
        'gas': 2000000,
        'gasPrice': w3.to_wei('50', 'gwei'),
        'nonce': nonce,
    })
    signed_tx = w3.eth.account.sign_transaction(transaction, Eth_private_key)
    tx_sent = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_sent)
    gas_used=receipt['gasUsed']
    tx_publish = receipt['transactionHash'].hex()
    print('')
    print(f'Task published round {r}:')
    print(f'     Tx_hash: {tx_publish}')
    print(f'     Gas: {gas_used} Wei')
    print(f'     Task ID: {Task_id}')
    print('-'*75)
    return tx_publish


def listen_for_updates(event_filter, event_queue):
    print('DEBUG SERVER: ModelUpdate listener thread started') # DEBUG
    # Add PoA middleware for Ganache (if needed)
    if geth_poa_middleware not in w3.middleware_onion:
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    # Create an instance of the contract with the ABI and address
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)
    # NOTE: Using a static filter creator can miss events; relying on the main loop's logic to handle event processing.
    # The current implementation uses a simple latest block filter, but the main loop's logic handles state.
    event_filter = contract.events.ModelUpdated.create_filter(fromBlock="latest")
    
    # Loop to listen for events
    while True:
        try:
            events = event_filter.get_new_entries()
            if events:
                for event in events:
                    event_queue.put(event)
                    print(f"DEBUG SERVER: ModelUpdate event caught for {event['args']['clientAddress']}") # DEBUG
            time.sleep(1)
        except Exception as e:
            print(f"DEBUG SERVER: Error in ModelUpdate listener: {e}")
            time.sleep(2)


def feedback_TX(r, task_id, project_id, client_address, feedback_score, T):
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)
    print(f"DEBUG SERVER: Sending Feedback (R:{r}) for {client_address}. Score: {feedback_score}") # DEBUG
    for attempt in range(3):
        try:
            nonce = w3.eth.get_transaction_count(Eth_address, 'pending') # Fetch the current pending nonce
            transaction = contract.functions.provideFeedback(r, task_id, project_id, client_address, feedback_score, T 
            ).build_transaction({
                'from': Eth_address,
                'gas': 2000000,
                'gasPrice': w3.to_wei('50', 'gwei'),
                'nonce': nonce,
            })
            # Sign the transaction
            signed_transaction = w3.eth.account.sign_transaction(transaction, Eth_private_key)
            tx_sent = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_sent)
            gas_used = receipt['gasUsed']
            tx_feedback = receipt['transactionHash'].hex()
            # Print transaction details
            print(f'Feedback:')
            print(f'           Client address: {client_address}')
            print(f'           Tx_hash: {tx_feedback}')
            print(f'           Gas: {gas_used} Wei')
            print(f'           Task ID: {task_id}')
            print(f'           Score: {feedback_score}')
            print('-'*75)
            return tx_feedback
        except ValueError as e:
            print(f"Transaction failed: {e}. Retrying...")
            time.sleep(2) 
    raise Exception("Feedback transaction failed after retries.")

def analyze_model (Local_model,Task_id,project_id_update):
    res=True
    Feedback_score=1
    return res, Feedback_score


def establish_root_key(client_socket,clients_dict,ecdh,kyber,salt_a,session_id):
    matching_addr = [address for address, details in clients_dict.items() if details.get("Session ID") == session_id] # find eth addr based Session ID
    if not matching_addr:
        print('DEBUG SERVER: Client session ID not found in registered clients.') # DEBUG
        client_socket.close()
        return None
    
    client_addr = matching_addr[0]
    print(f"DEBUG SERVER: Key exchange started for {client_addr}") # DEBUG

    msg_keys={'epk_b_pem':(ecdh.pk).hex(), 'kpk_b':(kyber.pk).hex()}
    client_socket.sendall(json.dumps(msg_keys).encode('utf-8'))
    
    data = client_socket.recv(4096).decode('utf-8')
    if data is None:
        print(f"DEBUG SERVER: Failed to receive client keys/ciphertext.") # DEBUG
        client_socket.close()
        return None
        
    received_data= json.loads(data) # Process the received Json data construct root, chain and model keys
    epk_a_pem = bytes.fromhex(received_data['epk_a_pem'])
    ct = bytes.fromhex(received_data['ciphertext']) 
    print(f"DEBUG SERVER: Received ct len: {len(ct)}, epk_a_pem len: {len(epk_a_pem)}") # DEBUG

    epk_a = ECC.import_key(epk_a_pem)
    ss_e = key_agreement(eph_priv=ecdh.sk, eph_pub=epk_a, kdf=kdf)
    ss_k = ml_kem_768.decrypt(kyber.sk, ct)
    SS = ss_k + ss_e
    Root_key= HKDF(SS, 32, salt_a, SHA384, 1)
    
    clients_dict[client_addr]['Hash_ct_epk_a']=hash_data(ct +epk_a_pem) 
    clients_dict[client_addr]['Root key'] = Root_key.hex()
    print(f"DEBUG SERVER: Root Key established for {client_addr}. Hash(ct||epk): {clients_dict[client_addr]['Hash_ct_epk_a'][:10]}...") # DEBUG
    return Root_key


def offchain_listener(server_socket):
    print('DEBUG SERVER: Off-chain listener thread started') # DEBUG
    while True:
        try:
            client_socket, client_address = server_socket.accept()
            print(f"DEBUG SERVER: New off-chain connection from {client_address}")
            client_thread = Thread(
                target=handle_offchain_client,
                args=(client_socket,),
                daemon=True
            )
            client_thread.start()
        except Exception as e:
            print(f"DEBUG SERVER: Error accepting connection: {e}") # DEBUG
            time.sleep(1)


# ! THis is the previos    
def handle_offchain_client(client_socket):
    global salt_a, salt_s, ecdh, kyber
    global wraped_global_model, model_info
    global clients_dict

    eth_address = None
    max_registration_attempts = 10

    while True:
        try:
            data_raw = client_socket.recv(4096)
            if not data_raw:
                print(f"DEBUG: Client {eth_address or 'Unknown'} disconnected.")
                break

            recv_msg = json.loads(data_raw.decode('utf-8'))
            msg_type = recv_msg.get("msg_type")

            print(f"DEBUG: Received message type: {msg_type}")

            # ------------------------------------------------
            # 1️⃣ HELLO + REGISTRATION CHECK
            # ------------------------------------------------
            if msg_type == 'Hello!':
                eth_address = recv_msg["Data"]

                is_registered = False
                for attempt in range(max_registration_attempts):
                    if eth_address in clients_dict:
                        is_registered = True
                        break
                    print(f"DEBUG: Waiting for blockchain registration...")
                    time.sleep(0.5)

                if not is_registered:
                    client_socket.send(
                        "You haven't registered on blockchain.".encode('utf-8')
                    )
                    continue

                session_id = clients_dict[eth_address]['Session ID']
                client_socket.send(
                    f"Session ID:{session_id}".encode('utf-8')
                )
                print(f"DEBUG: Sent Session ID {session_id}")

            # ------------------------------------------------
            # 2️⃣ INITIAL PUBKEY EXCHANGE
            # ------------------------------------------------
            elif msg_type == "pubkeys please":
                session_id = int(recv_msg["Data"])

                root_key = establish_root_key(
                    client_socket,
                    clients_dict,
                    ecdh,
                    kyber,
                    salt_a,
                    session_id
                )

                if root_key:
                    eth_address = [
                        addr for addr, d in clients_dict.items()
                        if d["Session ID"] == session_id
                    ][0]

                    chain_key, model_key = HKDF(
                        root_key, 32, salt_s, SHA384, 2
                    )

                    clients_dict[eth_address]['Model key'] = model_key.hex()
                    clients_dict[eth_address]['Chain key'] = chain_key.hex()

                    print(f"DEBUG: Keys derived for {eth_address}")

            # ------------------------------------------------
            # 3️⃣ RATCHETING
            # ------------------------------------------------
            elif msg_type == 'update pubkeys':
                session_id = int(recv_msg["Data"])

                root_key = establish_root_key(
                    client_socket,
                    clients_dict,
                    ecdh,
                    kyber,
                    salt_a,
                    session_id
                )

                if root_key:
                    print(f"DEBUG: Asymmetric ratchet success.")

            # ------------------------------------------------
            # 4️⃣ GLOBAL MODEL REQUEST (IPFS CID VERSION)
            # ------------------------------------------------
            elif msg_type == 'Global model please':
                session_id = int(recv_msg["Data"])

                eth_addr = [
                    addr for addr, d in clients_dict.items()
                    if d["Session ID"] == session_id
                ][0]

                model_key = bytes.fromhex(
                    clients_dict[eth_addr]['Model key']
                )
                try:
                    print(f"DEBUG SERVER: Model key for {eth_addr[:10]}... len={len(model_key)} hex-prefix={model_key.hex()[:16]} hash={hash_data(model_key)[:16]}") 
                except Exception:
                    print("DEBUG SERVER: Failed to print model_key debug info")

                # If the main loop hasn't prepared the wrapped model yet, don't crash the socket handler.
                # This can happen if a client replays an old TaskPublished event and requests the model early.
                if not isinstance(wraped_global_model, (bytes, bytearray)) or len(wraped_global_model) == 0:
                    client_socket.send(
                        json.dumps({
                            "msg_type": "Global model not ready",
                            "error": "Server has not prepared wraped_global_model yet. Wait and retry."
                        }).encode('utf-8')
                    )
                    print("DEBUG SERVER: wraped_global_model is None -> sent 'not ready' response.")
                    continue

                try:
                    model_ct = AES_encrypt_data(
                        model_key,
                        wraped_global_model
                    )
                except Exception as e:
                    client_socket.send(
                        json.dumps({
                            "msg_type": "Global model error",
                            "error": f"Failed to encrypt global model: {repr(e)}"
                        }).encode('utf-8')
                    )
                    print(f"DEBUG SERVER: Failed to encrypt wraped_global_model: {e}")
                    continue

                signed_ct = sign_data(
                    model_ct,
                    Eth_private_key,
                    w3
                )

                pkg = wrapfiles(
                    ('signature.bin', signed_ct),
                    ('global_model.enc', model_ct)
                )
                cid = upload_to_Ipfs(pkg)
                if not cid:
                    client_socket.send(
                        json.dumps({
                            "msg_type": "Global model error",
                            "error": "Failed to upload encrypted global model package to IPFS"
                        }).encode('utf-8')
                    )
                    print("DEBUG SERVER: Failed to upload global model package to IPFS (cid=None)")
                    continue

                client_socket.send(
                    json.dumps({
                        "msg_type": "Global model CID",
                        "CID": cid
                    }).encode('utf-8')
                )

                print(f"DEBUG: Sent Global Model CID {cid}")

            # ------------------------------------------------
            # 5️⃣ LOCAL MODEL UPDATE (CID VERSION)
            # ------------------------------------------------
            elif msg_type == 'local model update CID':
                session_id = int(recv_msg["Data"])
                cid = recv_msg["CID"]

                eth_addr = [
                    addr for addr, d in clients_dict.items()
                    if d["Session ID"] == session_id
                ][0]

                blob = get_from_Ipfs(cid)
                unwrapped_pkg = unwrap_files(blob)
                model_info[eth_addr] = {
                    'model_data': unwrapped_pkg
                }

                print(f"DEBUG: Stored local update from {eth_addr}")

            else:
                print(f"DEBUG: Unknown message type: {msg_type}")

        except json.JSONDecodeError:
            print("DEBUG: JSON decode error")
            break

        except Exception as e:
            print(f"DEBUG: Unexpected error: {e}")
            break

    client_socket.close()


def display_he_test_results(repo_root):
    """Reads and prints the results of all HE decryption attempts from the log file."""
    metrics_file = os.path.join(repo_root, 'metrics', 'he_decrypt_attempts.csv')
    
    print("\n" + "="*75)
    print("HOMOMORPHIC ENCRYPTION DECRYPTION ATTEMPTS SUMMARY")
    print("="*75)
    
    if not os.path.exists(metrics_file):
        print(f"No results file found: {metrics_file}. Run with HE_algorithm set (e.g., CKKS) to generate a report.")
        return

    total_attempts = 0
    successful_decryptions = 0
    
    try:
        with open(metrics_file, 'r', newline='') as f:
            reader = csv.reader(f)
            # Skip header, but check if the file is empty first
            try:
                header = next(reader) 
            except StopIteration:
                print("Results file is empty.")
                return
            
            # Print column headers for the report
            print(f"{'Alg':<5} | {'Model File':<30} | {'Success':<7} | {'Params Changed':<16} | {'Time (s)':<10} | Error")
            print("-" * 75)
            
            for row in reader:
                total_attempts += 1
                try:
                    # Structure: 'timestamp','alg','model_file','ctx_file','success','error','params_changed','time_s'
                    alg = row[1]
                    model_file = os.path.basename(row[2])
                    success = row[4]
                    error = row[5]
                    params_changed = int(row[6])
                    time_s = float(row[7])
                    
                    if success.lower() == 'true':
                        successful_decryptions += 1
                        
                    print(f"{alg:<5} | {model_file:<30} | {success:<7} | {params_changed:<16} | {time_s:<10.4f} | {error}")
                except IndexError:
                    print(f"Warning: Skipping malformed row: {row}")

    except Exception as e:
        print(f"Error reading results file: {e}")

    print("-" * 75)
    print(f"Total Decryption Attempts Logged: {total_attempts}")
    print(f"Successful Decryptions: {successful_decryptions}")
    print("="*75)


if __name__ == "__main__":
    start_time = time.time()
    print("--- SERVER START ---") # DEBUG: Execution starts here
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(script_dir)
    
    try: 
        onchain_addr = "http://127.0.0.1:7545"
        w3 = Web3(Web3.HTTPProvider(onchain_addr))
        print("Server connected to blockchain (Ganache) successfully\n")
    except Exception as e:
        print("An exception occurred in connecting to blockchain (Ganache) or offchain:", e)
        exit()
        
    # --- Network Setup ---
    offcahin_addr = ('localhost', 65432)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    server_socket.bind(offcahin_addr)
    server_socket.listen()
    print(f"DEBUG SERVER: Off-chain socket bound to {offcahin_addr}") # DEBUG


    # --- CLI Argument Parsing ---
    if len(sys.argv) < 8:
        print("Usage: python server.py <Eth_private_key> <contract_address> <project_id> <round_count> <client_req> <Dataset_type> <HE_algorithm>")
        sys.exit(1)

    Eth_private_key=sys.argv[1]
    contract_address = sys.argv[2]
    project_id=int(sys.argv[3])
    round_count=int(sys.argv[4])
    client_req=int(sys.argv[5])
    Dataset_type=sys.argv[6]
    HE_algorithm=sys.argv[7]

    account = Account.from_key(Eth_private_key)
    Eth_address = account.address
    print(f"DEBUG SERVER: Server ETH Address: {Eth_address}") # DEBUG

    # main_dir is already calculated above
    with open(main_dir+"/contract/contract-abi.json", "r") as abi_file:
        contract_abi = json.load(abi_file)
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)

    # --- Load HE Keys ---
    HE_config_without_key = None
    if HE_algorithm=='CKKS':
        with open(main_dir + f'/server/keys/CKKS_without_priv_key.pkl', "rb") as f:
            serialized_without_key = pickle.load(f)
        HE_config_without_key = ts.context_from(serialized_without_key)
    elif HE_algorithm=='BFV':
        with open(main_dir + f'/server/keys/BFV_without_priv_key.pkl', "rb") as f:
            serialized_without_key = pickle.load(f)
        HE_config_without_key = ts.context_from(serialized_without_key)
    
    if HE_config_without_key:
        print(f"DEBUG SERVER: HE Config loaded for algorithm: {HE_algorithm}") # DEBUG
    else:
        print(f"DEBUG SERVER: No HE Config loaded (HE_algorithm is '{HE_algorithm}')")

    
    # --- Initial Key Generation & Project Registration ---
    # --- Key Generation remains the same ---
    ecdh, kyber = generate_keys()
    hash_pubkeys = hash_data(kyber.pk + ecdh.pk) 
    print(f"DEBUG SERVER: Initial Pub Keys Hash: {hash_pubkeys[:10]}...") 

    # --- Model Initialization ---
    Init_Global_model = SimpleCNN(Dataset_type)
    # We serialize the state_dict to bytes to prepare it for IPFS upload
    Init_Global_model_bytes = pickle.dumps(Init_Global_model.state_dict()) 
    Global_Model = Init_Global_model_bytes
    print(f"DEBUG SERVER: Initial Model serialized. Size: {len(Init_Global_model_bytes)} bytes.") 

    # --- Updated Registration (IPFS + Blockchain) ---
    start_register_time = time.time()

    # NOTE: We now pass the BYTES, not a pre-calculated hash.
    # The register_project function will upload to IPFS and return the CID.
    print("Entering the register_project")
    Tx_r, model_cid = register_project(project_id, client_req, Init_Global_model_bytes, hash_pubkeys)

    # --- Log Server Metrics with CID ---
    log_server_metrics({
        'registration_tx_time_s': round(time.time() - start_register_time, 4),
        'initial_model_ipfs_cid': model_cid,  # Store the CID for your research records
        'initial_keys_hash': hash_pubkeys
    }, project_id, tag='register_project')


    # --- Client Registration & Setup ---
    registered_cnt=0
    salt_a = salt_s = b'\0'*32
    registration_queue = Queue()
    stop_event = Event()
    print(f"DEBUG SERVER: Starting registration listeners...") # DEBUG

    Thread(target=wait_for_clients, args=(registration_queue, stop_event), daemon=True).start()
    Thread(target=offchain_listener, args=(server_socket,), daemon=True).start()

    # Wait for clients with a reasonable overall deadline and frequent small timeouts.
    overall_wait_seconds = max(60, client_req * 60)
    deadline = time.time() + overall_wait_seconds

    while registered_cnt < client_req and time.time() < deadline:
        try:
            event = registration_queue.get(timeout=5)
            eth_address = event['args']['clientAddress']

            session_id = registered_cnt + 1
            clients_dict[eth_address] = {
                'Session ID': session_id,
                'score': event['args']['initialScore'],
                'hash_epk': event['args']['hash_PubKeys'],
                'registration_tx': event['transactionHash'].hex(),
                'block_number': event['blockNumber']
            }

            registered_cnt += 1
            print(f"DEBUG SERVER: Client {registered_cnt}/{client_req} registered: {eth_address}")
        except Empty:
           # Normal: no event during timeout — print progress instead of error
           print(f"DEBUG SERVER: Waiting for client registrations... {registered_cnt}/{client_req} registered.")
           continue
        except Exception as e:
            print(f"DEBUG SERVER: Error while waiting for registration: {repr(e)}")
            continue

    if registered_cnt < client_req:
        print(f"DEBUG SERVER: Registration deadline reached ({registered_cnt}/{client_req} registered).")


    print("All clients registered or timeout reached.")
    stop_event.set()
    print('-'*75)
    
    Model_CID = model_cid 
    Models=[]
    task_info= {}
    ratchet_renge=2
    accuracy_list=[]

    # --- Main FL Rounds Loop ---
    for r in range(1,round_count+1):
        print(f"\n====================== ROUND {r} START ======================") # DEBUG
        Task_id=int(str(project_id)+str(r))
        task_info['Round number'] = r
        task_info['Model CID'] = Model_CID # Hash of the previous round's global model
        task_info['Project id'] = project_id
        task_info['Task id'] = Task_id
        task_info['Deadline Task'] = int(time.time()) + 100000

    # Publish Task
        hash_pubkeys='None'
        start_publish_time = time.time() # Start timer before publish TX
        
        if r%ratchet_renge==0:
            ecdh, kyber = generate_keys() 
            hash_pubkeys=hash_data(kyber.pk+ecdh.pk)
            Tx_p = publish_task(r, Model_CID, hash_pubkeys, Task_id, project_id, task_info['Deadline Task'])
            task_info['Publish Tx'] = Tx_p
            print(f"DEBUG SERVER: Asymmetric ratcheting triggered. New key hash: {hash_pubkeys[:10]}...") # DEBUG
        else: 
            Tx_p = publish_task(r, Model_CID, hash_pubkeys, Task_id, project_id, task_info['Deadline Task'])
            task_info['Publish Tx'] = Tx_p 
            print("DEBUG SERVER: Symmetric ratcheting only for this round.") # DEBUG

        # --- Log Task Publish Metrics ---
        log_server_metrics({
            'round': r,
            'publish_tx_time_s': round(time.time() - start_publish_time, 4),
            'Model CID': Model_CID,
            'ratcheting': 'asymm' if r%ratchet_renge==0 else 'symm'
        }, project_id, tag='publish')
        # -------------------------------------


        json_task_info = json.dumps(task_info, indent=4)
        
        # Wrap the global model (bytes) for encryption
        if r!=1 and HE_algorithm!='None':
            # aggregated_HE_model is bytes (from previous round)
            wraped_global_model=wrapfiles(('task_info.json',json_task_info.encode()), ('global_HE_model.bin',aggregated_HE_model))
        else:
            # Global_Model is bytes (Initial model or normal aggregated model)
            wraped_global_model=wrapfiles(('task_info.json',json_task_info.encode()), ('global_model.pth',Global_Model)) 
        
        print(f"DEBUG SERVER: Global Model wrapped. Size: {len(wraped_global_model)} bytes.") # DEBUG


        print(f"Start Round {r}: Waiting for local model updates...\n"+'='*20)
        event_queue = Queue()
        worker = Thread(target=listen_for_updates, args=({}, event_queue), daemon=True) # Pass an empty dict as placeholder for filter args
        worker.start()
        client_addrs=[]
        update_dict={}
        cnt_models=0
        T= False


        while True:
            if not event_queue.empty():
                print(f'Received {cnt_models+1} Local model update Tx:')
                event = event_queue.get()
                r_update = event['args']['round']
                Task_id_update = event['args']['taskId']
                tx_u = event['transactionHash'].hex()
                project_id_update= event['args']['project_id']
                client_addr = event['args']['clientAddress']
                Hash_local_model = event['args']['HashModel']
                Hash_ct_epk_a = event['args']['hash_ct_epk']
                
                print(f"DEBUG SERVER: Processing Tx from {client_addr}. Hash: {Hash_local_model[:10]}...") # DEBUG

                if r_update==r and Task_id_update==Task_id and project_id_update==project_id:
                    update_dict[client_addr]= {'round': r_update, 'Task id':Task_id_update , 
                                                 'Tx_u': tx_u, 'Project id':project_id_update, 
                                                 'Local model hash':Hash_local_model} 
                else:
                    print('DEBUG SERVER: Model update info not related to current round/project. Skipping.') # DEBUG
                    continue
                    
                print(json.dumps(update_dict[client_addr], indent=4))
                client_addrs.append(client_addr)
                
            # recieved model info and verification
                time.sleep(2)
                if HE_algorithm!='None':
                    time.sleep(9)
                
                # Retrieve model data received via off-chain listener
                Recieved_model=model_info.get(client_addr, {}).get('model_data')
                if Recieved_model is None:
                    print(f"DEBUG SERVER: ERROR: Model data missing for {client_addr} in model_info. Waiting...") # DEBUG
                    continue # Skip this client and wait for it to send data off-chain

                unwrapped_msg=Recieved_model
                signature=unwrapped_msg['signature.bin']
                local_model_ct=unwrapped_msg['Local_model.enc']
                
                print(f"DEBUG SERVER: Verifying signature using Tx: {tx_u}") # DEBUG
                verify_sign(signature, local_model_ct, pubKey_from_tx(tx_u,w3))
                
                Model_key=bytes.fromhex(clients_dict[client_addr]['Model key'])
                dec_wrapfile=AES_decrypt_data(Model_key,local_model_ct)
                unwraped=unwrap_files(dec_wrapfile)
                Local_model_info =unwraped['Local_model_info.json']
                
                print(f"DEBUG SERVER: Decrypted local model info: {Local_model_info.decode('utf-8')[:100]}...") # DEBUG

                if Hash_ct_epk_a!='None':
                    # During asymmetric ratcheting, the client sends a NEW hash_ct_epk_a
                    # Update the stored value if it's different (asymmetric ratcheting happened)
                    if clients_dict[client_addr]['Hash_ct_epk_a'] != Hash_ct_epk_a:
                        print(f"DEBUG SERVER: Updating Hash_ct_epk_a (asymmetric ratcheting detected). Old: {clients_dict[client_addr]['Hash_ct_epk_a'][:10]}..., New: {Hash_ct_epk_a[:10]}...")
                        clients_dict[client_addr]['Hash_ct_epk_a'] = Hash_ct_epk_a
                    else:
                        assert clients_dict[client_addr]['Hash_ct_epk_a'] == Hash_ct_epk_a , " off- and on-chain keys not match :(" 
                    print("DEBUG SERVER: Asymmetric Key Hash Check PASSED.") # DEBUG

                    if HE_algorithm == 'None':
                        local_model_bytes = unwraped[f'local_model_{client_addr}.pth']
                    else:
                        local_model_bytes = unwraped[f'local_HE_model_{client_addr}.bin']

                    # --- START INTEGRITY VERIFICATION (HASH-BASED) --- 
                    # We do NOT re-upload to IPFS here because gzip compression and chunking
                    # make the CID non-deterministic across uploads for the same logical data.
                    # Instead, we:
                    #   1. Parse the client's embedded metadata (Local_model_info.json)
                    #   2. Compare the SHA-256 hash of the recovered local_model_bytes
                    #      to the 'Model hash' field provided by the client.
                    try:
                        local_info = json.loads(Local_model_info.decode('utf-8'))
                        expected_hash = local_info.get("Model hash")
                        computed_hash = hash_data(local_model_bytes)
                        print(f"DEBUG SERVER: Expected model hash (from client): {expected_hash}")
                        print(f"DEBUG SERVER: Computed model hash (server-side): {computed_hash}")
                        assert expected_hash == computed_hash, \
                            f"Integrity Error! Client hash {expected_hash} != Server hash {computed_hash}"
                    except Exception as e:
                        raise AssertionError(f"Integrity verification failed for {client_addr}: {e}")
                    # --- END INTEGRITY VERIFICATION ---
                    
                    print(f"✅ Local model integrity verification PASSED for {client_addr}")
                    cnt_models += 1
                    
                    # Save the HE model to disk (for HE modes)
                    if HE_algorithm != 'None':
                        open(main_dir + f"/server/files/local models/local_HE_model_{client_addr}.bin",'wb').write(local_model_bytes)
                    try:
                       he_tool = os.path.join(main_dir, 'server', 'tools', 'he_decrypt_attempt.py')
                       ctx_file = os.path.join(main_dir, 'server', 'keys', f"{HE_algorithm}_without_priv_key.pkl")
                       model_file = os.path.join(main_dir, 'server', 'files', 'local models', f"local_HE_model_{client_addr}.bin")
                       # run the he_decrypt_attempt.py inside same interpreter, capture stdout/stderr
                       argv_backup = sys.argv.copy()
                       out_buf = io.StringIO()
                       err_buf = io.StringIO()
                       def _run_he():
                           try:
                               sys.argv = [he_tool, HE_algorithm, model_file, ctx_file, Dataset_type]
                               with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                                   runpy.run_path(he_tool, run_name="__main__")
                           finally:
                               sys.argv = argv_backup
                       th = threading.Thread(target=_run_he, daemon=True)
                       th.start()
                       th.join(timeout=60)   # adjust timeout as needed
                       stdout = out_buf.getvalue()
                       stderr = err_buf.getvalue()
                       # persist logs
                       log_dir = os.path.join(main_dir, 'metrics'); os.makedirs(log_dir, exist_ok=True)
                       open(os.path.join(log_dir, f"he_attempt_{client_addr}.out.log"), 'a', encoding='utf-8').write(stdout)
                       open(os.path.join(log_dir, f"he_attempt_{client_addr}.err.log"), 'a', encoding='utf-8').write(stderr)
                       # parse summary if printed
                       try:
                           summary = json.loads(stdout.strip().splitlines()[-1])
                       except Exception:
                           summary = None
                       if th.is_alive():
                           print(f"DEBUG SERVER: HE check timed out for {client_addr}")
                       else:
                           print(f"DEBUG SERVER: HE check finished for {client_addr}: {summary}")
                    except Exception as e:
                       print(f"DEBUG SERVER: Error running HE check in-process for {client_addr}: {e}")
                   # <-- INSERT END                    
                    # --- HE DECRYPTION TEST START (Non-blocking) ---
                    try:
                        # Define paths relative to the main directory
                        ctx_file = os.path.join(main_dir, 'server', 'keys', f"{HE_algorithm}_without_priv_key.pkl")
                        model_file = os.path.join(main_dir, 'server', 'files', 'local models', f"local_HE_model_{client_addr}.bin")
                        he_tool_path = os.path.join(main_dir, 'server', 'tools', 'he_decrypt_attempt.py')
                        
                        cmd = [sys.executable, he_tool_path, HE_algorithm, model_file, ctx_file, Dataset_type]
                        print(f"DEBUG SERVER: Launching HE decrypt attempt for {client_addr} with interpreter {sys.executable} (non-blocking).")
                        # ensure metrics dir exists and capture logs
                        log_dir = os.path.join(main_dir, 'metrics')
                        os.makedirs(log_dir, exist_ok=True)
                        out_log = os.path.join(log_dir, f"he_attempt_{client_addr}.out.log")
                        err_log = os.path.join(log_dir, f"he_attempt_{client_addr}.err.log")
                        # non-blocking launch using same interpreter and repo cwd so imports resolve
                        Popen(cmd, cwd=main_dir, stdout=open(out_log, "ab"), stderr=open(err_log, "ab")) 
                    except Exception as e:
                        print(f"DEBUG SERVER: Failed to launch he_decrypt_attempt: {e}")
                    # --- HE DECRYPTION TEST END ---
                    
                    Feedback_score=0
                    Tx_f=feedback_TX (r,Task_id, project_id, client_addr, Feedback_score, T) 
                    accuracy=0.0 # HE models need decryption to get final accuracy/weights

                
                print(f"DEBUG SERVER: Successfully processed local model {cnt_models}/{registered_cnt}.") # DEBUG

                if cnt_models==registered_cnt:
                    print("DEBUG SERVER: All required models received. Starting aggregation.") # DEBUG
                    
                    start_agg_time = time.time() # Start timer before aggregation
                    if HE_algorithm == 'None':

    
                        normal_aggregated, accuracy = aggregate.aggregate_models(client_addrs, HE_algorithm, Dataset_type)
                        
                        # 2. Convert the model object to bytes for IPFS
                        Global_Model_bytes = pickle.dumps(normal_aggregated.state_dict()) 
                        
                        # 3. UPLOAD BYTES: Get the CID instead of a local hash
                        # This CID is what you will send to clients via the blockchain
                        Model_CID = upload_to_Ipfs(Global_Model_bytes) 
                        
                        # 4. Update the local tracking variable for the round loop
                        Hash_model = Model_CID 
                        
                        # Optional: Save locally for debugging
                        torch.save(normal_aggregated.state_dict(), main_dir + '/server/files/global_model.pth')

                    else:
                            # 1️⃣ Aggregate HE model (returns encrypted bytes)
                            aggregated_HE_model = aggregate.aggregate_models(
                                client_addrs, HE_algorithm, Dataset_type
                            )

                            # 2️⃣ UPLOAD BYTES DIRECTLY: Get the CID for the actual model content
                            # FIX: Pass aggregated_HE_model (bytes), NOT he_model_path (string)
                            model_cid = upload_to_Ipfs(aggregated_HE_model)
                            
                            # 3️⃣ Update variables for tracking and task publishing
                            Model_CID = model_cid
                            Hash_model = model_cid  # Ensure the loop tracker uses the CID
                            
                            # 4️⃣ Update the Global_Model reference for the next round's wrapper
                            Global_Model = aggregated_HE_model 
                            
                            # 5️⃣ Optional: Save encrypted model locally for backup
                            he_model_path = main_dir + "/server/files/global_HE_model.bin"
                            with open(he_model_path, 'wb') as f:
                                f.write(aggregated_HE_model)

                            print(f"Global HE model CID: {model_cid}")
                            accuracy = 0.0  # until decrypted evaluation implemented
                    
                    
                    end_agg_time = time.time() # End timer

                    # --- Log Aggregation Metrics ---
                    log_server_metrics({
                        'round': r,
                        'aggregation_time_s': round(end_agg_time - start_agg_time, 4),
                        'num_models_aggregated': cnt_models,
                        'Model CID': Model_CID,
                        'accuracy': round(accuracy, 5) # Log accuracy if available
                    }, project_id, tag='aggregate')
                    # -------------------------------------
                    
                    print(f"DEBUG SERVER: Aggregation complete. New Global Hash: {Model_CID[:10]}...") # DEBUG
                    break
            else:
                time.sleep(2)

        # Symmetric ratcheting of model-key for each client at the end of round
        print("\nDEBUG SERVER: Starting symmetric ratcheting for all clients.") # DEBUG
        for addr in clients_dict:
            chain_key=bytes.fromhex(clients_dict[addr]['Chain key'])
            if r%ratchet_renge==0: # Check if asymmetric ratcheting happened this round
                Root_key=bytes.fromhex(clients_dict[addr]['Root key'])
                chain_key, Model_key = HKDF(Root_key, 32, salt_s, SHA384, 2)
                print(f"DEBUG SERVER: Client {addr[:6]}...: Used Root Key for symmetric derivation.") # DEBUG
            else:
                chain_key, Model_key = HKDF(chain_key, 32, salt_s, SHA384, 2)
                print(f"DEBUG SERVER: Client {addr[:6]}...: Used Chain Key for symmetric derivation.") # DEBUG

            clients_dict[addr]['Model key'] = Model_key.hex()
            clients_dict[addr]['Chain key'] = chain_key.hex()
            
        salt_s=(bytes_to_long(salt_s)+1).to_bytes(32, byteorder='big')
        salt_a=(bytes_to_long(salt_a)+1).to_bytes(32, byteorder='big')
        print(f"DEBUG SERVER: Ratcheting finished. New salt_s: {salt_s.hex()[:10]}...") # DEBUG
        
    finish_tash(Task_id,project_id) # transaction termination for recording on  blockchain
    finish_project(project_id)
    
    # --- FINAL HE DECRYPTION TEST REPORT ---
    # Give a short moment for any pending Popen process to finish logging
    print("DEBUG SERVER: Waiting for final HE decryption test logs...")
    
    # display_he_test_results(main_dir)
    
    print("--- SERVER FINISHED ---") # DEBUG: Execution ends here
    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"\n--- Total Program Runtime: {total_runtime:.2f} seconds ---")