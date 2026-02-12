import subprocess
import os
import argparse
import time


def run_client(client_eth_key, contract_address, client_path, project_id, round_num, client_req, dataset, homomorphic):
    cmd = f'python "{client_path}" {client_eth_key} {contract_address} {project_id} {round_num} {client_req} {dataset} '
    if homomorphic:
        cmd += f"{homomorphic}"
    else:
        cmd += "None"
    subprocess.call(cmd, shell=True)


def run_server(server_eth_key, contract_address, server_path, project_id, round_num, participants, dataset, homomorphic):
    cmd = f'python "{server_path}" {server_eth_key} {contract_address} {project_id} {round_num} {participants} {dataset} '

    if homomorphic:
        cmd += f"{homomorphic}"
    else:
        cmd += "None"
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Run participant or server mode with specific arguments")
    parser.add_argument("-m", "--mode", choices=["participant", "server"], required=True)
    parser.add_argument("-c", "--contract", required=True, help="Contract address in hex (0x...)")
    parser.add_argument("-ek", "--eth_key", required=True, help="ETH private key in hex (0x...)")
    parser.add_argument("-id", "--project_id", type=int, required=True, help="Project ID")
    parser.add_argument("-r", "--round", type=int, default=2, help="Round number")
    parser.add_argument("-p", "--participants", type=int, default=2, help="Required participants/clients")
    parser.add_argument("-d", "--dataset", choices=["MNIST", "UCI_HAR"], default="UCI_HAR", help="Dataset")
    parser.add_argument("-H", "--homomorphic", choices=["CKKS", "BFV"], help="Homomorphic encryption (CKKS or BFV)")
    args = parser.parse_args()

    main_dir = os.path.dirname(__file__)

    if args.mode == "participant":
        client_path = main_dir + "/participant/client.py"
        run_client(args.eth_key, args.contract, client_path, args.project_id, args.round, args.participants, args.dataset, args.homomorphic)

    elif args.mode == "server":
        server_path = main_dir + "/server/server.py"
        run_server(args.eth_key, args.contract, server_path, args.project_id, args.round, args.participants, args.dataset, args.homomorphic)

    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"\n--- Total Program Runtime: {total_runtime:.2f} seconds ---")
