from web3 import Web3
from solcx import install_solc, set_solc_version, compile_standard
import json
import os
import sys

# ------------------------------------------------------------------------------
# 1. Install and set Solidity compiler version
# ------------------------------------------------------------------------------
install_solc('0.8.0')
set_solc_version('0.8.0')

# ------------------------------------------------------------------------------
# 2. Connect to Ganache
# ------------------------------------------------------------------------------
ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

if not web3.is_connected():
    print("❌ Could not connect to Ganache. Is it running?")
    sys.exit(1)
else:
    print("✅ Connected to Ganache")

# ------------------------------------------------------------------------------
# 3. Get deployer address and private key
# ------------------------------------------------------------------------------
if len(sys.argv) == 3:
    deployer_account = sys.argv[1]
    private_key = sys.argv[2]
else:
    deployer_account = input("Enter deployer account address: ").strip()
    private_key = input("Enter deployer private key: ").strip()

web3.eth.default_account = deployer_account

# ------------------------------------------------------------------------------
# 4. Load and compile Solidity contract
# ------------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
contract_path = os.path.join(script_dir, "contract.sol")

with open(contract_path, "r") as file:
    contract_source = file.read()

compiled_sol = compile_standard(
    {
        "language": "Solidity",
        "sources": {"contract.sol": {"content": contract_source}},
        "settings": {
            "outputSelection": {
                "*": {
                    "*": ["abi", "metadata", "evm.bytecode", "evm.bytecode.sourceMap"]
                }
            }
        }
    }
)

# ------------------------------------------------------------------------------
# 5. Extract ABI and Bytecode
# ------------------------------------------------------------------------------
contract_name = list(compiled_sol["contracts"]["contract.sol"].keys())[0]
contract_interface = compiled_sol["contracts"]["contract.sol"][contract_name]

abi = contract_interface["abi"]
bytecode = contract_interface["evm"]["bytecode"]["object"]

# Save ABI and compiled code
with open(os.path.join(script_dir, "contract-abi.json"), "w") as f:
    json.dump(abi, f, indent=2)

with open(os.path.join(script_dir, "compiled-code.json"), "w") as f:
    json.dump(compiled_sol, f, indent=2)

# ------------------------------------------------------------------------------
# 6. Deploy Contract
# ------------------------------------------------------------------------------
print("🚀 Deploying contract...")

contract = web3.eth.contract(abi=abi, bytecode=bytecode)
nonce = web3.eth.get_transaction_count(deployer_account)

transaction = contract.constructor().build_transaction({
    "chainId": web3.eth.chain_id,
    "from": deployer_account,
    "nonce": nonce,
    "gas": 6721975,
    "gasPrice": web3.to_wei("20", "gwei")
})

signed_tx = web3.eth.account.sign_transaction(transaction, private_key)
tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)

receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

print(f"✅ Contract deployed at: {receipt.contractAddress}")
