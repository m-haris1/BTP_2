"""
Utility script to attempt HE decryption of a model file
using the server's public key context for testing purposes.
This script runs non-blocking via subprocess call from server.py.
"""
import os
import sys
import time
import json

import csv
import pickle
import traceback
import torch # Required for tensor manipulation (e.g., .clone(), .cpu(), .numpy())

# Determine repo root (parent of 'server' directory)
script_dir = os.path.dirname(os.path.abspath(__file__))        # .../server/tools
# project root is two levels up from tools
repo_root = os.path.dirname(os.path.dirname(script_dir))      # project root
# Put repo root at front so local imports resolve in subprocess
sys.path.insert(0, repo_root)

try:
    import tenseal as ts
except Exception:
    # tenseal import may fail when running tests without package; let later code handle it
    ts = None

# Placeholder imports assuming utils.py and simple_cnn_config.py are in the project structure
try:
    from utils import deserialize_data, HE_decrypt_model
    # NOTE: SimpleCNN is assumed to be defined in simple_cnn_config.py
    from simple_cnn_config import SimpleCNN 
except Exception as e:
    # Provide extra debug info to logs and exit with non-zero code
    print(f"Error: Required modules not found or import failed: {e}")
    print(f"sys.executable: {sys.executable}")
    print(f"sys.path (first entries): {sys.path[:8]}")
    traceback.print_exc()
    sys.exit(2)


def log_result(data):
    """
    Append a row to the metrics/he_decrypt_attempts.csv file.
    data: list of values: [timestamp, alg, model_file, ctx_file, success, error, params_changed, time_s]
    """
    metrics_file = os.path.join(repo_root, 'metrics', 'he_decrypt_attempts.csv')
    write_header = not os.path.exists(metrics_file)
    try:
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['timestamp', 'alg', 'model_file', 'ctx_file', 'success', 'error', 'params_changed', 'time_s'])
            writer.writerow(data)
    except Exception as e:
        print(f"ERROR: Failed to write metrics CSV: {e}")


def load_server_ctx(ctx_file):
    """Load the TenSEAL context from the given file path (expected to be an absolute path)."""
    if ts is None:
        raise ImportError("tenseal library is not available.")
    
    if not os.path.exists(ctx_file):
        raise FileNotFoundError(f"Context file not found at {ctx_file}")

    with open(ctx_file, 'rb') as f:
        serialized_context = pickle.load(f)
    
    # Use context_from to load the context
    ctx = ts.context_from(serialized_context)
    return ctx


def attempt_decrypt(alg, model_file, ctx_file, dataset_type='MNIST'):
    """Performs the decryption attempt and logs the result."""
    start = time.time()
    error = None
    success = False
    params_changed = 0
    
    # 1. Load Context
    try:
        ctx = load_server_ctx(ctx_file)
    except Exception as e:
        error = f"load_ctx_failed:{e}"
        # Log result immediately if context loading fails
        log_result([int(time.time()), alg, model_file, ctx_file, False, error, 0, round(time.time()-start,4)])
        return

    # 2. Deserialize model and attempt decryption
    try:
        with open(model_file, 'rb') as f:
            ser = f.read()
            
        enc_weights, metadata = deserialize_data(ser, ctx) 
        algorithm = metadata.get('algorithm', alg)
        
        model = SimpleCNN(dataset_type)
        before_state = {k: v.clone() for k,v in model.state_dict().items()}
        
        try:
            model_after = HE_decrypt_model(enc_weights, model, ctx, algorithm, metadata)
            after_state = model_after.state_dict()
            
            # compare states
            params_changed = 0
            for k in before_state.keys():
                b = before_state[k].cpu().numpy().ravel()
                a = after_state[k].cpu().numpy().ravel()
                if b.size != a.size:
                    params_changed += max(b.size, a.size)
                else:
                    # Count how many parameters have changed by more than a small epsilon
                    params_changed += int((abs(b - a) > 1e-6).sum())
            if params_changed > 0:
                success = True
        except Exception as e:
            error = f"decrypt_exception:{e}"
            # capture full traceback in error logs (this was missing/empty in original patch context)
            tb = traceback.format_exc()
            print(tb, file=sys.stderr)
            success = False

    except Exception as e:
        error = f"deserialize_or_io_error:{e}"
        success = False

    # 3. Final Logging (ensures a row is written even on failure)
    elapsed = round(time.time() - start, 4)
    # Log result, ensuring error is logged if present
    log_result([int(time.time()), alg, model_file, ctx_file, success, (error or ""), params_changed, elapsed])
    
    # Print result summary (used by server.py for console debug)
    print(json.dumps({
        "algorithm": alg, "model_file": os.path.basename(model_file), "ctx_file": os.path.basename(ctx_file),
        "success": success, "error": error, "params_changed": params_changed, "time_s": elapsed
    }, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python he_decrypt_attempt.py <algorithm> <model_file> <context_file> <dataset_type>")
        sys.exit(1)
    
    alg = sys.argv[1]
    model_file = sys.argv[2]
    ctx_file = sys.argv[3]
    dataset_type = sys.argv[4]

    attempt_decrypt(alg, model_file, ctx_file, dataset_type)