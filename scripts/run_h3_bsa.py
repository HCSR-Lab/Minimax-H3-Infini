"""H3-BSA: establish producer completion at Torch -> InfiniCore bridges.

Runs the real official entry; never reuses reference conditions or latents.
The wrapper leaves the source and installed package unchanged.
"""
import json
import os
from pathlib import Path
import runpy
import sys

if '--legacy' in sys.argv:
    raise RuntimeError('The Infini candidate must not enable legacy mode')

import torch
import infinicore

original = infinicore.from_torch
counts = {}
pending = {}


def synchronized_from_torch(tensor):
    if tensor.is_cuda:
        key = str(tensor.device)
        counts[key] = counts.get(key, 0) + 1
        if not torch.cuda.current_stream(tensor.device).query():
            pending[key] = pending.get(key, 0) + 1
        torch.cuda.synchronize(tensor.device)
    return original(tensor)


infinicore.from_torch = synchronized_from_torch
# Some callers may import the defining module instead of the package alias.
import infinicore.tensor
infinicore.tensor.from_torch = synchronized_from_torch
entry = Path(os.environ['INFINILM_SOURCE']).resolve() / 'examples/minimax_h3.py'
sys.argv[0] = str(entry)
print('CANDIDATE_SYNC_BRIDGE_ENABLED', flush=True)
try:
    runpy.run_path(str(entry), run_name='__main__')
finally:
    print('SYNC_BRIDGE_COUNTS', json.dumps(counts), flush=True)
    print('TORCH_STREAM_PENDING_AT_BRIDGE', json.dumps(pending), flush=True)
