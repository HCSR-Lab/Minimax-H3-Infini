"""Check fixed input and device availability before loading model weights."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

mode = sys.argv[1]
root = Path(__file__).resolve().parents[1]
case = json.loads((root / 'manifests/case.json').read_text())
image = Path(os.environ['FIRST_FRAME'])
if hashlib.sha256(image.read_bytes()).hexdigest() != case['image_sha256']:
    raise SystemExit('First-frame SHA256 mismatch')
if not Path(os.environ['MODEL_PATH']).is_dir():
    raise SystemExit('MODEL_PATH is not a directory')
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit('Expected eight visible accelerator devices; no fallback is enabled')
for index in range(8):
    torch.empty(1, device=f'cuda:{index}')
    torch.cuda.synchronize(index)
modules = {}
for name in ['torch', 'diffusers', 'transformers'] + ([] if mode == 'pytorch' else ['infinicore', 'infinilm']):
    mod = importlib.import_module(name)
    modules[name] = {'version': getattr(mod, '__version__', None), 'path': mod.__file__}
if mode != 'pytorch':
    source = Path(os.environ['INFINILM_SOURCE'])
    if not (source / 'examples/minimax_h3.py').is_file():
        raise SystemExit('Official InfiniLM entry is missing')
    flash = Path(os.environ.get('FLASH_ATTN_2_CUDA_SO', ''))
    if not flash.is_file():
        raise SystemExit('Set FLASH_ATTN_2_CUDA_SO to the actual cp311 vendor extension')
print(json.dumps({'mode': mode, 'python': sys.version, 'executable': sys.executable,
                  'hip': torch.version.hip, 'modules': modules,
                  'devices': [torch.cuda.get_device_name(i) for i in range(8)]}, indent=2))
