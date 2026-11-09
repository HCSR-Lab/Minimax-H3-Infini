"""Fetch fixed upstream revisions into a fresh source directory."""
import argparse
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
sources = json.loads((root / 'manifests/sources.json').read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--only', choices=list(sources), nargs='+')
args = parser.parse_args()
selected = args.only or list(sources)
dest = root / 'sources'
for name in selected:
    if (dest / name).exists():
        raise SystemExit(f'sources/{name} already exists; preserve it and use a fresh checkout.')
dest.mkdir(exist_ok=True)
for name in selected:
    spec = sources[name]
    path = dest / name
    subprocess.run(['git', 'init', str(path)], check=True)
    subprocess.run(['git', '-C', str(path), 'remote', 'add', 'origin', spec['url']], check=True)
    subprocess.run(['git', '-C', str(path), 'fetch', '--depth=1', 'origin', spec['commit']], check=True)
    subprocess.run(['git', '-C', str(path), 'checkout', '--detach', spec['commit']], check=True)
    subprocess.run(['git', '-C', str(path), 'submodule', 'update', '--init', '--recursive'], check=True)
patch = root / 'patches/infinicore-flash-layout.patch'
core = dest / 'InfiniCore'
if 'InfiniCore' in selected:
    subprocess.run(['git', '-C', str(core), 'apply', '--check', str(patch)], check=True)
    subprocess.run(['git', '-C', str(core), 'apply', str(patch)], check=True)
print('Fixed sources prepared; dependency installation and building are separate steps.')
