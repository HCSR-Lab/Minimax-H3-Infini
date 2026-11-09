"""Compare unshifted decoded PCM; no resampling or gain adjustment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def read_audio(path):
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'a:0',
        '-show_entries', 'stream=sample_rate,channels', '-of', 'json', str(path)]))
    stream = probe['streams'][0]
    channels = stream['channels']
    raw = subprocess.check_output([
        'ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:a:0', '-vn',
        '-acodec', 'pcm_f32le', '-f', 'f32le', '-'])
    pcm = np.frombuffer(raw, dtype='<f4').astype(np.float64).reshape(-1, channels)
    if not pcm.size or not np.isfinite(pcm).all():
        raise ValueError(f'Empty or nonfinite audio: {path}')
    return pcm, {'sample_rate': int(stream['sample_rate']), 'channels': channels,
                 'frames': len(pcm), 'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def compare(reference, candidate):
    x, xm = read_audio(reference)
    y, ym = read_audio(candidate)
    for key in ['sample_rate', 'channels', 'frames']:
        if xm[key] != ym[key]:
            raise ValueError(f'{key} mismatch: {xm[key]} vs {ym[key]}')
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    if norm == 0:
        raise ValueError('Silent audio cannot pass cosine acceptance')
    cosine = float(np.sum(x * y) / norm)
    return {'reference': str(reference), 'candidate': str(candidate),
            'reference_metadata': xm, 'candidate_metadata': ym,
            'method': 'ffmpeg pcm_f32le, all channels, zero shift, no gain or resampling',
            'pcm_cosine': cosine, 'threshold': 0.95, 'passed': cosine >= 0.95,
            'rmse': float(np.sqrt(np.mean((x-y)**2))),
            'mae': float(np.mean(np.abs(x-y))),
            'exact': bool(np.array_equal(x, y)), 'listening_performed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['pcm_cosine', 'rmse', 'passed']}))
