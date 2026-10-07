"""Start the explicit bounded-working-set Huihui service after fresh admission.

The lower startup buffer does not lower MemoryGovernor's live reserve. The
service loads its model lazily and individual allocations remain fail-closed.
No user process is stopped, no model is downloaded, and no retry task is created.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

from .profiles import apply_runtime_profiles

PROFILE = 'huihui-qwen38-27b-low-memory-online'
LAUNCH_AVAILABLE = 5_000_000_000
LIVE_AVAILABLE = 4_500_000_000
JOB_LABEL = 'com.voom.huihui.low-memory'
REQUIRED_SETTINGS = {
    'VMODEL_QWEN_MTP_SPECULATIVE': '0',
    'VMODEL_QWEN35_SERIAL_KV_RECLAIM': '0',
    'VMODEL_QWEN35_SERIAL_KV_RECLAIM_TOPUP': '0',
    'VMODEL_QWEN35_PREFILL_CHUNK_CEILING': '8',
    'VMODEL_QWEN35_WEIGHT_CACHE_MB': '256',
    'VMODEL_QWEN35_PREFETCH_DEPTH': '0',
    'VMODEL_QWEN35_KV_MAX_MB': '64',
    'VMODEL_QWEN35_MXFP4_HEAD_ROWS': '8192',
    'VMODEL_QWEN35_MIN_AVAILABLE_MB': '4500',
    'VMODEL_QWEN35_LOSSY_SUFFIX_PREFILL': 'off',
    'VMODEL_QWEN35_SPLIT_SINGLE_TOKEN_WEIGHTS': '1',
}


def validate_profile(env):
    for key, expected in REQUIRED_SETTINGS.items():
        if env.get(key) != expected:
            raise ValueError('Low-memory serving requires ' + key + '=' + expected)


def validate_preflight(pre, *, now):
    window = pre.get('pressure_window') or {}
    if not (pre.get('passed') is True and pre.get('sample_seconds', 0) >= 30
            and 0 <= now - pre['end']['monotonic_s'] < 120
            and window.get('complete') is True
            and window.get('minimum_available_bytes', 0) >= LAUNCH_AVAILABLE
            and window.get('minimum_root_free_bytes', 0) >= 10_000_000_000
            and pre.get('swap_growth_bytes', float('inf')) <= 16_000_000
            and pre.get('swap_out_growth_bytes', float('inf')) <= 16_000_000
            and pre.get('known_transcoders', {}).get('passed') is True):
        raise ValueError('Fresh low-memory serving admission did not pass')


def submit_background(port, result):
    """One login-session job, not a retry loop or a login/startup installation."""
    if platform.system() != 'Darwin':
        raise ValueError('background serving requires macOS launchctl')
    existing = subprocess.run(['/bin/launchctl', 'list', JOB_LABEL],
                              capture_output=True, check=False)
    if existing.returncode == 0:
        raise ValueError('Serving job already exists; inspect it before restarting')
    if existing.returncode != 113:
        raise RuntimeError('Cannot establish whether the serving job exists')
    result = result.resolve()
    result.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    child = (f'import os; os.chdir({str(root)!r}); '
             'from runtime.huihui_serve import main; raise SystemExit(main())')
    command = ['/bin/launchctl', 'submit', '-l', JOB_LABEL,
               '-o', str(result.with_suffix('.stdout.log')),
               '-e', str(result.with_suffix('.stderr.log')), '--',
               '/usr/bin/caffeinate', '-is', sys.executable, '-c', child,
               '--port', str(port), '--preflight-result', str(result)]
    subprocess.run(command, check=True)
    print(f'Submitted {JOB_LABEL}; admission/inference readiness not yet confirmed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8077)
    parser.add_argument('--background', action='store_true',
                        help='macOS one-shot managed job; no recurring retries or login installation')
    parser.add_argument('--preflight-result', type=Path,
                        default=Path('logs/huihui-low-memory-serve.preflight.json'))
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be in1..65535')
    if any(key.startswith('VMODEL_') for key in os.environ):
        parser.error('use a shell without VMODEL overrides')
    env = {}
    apply_runtime_profiles([PROFILE], environ=env)
    validate_profile(env)
    # Do not terminate an unknown service occupying the requested port.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', args.port))
    if args.background:
        submit_background(args.port, args.preflight_result)
        return 0
    command = [sys.executable, '-m', 'runtime.memory_preflight',
               '--result', str(args.preflight_result), '--sample-seconds', '30',
               '--sample-memory-window', '--min-root-free-gb', '10',
               '--min-stable-available-gb', '5.0', '--require-no-transcoders']
    if subprocess.run(command, check=False).returncode:
        return 1
    validate_preflight(json.loads(args.preflight_result.read_text()), now=time.monotonic())
    os.execv(sys.executable, [sys.executable, '-m', 'runtime.server',
                            '--profile', PROFILE, '--port', str(args.port)])
    return 1  # pragma: no cover - exec does not return


if __name__ == '__main__':
    raise SystemExit(main())
