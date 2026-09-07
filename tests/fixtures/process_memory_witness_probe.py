#!/usr/bin/env python3
"""CPU-only installed-SDK ABI/observer-cost gate. No model or MLX import.

Compiles a small read-only self-task oracle in a temporary external directory,
checks every ctypes prefix offset/count against the SDK, observes a bounded
16MiB touched host allocation, and times scalar reads plus flushed JSONL writes.
This is NOT a model latency, live-compression, or global-swap attribution proof.
"""

import argparse
from contextlib import redirect_stdout
import ctypes
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import process_memory_witness as witness
from runtime.memory_preflight import _atomic_json as atomic_json


def abi_source():
    fields = [name for name, _ in witness.TaskVMInfoRev1._fields_]
    entries = [f'"{name}":%zu' for name in fields]
    format_string = ('{"rev1_count":%u,"prefix_bytes":%zu,"offsets":{'
                     + ','.join(entries)
                     + '},"return_code":%d,"returned_count":%u}')
    arguments = ['(unsigned)TASK_VM_INFO_REV1_COUNT',
                 'offsetof(struct task_vm_info, min_address)']
    arguments.extend(f'offsetof(struct task_vm_info, {name})' for name in fields)
    arguments.extend(['(int)kr', '(unsigned)count'])
    return ('#include <mach/mach.h>\n#include <stddef.h>\n#include <stdio.h>\n'
            'int main(void) { task_vm_info_data_t info = {0};\n'
            'mach_msg_type_number_t count = TASK_VM_INFO_REV1_COUNT;\n'
            'kern_return_t kr = task_info(mach_task_self(), TASK_VM_INFO, '
            '(task_info_t)&info, &count);\n'
            'printf(' + json.dumps(format_string) + ', ' + ', '.join(arguments)
            + '); return kr == KERN_SUCCESS ? 0 : 1; }\n')


def timing_summary(values):
    ordered = sorted(values)
    return dict(samples=len(values), total_seconds=sum(values),
                median_seconds=statistics.median(values),
                p95_seconds=ordered[min(len(ordered)-1, int(len(ordered)*0.95))],
                maximum_seconds=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=256)
    args = parser.parse_args()
    if not 16 <= args.samples <= 1024:
        parser.error('samples must be 16..1024')
    if args.result.exists():
        parser.error('refusing existing result')
    result = {"schema": "voom.process-memory-probe.v1", "passed": False,
              "scope": "CPU-only native ABI, bounded allocation and observer cost",
              "model_executed": False, "physical_swap_attribution_proven": False,
              "failures": []}
    started = time.perf_counter()
    try:
        source = abi_source()
        sdk = subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()
        header = Path(sdk) / 'usr/include/mach/task_info.h'
        result['sdk_header_sha256'] = hashlib.sha256(header.read_bytes()).hexdigest()
        result['oracle_source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
        result['compiler'] = subprocess.check_output(['xcrun', 'clang', '--version'], text=True)
        with tempfile.TemporaryDirectory(prefix='process-memory-abi-', dir=ROOT/'logs') as temporary:
            binary = Path(temporary) / 'oracle'
            subprocess.run(['xcrun', 'clang', '-x', 'c', '-', '-o', str(binary)],
                           input=source, text=True, check=True, capture_output=True)
            abi = json.loads(subprocess.check_output([str(binary)], text=True))
            expected = {name: getattr(witness.TaskVMInfoRev1, name).offset
                        for name, _ in witness.TaskVMInfoRev1._fields_}
            result['sdk_oracle'] = abi
            result['ctypes_offsets'] = expected
            assert abi['offsets'] == expected, 'SDK offset mismatch'
            assert abi['prefix_bytes'] == ctypes.sizeof(witness.TaskVMInfoRev1), 'SDK size mismatch'
            assert abi['rev1_count'] == abi['returned_count'] == witness.REV1_COUNT, 'SDK count mismatch'
            assert abi['return_code'] == 0, 'native SDK call failed'
            rng = random.getstate()
            before = witness.sample_self_memory()
            assert before['available'], 'self reader unavailable'
            allocation_bytes = 16 * 1024 * 1024
            allocation = bytearray(allocation_bytes)
            for offset in range(0, allocation_bytes, 4096):
                allocation[offset] = 123
            during = witness.sample_self_memory()
            assert during['available'], 'allocated self reader unavailable'
            del allocation
            after = witness.sample_self_memory()
            assert after['available'], 'released self reader unavailable'
            delta = during['physical_footprint_bytes'] - before['physical_footprint_bytes']
            assert delta >= allocation_bytes // 2, 'bounded allocation not observed'
            result['bounded_allocation'] = dict(bytes=allocation_bytes,
                footprint_increase_bytes=delta, before=before, during=during, after=after,
                immediate_full_reclamation_required=False)
            durations = []
            for _ in range(args.samples):
                begin = time.perf_counter()
                sample = witness.sample_self_memory()
                durations.append(time.perf_counter() - begin)
                assert sample['available'], 'native reader failed during cost sample'
            result['self_read_cost'] = timing_summary(durations)
            durations = []
            observer = witness.GovernorProcessMemoryObserver()
            # Accelerated poll timestamps exercise the exact serializer/write
            # path; not a real duration/pressure trace. Existing server log sink.
            with (Path(temporary)/'trace.log').open('w+') as sink, redirect_stdout(sink):
                for index in range(args.samples):
                    begin = time.perf_counter()
                    observer.record(governor_monotonic_s=index * 2.0,
                        system_available_bytes=0, system_swap_used_bytes=0,
                        system_swap_out_bytes=0, metal_active_bytes=0,
                        cache_budget_bytes_after_response=0, swap_pressure_response=False)
                    durations.append(time.perf_counter() - begin)
                sink.seek(0)
                trace = sink.read()
            rows = [json.loads(line.removeprefix('[process-memory] ')) for line in trace.splitlines()]
            assert len(rows) == args.samples and all(row['process']['available'] for row in rows)
            assert random.getstate() == rng, 'Python RNG changed'
            result['flushed_jsonl_cost'] = timing_summary(durations)
            result['trace_cost_fixture'] = dict(accelerated_timestamps=True,
                synthetic_governor_inputs=True, serialized_bytes=len(trace.encode()),
                samples=len(rows), raw_trace_retained=False)
            result['python_rng_unchanged'] = True
            assert 'mlx.core' not in sys.modules, 'MLX imported by observer'
        result['passed'] = True
    except Exception as error:
        result['failures'].append(type(error).__name__ + ': ' + str(error))
    result['wall_seconds'] = time.perf_counter() - started
    atomic_json(args.result, result)
    print(json.dumps({key: result.get(key) for key in (
        'passed', 'wall_seconds', 'self_read_cost', 'flushed_jsonl_cost', 'failures')}, sort_keys=True))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
