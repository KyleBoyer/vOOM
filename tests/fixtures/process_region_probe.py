#!/usr/bin/env python3
"""CPU-only SDK ABI, bounded tagged-allocation and map-reader cost gate."""

import argparse
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
from runtime import process_region_witness as witness
from qwen4_hot_boundary_http_probe import _atomic_write_private


def abi_source():
    names = [name for name, _ in witness.RegionInfo64._fields_]
    tags = list(witness.TAG_LABELS.items())
    fmt = ('{"count":%u,"bytes":%zu,"offsets":{'
           + ','.join('"' + name + '":%zu' for name in names)
           + '},"tags":{' + ','.join('"' + label + '":%u' for _, label in tags)
           + '},"return_code":%d,"returned_count":%u}')
    arguments = ['(unsigned)VM_REGION_SUBMAP_INFO_COUNT_64',
                 'sizeof(vm_region_submap_info_data_64_t)']
    arguments += [f'offsetof(struct vm_region_submap_info_64, {name})' for name in names]
    arguments += [f'(unsigned)VM_MEMORY_{label}' for _, label in tags]
    arguments += ['(int)kr', '(unsigned)count']
    return ('#include <mach/mach.h>\n#include <mach/mach_vm.h>\n'
            '#include <mach/vm_statistics.h>\n#include <stddef.h>\n#include <stdio.h>\n'
            'int main(void) { mach_vm_address_t address=0; mach_vm_size_t size=0;\n'
            'natural_t depth=0; vm_region_submap_info_data_64_t info={0};\n'
            'mach_msg_type_number_t count=VM_REGION_SUBMAP_INFO_COUNT_64;\n'
            'kern_return_t kr=mach_vm_region_recurse(mach_task_self(), &address, &size, '
            '&depth, (vm_region_recurse_info_t)&info, &count);\n'
            'printf(' + json.dumps(fmt) + ', ' + ', '.join(arguments)
            + '); return kr == KERN_SUCCESS ? 0 : 1; }\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    if args.result.exists():
        parser.error('refusing existing result')
    started = time.perf_counter()
    result = {'schema': 'voom.process-region-probe.v1', 'passed': False,
              'scope': 'CPU-only SDK ABI, 16MiB touched malloc, eight map walks',
              'model_executed': False, 'failures': []}
    try:
        sdk = subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()
        header = Path(sdk)/'usr/include/mach/vm_region.h'
        result['sdk_header_sha256'] = hashlib.sha256(header.read_bytes()).hexdigest()
        source = abi_source()
        result['oracle_source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
        with tempfile.TemporaryDirectory(prefix='region-abi-', dir=ROOT/'logs') as directory:
            binary = Path(directory)/'oracle'
            subprocess.run(['xcrun', 'clang', '-x', 'c', '-', '-o', str(binary)],
                           input=source, text=True, check=True, capture_output=True)
            abi = json.loads(subprocess.check_output([str(binary)], text=True))
        expected_offsets = {name: getattr(witness.RegionInfo64, name).offset
                            for name, _ in witness.RegionInfo64._fields_}
        result['sdk_oracle'] = abi
        assert abi['offsets'] == expected_offsets
        assert abi['bytes'] == ctypes.sizeof(witness.RegionInfo64)
        assert abi['count'] == abi['returned_count'] == witness.INFO_COUNT
        assert abi['return_code'] == 0
        assert abi['tags'] == {label: tag for tag, label in witness.TAG_LABELS.items()}
        rng_before = random.getstate()
        before = witness.sample_self_regions()
        assert before['coverage_complete'], before['reason']
        lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
        lib.malloc.argtypes = [ctypes.c_size_t]
        lib.malloc.restype = ctypes.c_void_p
        lib.free.argtypes = [ctypes.c_void_p]
        lib.free.restype = None
        size = 16 * 1024 * 1024
        allocation = lib.malloc(size)
        if not allocation:
            raise MemoryError('bounded malloc failed')
        try:
            ctypes.memset(allocation, 165, size)
            during = witness.sample_self_regions()
            assert during['coverage_complete'], during['reason']
        finally:
            lib.free(allocation)
        after = witness.sample_self_regions()
        assert after['coverage_complete'], after['reason']
        def large_bytes(sample):
            return sum(row['mapped_bytes'] for row in sample['groups']
                       if row['user_tag'] in (3, 4) and not row['external_pager'])
        delta = large_bytes(during) - large_bytes(before)
        assert delta >= size, 'touched malloc not observed in large-allocation tags'
        result['bounded_allocation'] = {'bytes': size, 'large_tag_mapped_delta': delta,
            'before': before, 'during': during, 'after': after,
            'immediate_reclamation_required': False}
        observations = [witness.sample_self_regions() for _ in range(8)]
        assert all(row['coverage_complete'] for row in observations)
        durations = [row['observation_seconds'] for row in observations]
        result['cost'] = {'samples': len(durations), 'median_seconds': statistics.median(durations),
                         'maximum_seconds': max(durations), 'seconds': durations,
                         'leaf_region_counts': [row['leaf_regions'] for row in observations]}
        result['python_rng_equal'] = random.getstate() == rng_before
        assert result['python_rng_equal'] and 'mlx.core' not in sys.modules
        result['passed'] = True
    except Exception as error:
        result['failures'].append(type(error).__name__ + ': ' + str(error))
    result['wall_seconds'] = time.perf_counter() - started
    _atomic_write_private(args.result, result)
    print(json.dumps({k: result.get(k) for k in ('passed', 'wall_seconds', 'cost', 'failures')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
