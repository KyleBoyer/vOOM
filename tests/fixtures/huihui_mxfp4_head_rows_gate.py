#!/usr/bin/env python3
"""Actual Huihui head, synthetic activations: strict all-logit row-tiling gate.

No runtime wiring, weights changed, shortlist or tolerance. This is not a model
request, Plex score, hidden-state proof, cold-storage or serving speed result.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODEL = ROOT/'models/Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4'
METADATA = {
    'config.json': '080b04530c47fbb82f1379ff017de28dd5dc3537a10e3ea9742d8b5263394cf3',
    'model.safetensors.index.json': '8be4f8c5ba279604bf62918afa375435f22dfeadb3269e820b1913df04d9579c',
}


def native_reference_reservation_bytes(store, head_bytes, mlx_version):
    """Audited MLX 0.32 native load reads into the shared destination itself.

    No second full host buffer is created by this narrow WeightStore path.
    Small hash/input/logit buffers remain covered by the unchanged 400MB
    governor margin. Other loaders/representations require a new audit.
    """
    if type(head_bytes) is not int or head_bytes != 675430400 or mlx_version != '0.32.0':
        raise ValueError('reference workspace requires the audited version/geometry')
    for name in ('vpack2', 'packed', 'gguf', 'fast_dirs', '_raw_fast_tier_manifest',
                 'k3_scale_sidecar', 'bf16_nf12_sidecar', '_ct_int4_aux',
                 '_ct_mxfp4_aux', '_glm53_fp8_aux', '_dsv4_aux',
                 '_qwen4_fused_expert_slices'):
        if getattr(store, name, True):
            raise ValueError('reference workspace requires native raw MXFP4: '+name)
    aux = store._quant_aux.get('lm_head.weight')
    if (aux is None or (aux.bits, aux.group_size, aux.mode,
            aux.scales, aux.biases) != (4, 32, 'mxfp4', 'lm_head.scales', None)):
        raise ValueError('reference workspace requires the unchanged native head pair')
    return head_bytes


def run(preflight, result_path):
    pre = json.loads(preflight.read_text())
    assert pre['passed'] and pre['sample_seconds'] >= 30 and pre['known_transcoders']['passed']
    assert 0 <= time.monotonic() - pre['end']['monotonic_s'] < 120
    assert pre['end']['root_free_bytes'] >= 10_000_000_000 and not result_path.exists()
    assert not any(key.startswith('VMODEL_') for key in os.environ)
    for name, expected in METADATA.items():
        assert hashlib.sha256((MODEL/name).read_bytes()).hexdigest() == expected
    # This is the first possible MLX import/array or model payload read.
    import mlx.core as mx
    import numpy as np
    import psutil
    import resource
    from runtime.host_activity_witness import sample_known_transcoders, summarize_known_transcoders
    from runtime.process_memory_witness import sample_self_memory
    from runtime.model_loader import WeightStore
    from runtime.pressure import MemoryGovernor
    from runtime.quant import QTensor, matmul
    from runtime.weight_cache import WeightCache
    from tests.fixtures.mxfp4_head_rows import HeadRows
    from tests.fixtures.qwen4_hot_boundary_http_probe import _atomic_write_private

    document = dict(schema='voom.huihui-mxfp4-head-rows-gate.v1', passed=False,
        scope=__doc__, metadata_sha256=METADATA, cases=[], failures=[],
        preflight_sha256=hashlib.sha256(preflight.read_bytes()).hexdigest())
    started = time.perf_counter()
    store = WeightStore(MODEL)
    store._ensure_raw_fast_tier_loaded()
    assert store.dir.resolve() == MODEL.resolve()
    cache = WeightCache(store, 1024**2)
    governor = MemoryGovernor(cache, critical_available=5_600_000_000,
        floor_bytes=1024**2, metal_limit=8_500_000_000)
    reader = None
    observations = []

    def observe(phase):
        swap = psutil.swap_memory()
        value = dict(phase=phase, active_metal_bytes=int(mx.get_active_memory()),
            allocator_cache_bytes=int(mx.get_cache_memory()),
            system_available_bytes=psutil.virtual_memory().available,
            swap_used_bytes=swap.used, swap_out_bytes=swap.sout,
            process=sample_self_memory(),
            darwin_process_rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            known_transcoders=sample_known_transcoders())
        observations.append(value)
        return value

    def bits(value):
        mx.eval(value)
        assert value.dtype == mx.bfloat16
        return np.array(value.view(mx.uint16), copy=True)

    def project_serial(head, activations):
        # Preserve Qwen final_logits' (1,1,H) contraction shape at each position.
        rows = []
        for i in range(activations.shape[1]):
            value = matmul(activations[:, i:i+1, :], head)
            mx.eval(value)
            rows.append(value)
        value = mx.concatenate(rows, axis=1)
        mx.eval(value)
        return value

    try:
        observe('before')
        reader = HeadRows(MODEL)
        assert (reader.vocab, reader.hidden) == (248320, 5120)
        head_bytes = sum(extent.nbytes for extent in reader.extents)
        assert head_bytes == 675430400
        document.update(head_bytes=head_bytes, head_shape=[reader.vocab, reader.hidden],
            component_extents=[vars(e) for e in reader.extents], source_stat=list(reader.identity))
        version = importlib.metadata.version('mlx')
        incoming = native_reference_reservation_bytes(store, head_bytes, version)
        document['reference_admission'] = dict(incoming_bytes=incoming,
            margin_bytes=400000000, host_reserve_bytes=5600000000, mlx_version=version,
            loader='audited-native-mx-load-shared-destination',
            separate_whole_host_staging_bytes=0)
        governor.reserve(incoming, reason='mxfp4-head-row-oracle')
        rng = np.random.default_rng(64013)
        raw_inputs = rng.standard_normal((1, 6, reader.hidden)).astype(np.float32)
        raw_inputs /= np.sqrt(np.mean(raw_inputs**2, axis=-1, keepdims=True))
        activations = mx.array(raw_inputs).astype(mx.bfloat16)
        mx.eval(activations)
        document['input_bf16_sha256'] = hashlib.sha256(bits(activations).tobytes()).hexdigest()
        document['input_shape'] = list(activations.shape)
        t0 = time.perf_counter()
        fetched, fetch_seconds, fetched_bytes = store.fetch(['lm_head.weight'])
        resident = fetched.pop('lm_head.weight')
        assert isinstance(resident, QTensor) and resident.nbytes == head_bytes
        assert (resident.bits, resident.group_size, resident.mode) == (4, 32, 'mxfp4')
        mx.eval(resident.wq, resident.scales)
        loaded = observe('resident_loaded')
        logits = project_serial(resident, activations)
        reference = bits(logits)
        assert reference.shape == (1, 6, reader.vocab)
        assert np.isfinite(np.asarray(logits.astype(mx.float32))).all()
        document['reference'] = dict(wall_seconds=time.perf_counter()-t0,
            fetch_seconds=fetch_seconds, logical_fetch_bytes=fetched_bytes,
            logits_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
            loaded_observation=loaded)
        # Hash actual resident packed/scales bytes in small rows, avoiding a
        # second whole-head host copy. Every streamed scan must reproduce both.
        resident_hashes = []
        for array in (resident.wq, resident.scales):
            digest = hashlib.sha256()
            for start in range(0, reader.vocab, 8192):
                part = np.asarray(array[start:start+8192])
                digest.update(memoryview(part).cast('B'))
                del part
            resident_hashes.append(digest.hexdigest())
        document['component_sha256'] = resident_hashes
        del resident, fetched, logits, array
        mx.clear_cache()
        observe('resident_released')
        for block_rows in (8192, 32768, 65536):
            t0 = time.perf_counter()
            case_observations = len(observations)
            chunks = []
            hashes = [hashlib.sha256(), hashlib.sha256()]
            read_bytes = read_extents = 0
            read_seconds = project_seconds = 0.0
            for start in range(0, reader.vocab, block_rows):
                stop = min(reader.vocab, start+block_rows)
                block_bytes = (stop-start) * sum(e.row_bytes for e in reader.extents)
                governor.reserve(2 * block_bytes, reason='mxfp4-head-row-block')
                arrays = []
                r0 = time.perf_counter()
                for index, extent in enumerate(reader.extents):
                    raw = reader.read_component(index, start, stop)
                    hashes[index].update(raw)
                    read_bytes += len(raw)
                    read_extents += 1
                    host = np.frombuffer(raw, dtype='<u4' if index == 0 else 'u1')
                    array = mx.array(host.reshape(stop-start, extent.columns))
                    mx.eval(array)
                    arrays.append(array)
                    del raw, host, array
                read_seconds += time.perf_counter()-r0
                head = QTensor(arrays[0], arrays[1], None, 4, 32, 'mxfp4')
                observe(f'rows{block_rows}:{start}:loaded')
                p0 = time.perf_counter()
                block_logits = project_serial(head, activations)
                project_seconds += time.perf_counter()-p0
                chunks.append(block_logits)
                del head, arrays, block_logits
                observe(f'rows{block_rows}:{start}:released')
            tiled = mx.concatenate(chunks, axis=-1)
            mx.eval(tiled)
            actual = bits(tiled)
            assert read_bytes == head_bytes and [h.hexdigest() for h in hashes] == resident_hashes
            unequal = int(np.count_nonzero(actual != reference))
            left = actual.astype(np.uint32) << 16
            right = reference.astype(np.uint32) << 16
            left_float, right_float = left.view(np.float32), right.view(np.float32)
            finite = bool(np.isfinite(left_float).all())
            case = dict(block_rows=block_rows, tail_rows=reader.vocab % block_rows,
                wall_seconds=time.perf_counter()-t0, read_and_upload_seconds=read_seconds,
                projection_seconds=project_seconds, read_bytes=read_bytes, read_extents=read_extents,
                component_sha256=[h.hexdigest() for h in hashes], logits_compared=int(actual.size),
                logits_sha256=hashlib.sha256(actual.tobytes()).hexdigest(), unequal_logits=unequal,
                max_abs_difference=float(np.max(np.abs(left_float-right_float))), finite=finite,
                argmax_equal=bool(np.array_equal(np.argmax(left_float,axis=-1),
                    np.argmax(right_float,axis=-1))),
                max_observed_active_metal_bytes=max(o['active_metal_bytes']
                    for o in observations[case_observations:]))
            document['cases'].append(case)
            if unequal or not finite:
                document['failures'].append(f'rows{block_rows}: exact all-logit mismatch')
            del chunks, tiled, actual, left, right, left_float, right_float
            mx.clear_cache()
        reader.check_unchanged()
    except Exception as error:
        document['failures'].append(type(error).__name__+': '+str(error))
    finally:
        if reader is not None:
            reader.close()
        store.close()
        observe('after')
        document.update(wall_seconds=time.perf_counter()-started, observations=observations,
            whole_process_metal_peak=int(mx.get_peak_memory()),
            governor_reservation_calls=governor.reservation_calls,
            governor_reservation_failures=governor.reservation_failures)
        document['pressure'] = dict(minimum_observed_available_bytes=min(
            o['system_available_bytes'] for o in observations),
            swap_growth_bytes=observations[-1]['swap_used_bytes']-observations[0]['swap_used_bytes'],
            swap_out_growth_bytes=observations[-1]['swap_out_bytes']-observations[0]['swap_out_bytes'])
        p = document['pressure']
        document['known_transcoders'] = summarize_known_transcoders(
            o['known_transcoders'] for o in observations)
        if not document['known_transcoders']['passed']:
            document['failures'].append('known-transcoder observation gate')
        if not all(o['process']['available'] for o in observations):
            document['failures'].append('native process observation coverage')
        if (p['minimum_observed_available_bytes'] < 5_300_000_000
                or max(p['swap_growth_bytes'], p['swap_out_growth_bytes']) > 16_000_000
                or not 0 < document['whole_process_metal_peak'] <= 8_500_000_000):
            document['failures'].append('observed pressure gate')
        document['passed'] = len(document['cases']) == 3 and not document['failures']
        _atomic_write_private(result_path, document)
        print(json.dumps({k:v for k,v in document.items() if k!='observations'},sort_keys=True))
    return 0 if document['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.preflight, args.result))
