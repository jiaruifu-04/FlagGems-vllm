"""External, reproducible #835 / PPU validation; never imported by production."""
import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys

p = argparse.ArgumentParser()
p.add_argument('--precision', choices=['int4', 'fp8'], required=True)
p.add_argument('--scope', choices=['acceptance', 'trace', 'memory', 'smoke'], required=True)
p.add_argument('--output', required=True)
p.add_argument('--indices')
a = p.parse_args()
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'source'))
import torch
import triton
import vllm
import flaggems_vllm as fg
from flaggems_vllm.runtime import torch_device_fn as dev
from benchmark import conftest as bc, consts
bc.Config = bc.BenchConfig()
bc.Config.repetition = 100
bc.Config.warm_up = 1000
bc.Config.mode = consts.BenchMode.CUDAGRAPH if a.scope == 'trace' else consts.BenchMode.KERNEL
from benchmark import base
assert fg.vendor_name == 'mthreads'
suffix = {'int4':'w4a16_int4','fp8':'w8a16_fp8'}[a.precision]
bm = importlib.import_module('benchmark.test_fused_marlin_moe_' + suffix)
cls = getattr(bm, 'FusedMarlinMoE' + {'int4':'W4A16INT4','fp8':'W8A16FP8'}[a.precision] + 'Benchmark')
native = getattr(bm, '_vllm_baseline' if a.precision == 'int4' else '_vllm_baseline_fp8')
bench = cls('fused_marlin_moe', native, [torch.bfloat16])
if a.scope == 'trace':
    manifest = json.loads((root / 'evidence/ppu-trace.json').read_text())
    shapes = [tuple(row['shape']) for row in manifest['cases']]
    counts = {tuple(row['shape']):row['count'] for row in manifest['cases']}
    assert len(shapes) == 53 and sum(counts.values()) == 10105
elif a.scope == 'acceptance':
    bench.set_shapes()
    ordinary = ([(m,8,4096,14336,2) for m in (1,4,16,64,128,256,512)]
                + [(m,256,7168,2048,8) for m in (1,4,16,64,128,256)]
                + [(m,256,2048,128,8) for m in (1,16,64,512,1035)])
    supplemental = [(1024,*g) for g in ((8,4096,14336,2),(256,7168,2048,8),(512,4096,1024,10),(256,4096,2048,6))] if a.precision == 'int4' else []
    shapes = list(dict.fromkeys(bench.shapes + ordinary + supplemental))
    assert len(shapes) == (42 if a.precision == 'int4' else 38)
    counts = {}
elif a.scope == 'memory':
    shapes = [(m,256,4096,256,6) for m in (16384,32768,65536)]
    counts = {}
else:
    shapes = [(33,4,128,256,2)]
    counts = {}
if a.indices:
    shapes = [shapes[int(i)] for i in a.indices.split(',')]
backend = importlib.import_module(fg.fused_marlin_moe.__module__)
report = dict(precision=a.precision, scope=a.scope, dtype='bfloat16', torch=torch.__version__, triton=triton.__version__, vllm=vllm.__version__, device=dev.get_device_name(0), backend_sha256=hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest(), benchmark_sha256=hashlib.sha256(Path(bm.__file__).read_bytes()).hexdigest(), mode=bc.Config.mode.name, warmup_ms=1000, repeat_ms=100, rounds=3 if a.scope == 'trace' else 1, baseline='native quantized INT4 fused_experts' if a.precision == 'int4' else 'native BF16 fused_experts on identical decoded E4M3 weights', cases=[])
if a.scope == 'trace':
    report['trace_manifest'] = manifest

def metrics(actual, reference):
    y, ref = actual.float(), reference.float()
    assert bool(torch.isfinite(y).all()) and bool(torch.isfinite(ref).all())
    diff = y - ref
    rms = float(diff.square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-20))
    peak = float(diff.abs().max() / ref.abs().max().clamp_min(1e-20))
    mean = float(diff.abs().mean() / ref.abs().mean().clamp_min(1e-20))
    return dict(relative_rms=rms, relative_global_peak=peak, mean_relative=mean, pass_bounds=rms <= .01 and peak <= .02)

def decoded_expert(w, s, expert):
    if a.precision == 'int4':
        q = w[expert]
        values = torch.stack((q & 15, q >> 4), dim=-1).flatten(-2).float() - 8
    else:
        values = w[expert].float()
    return (values.reshape(values.shape[0], -1, 128) * s[expert].float().unsqueeze(-1)).reshape(values.shape).to(torch.bfloat16).float()

def fp32_reference(x, w1, w2, s1, s2, tw, ti):
    # Independent FP32 expert math using the same quantized effective weights.
    # No activation-stage A16 rounding; this is stricter than comparing two A16 implementations.
    out = torch.zeros_like(x, dtype=torch.float32)
    for e in range(w1.shape[0]):
        tokens, slots = torch.where(ti == e)
        tokens, slots = tokens.contiguous(), slots.contiguous()
        if tokens.numel() == 0:
            continue
        gu = x[tokens].float() @ decoded_expert(w1,s1,e).T
        gate, up = gu.chunk(2, -1)
        val = (torch.nn.functional.silu(gate) * up) @ decoded_expert(w2,s2,e).T
        out.index_add_(0, tokens, val * tw[tokens,slots,None].float())
    return out

def save(row):
    report['cases'].append(row)
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(row),flush=True)

for shape in shapes:
    torch.manual_seed(0)
    inputs = next(bench._gen(shape, torch.bfloat16))
    if a.precision == 'fp8':
        x, native_w1, native_w2, _, _, w1, w2, s1, s2, tw, ti = inputs
    else:
        x, w1, w2, s1, s2, _, _, _, _, tw, ti = inputs
    def candidate():
        return fg.fused_marlin_moe(x,w1,w2,None,None,s1,s2,tw,ti,0 if a.precision == 'int4' else 2)
    reference = fp32_reference(x,w1,w2,s1,s2,tw,ti)
    baseline = native(*inputs)
    actual = candidate()
    row = dict(shape=shape, count=counts.get(shape), error_vs_native=metrics(actual,baseline), error_vs_fp32=metrics(actual,reference), native_error_vs_fp32=metrics(baseline,reference))
    del actual, baseline, reference
    if a.scope == 'memory':
        candidate()
        dev.synchronize()
        gc.collect()
        dev.empty_cache()
        initial = dev.memory_allocated()
        dev.reset_peak_memory_stats()
        actual = candidate()
        dev.synchronize()
        eager_peak = dev.max_memory_allocated() - initial
        output_bytes = x.numel() * x.element_size()
        del actual
        gc.collect()
        dev.empty_cache()
        initial = dev.memory_allocated()
        dev.reset_peak_memory_stats()
        graph = dev.MUSAGraph()
        with dev.graph(graph):
            actual = candidate()
        graph.replay()
        dev.synchronize()
        graph_peak = dev.max_memory_allocated() - initial
        row.update(eager_extra_peak_bytes=eager_peak, graph_extra_peak_bytes=graph_peak, output_bytes=output_bytes, eager_scratch_bytes=eager_peak-output_bytes, graph_scratch_bytes=graph_peak-output_bytes, budget_bytes=2<<30)
        row['graph_before_update'] = metrics(actual,native(*inputs))
        x.mul_(0.5)
        graph.replay()
        dev.synchronize()
        row['graph_after_activation_update'] = metrics(actual,native(*inputs))
        row['graph_after_update_vs_fp32'] = metrics(actual,fp32_reference(x,w1,w2,s1,s2,tw,ti))
        row['workspace_pass'] = eager_peak-output_bytes <= 2<<30 and graph_peak-output_bytes <= 2<<30
        del actual, graph
    else:
        funcs = {'candidate':candidate,'native':lambda:native(*inputs)}
        rounds = []
        for r in range(report['rounds']):
            order = ['native','candidate'] if r % 2 == 0 else ['candidate','native']
            rounds.append({name:bench.get_latency(funcs[name]) for name in order})
        med = {name:statistics.median(r[name] for r in rounds) for name in funcs}
        row.update(rounds_ms=rounds,median_ms=med,speedup=med['native']/med['candidate'])
        del funcs
    row['pass'] = row['error_vs_native']['pass_bounds'] and row['error_vs_fp32']['pass_bounds']
    if a.scope == 'memory':
        row['pass'] = row['pass'] and row['workspace_pass'] and row['graph_before_update']['pass_bounds'] and row['graph_after_activation_update']['pass_bounds'] and row['graph_after_update_vs_fp32']['pass_bounds']
    save(row)
    del inputs,x,w1,w2,s1,s2,tw,ti,_
    if a.precision == 'fp8':
        del native_w1,native_w2
    gc.collect()
    dev.empty_cache()
assert all(c['pass'] for c in report['cases']), 'Validation failures retained in JSON; no cases discarded'
print('COMPLETE',a.precision,a.scope,len(report['cases']),flush=True)
