"""Fixed-input diagnostic only: exact route grids are never used in production."""
from pathlib import Path
prefix = (Path(__file__).parent / 'validate_835.py').read_text().split('for shape in shapes:\n')[0]
exec(compile(prefix, 'validate_835.py', 'exec'))
import contextlib
bc.Config.mode = consts.BenchMode.CUDAGRAPH
report.update(scope='route-capacity-diagnostic',mode='CUDAGRAPH',rounds=5)
report['caveat'] = 'Exact grids use CPU-precomputed counts for frozen routes. This is an upper-bound diagnostic, not a deployable graph-safe dynamic dispatch.'
shapes = [(8,8,4096,14336,2), (64,256,2048,128,8)] + [(m,256,4096,256,6) for m in (32,40,80,2048,16384,65536)]
if a.indices:
    shapes = [shapes[int(i)] for i in a.indices.split(",")]
original_align, original_gemm = backend._align_routes, backend._gemm
state = {}
class ExactGrid:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            def exact(meta):
                dims = grid(meta)
                return (state['blocks'], *dims[1:])
            return original_gemm[exact](*args, **kwargs)
        return launch

def align(ids, bm, experts, group_small=False):
    state['blocks'] = block_counts[(ids.storage_offset(), ids.shape[0])]
    return original_align(ids,bm,experts,group_small=group_small)

@contextlib.contextmanager
def exact_grid():
    backend._align_routes,backend._gemm = align,ExactGrid()
    try: yield
    finally: backend._align_routes,backend._gemm = original_align,original_gemm

for shape in shapes:
    torch.manual_seed(0)
    inputs = next(bench._gen(shape,torch.bfloat16))
    if a.precision == 'fp8':
        x,nw1,nw2,_,_,w1,w2,s1,s2,tw,ti = inputs
    else: x,w1,w2,s1,s2,_,_,_,_,tw,ti = inputs
    m,e,k,n,topk = shape
    chunk = backend._workspace_tokens(m,k,n,e,topk,2)
    chunks=[]; block_counts={}
    for start in range(0,m,chunk):
        ids=ti[start:min(start+chunk,m)];r=ids.numel();bm=16 if r<e*16 else 64
        direct=ids.shape[0]<=4 or (a.precision=='fp8' and r<=e)
        rec=dict(start=start,tokens=ids.shape[0],routes=r,bm=bm,direct=direct)
        if not direct:
            route,expert,padded=original_align(ids,bm,e,group_small=k>=4096 and n>=1024)
            valid=int(padded.item());actual=(valid+bm-1)//bm;capacity=(route.numel()+bm-1)//bm
            cpu_counts=torch.bincount(ids.cpu().flatten(),minlength=e)
            expected=int(((cpu_counts+bm-1)//bm).sum())
            assert expected==actual
            block_counts[(ids.storage_offset(),ids.shape[0])]=actual
            rec.update(route_capacity=route.numel(),valid_padded_routes=valid,capacity_blocks=capacity,active_blocks=actual,empty_block_fraction=1-actual/capacity,gate_ctas=capacity*triton.cdiv(n,32 if bm==64 else 64),down_ctas=capacity*triton.cdiv(k,32 if bm==64 else 64),align_workspace_bytes=4*(e+1)**2 if r>16 else 0)
            del route,expert,padded
        chunks.append(rec)
    def candidate():
        return fg.fused_marlin_moe(x,w1,w2,None,None,s1,s2,tw,ti,0 if a.precision=='int4' else 2)
    current=candidate()
    with exact_grid(): tight=candidate()
    error=metrics(tight,current)
    del current,tight
    rounds=[]
    for iteration in range(5):
        timings={}
        for label in (['capacity','exact'] if iteration%2==0 else ['exact','capacity']):
            with exact_grid() if label=='exact' else contextlib.nullcontext():
                timings[label]=bench.get_latency(candidate)
        rounds.append(timings)
    med={label:statistics.median(t[label] for t in rounds) for label in ('capacity','exact')}
    save(dict(shape=shape,chunks=chunks,error_vs_capacity=error,rounds_ms=rounds,median_ms=med,gain=med['capacity']/med['exact'],pass_bounds=error['pass_bounds']))
    del inputs,x,w1,w2,s1,s2,tw,ti
    if a.precision=='fp8':del nw1,nw2
    gc.collect();dev.empty_cache()
assert all(c['pass_bounds'] for c in report['cases'])
print('COMPLETE',a.precision,len(report['cases']),flush=True)
