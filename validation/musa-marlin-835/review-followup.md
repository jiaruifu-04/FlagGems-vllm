# Review follow-up for MUSA #835 / #1498

Code revision: `9e2c7a94325d17fb28b764dc8c73262e2e989a36`, on top of `c3c14b502b48e3c450b64a6c1f90719c49e10ad0`. This applies the portable review lessons from #744/#840. It does not mark another author's review discussions resolved.

## Performance aggregation

The primary shape-normalized aggregate is the unweighted geometric mean, `exp(sum(log(native_ms / candidate_ms)) / n)`. All shapes and regressions remain included. These are recomputations of the complete frozen raw measurements, not newly timed matrices. The original #835 arithmetic gate and #752 trace-count arithmetic statistic are retained and explicitly labeled.

| Set | Shapes | INT4 geometric mean | FP8 geometric mean |
|---|---:|---:|---:|
| Original acceptance | 42 / 38 | 1.607342x | 1.880006x |
| PPU trace | 53 / 53 | 1.418835x | 1.703614x |

For the trace only, call-count weighted geometric means are 1.413224x / 1.570517x. The PPU-compatible call-count weighted arithmetic means remain 1.426678x / 1.701389x. A weighted geometric mean is not interchangeable with either of the other statistics. All 10,105 calls contribute.

## Exact interface and dispatch contract

- Let input be `(M,K)` with intermediate width `N`. INT4 weights are `(E,2*N,K/2)` and `(E,K,N/2)`, low nibble first with zero code 8. FP8 weights are `(E,2*N,K)` and `(E,K,N)`, carrying E4M3FN bytes.
- Scales are `(E,2*N,K/group_size)` and `(E,K,N/group_size)`. Positive integer group sizes divide both K and N. FP8 alone also accepts `-1` with last scale dimension 1. K and N are positive multiples of 32. Group 128 remains the benchmark contract. Additional tested positive groups are 32, 64, 96 and 384. This is not a claim of exhaustive validation of all divisors.
- Activations/output: FP16 or BF16. INT4 scales match activation dtype; FP8 scales accept FP16, BF16 or FP32. Activations, routing and explicit output are contiguous; weight/scale strides are carried into the kernels.
- BF16 direct GEMV is selected when a chunk has `M <= 4`, or for FP8 when `M*topk <= E`. Otherwise grouped GEMM uses BM16 below `16*E` routes and BM64 at or above it. Shared small grouping requires at most 16 routes, more routes than experts, K >= 4096 and N >= 1024. MetaX's `routes_per_expert > 128` threshold is not used here.
- Output can alias activations, including positive and negative contiguous shifts. Chunk order protects unread activations. Sharing storage with weights/scales/routing is rejected conservatively even when byte ranges are disjoint. The new test explicitly records both actual overlap and disjoint shared storage behavior.
- `global_num_experts` is -1 or local E; activation is None/SiLU. Non-None bias, custom activation/reduction, expert maps, global scales, act-order, zero points, external workspace/caches, input_dtype overrides and clipping are rejected, as is `is_k_full=False`. They are compatibility signature parameters, not silently ignored supported features.
- No persistent decoded-weight/scale cache. Each invocation and graph replay reads current tensor storage. Existing inference-mode updates of both weights and both scales remain in the suite. There is no shared NaN-flag writer, TLE call path or 2V1C ring buffer in this implementation.

## Functional validation

Final complete MUSA suite at the revision above: **302 passed, 416 platform/capability-gated skipped**, 102.94 s. The 113 additional passing cases compared with the earlier 189-pass run are not added to the total again. All 36 group-layout cases pass the strict 1% RMS / 2% global peak bounds. See `evidence/review-final-tests.xml`.

New coverage includes 36 non-default group-layout cases across INT4/FP8, FP16/BF16 and M=1/33, checked against independent decoded-weight expert math with 1% RMS / 2% global peak bounds; nine invalid group cases; 24 weight/scale/routing shared-storage rejection cases; four additional backward-shifted chunk cases. The existing finite-E4M3-code and NaN/Inf/large-scale tests are reused on MUSA instead of copying their reference logic (8 finite-code and 32 special-value cases). Special-value tests retain their explicit NaN-aware comparison, rather than applying finite relative-error metrics to NaN/Inf.

Black/isort/flake8, whitespace and no-torch-compute-fallback audit pass. This follow-up changes only production docstrings plus tests. `review-source-equivalence.json` verifies that the production AST after removing docstrings is identical to c3c14b5. Therefore original full 42/38, 53+53 and memory evidence remains applicable to executable behavior; its raw byte hashes correctly refer to the earlier documented source, not this updated docstring. CTA measurements include their own observed source hashes; the minor docstring revision between the first diagnostic and recheck also leaves executable AST unchanged.

## Accuracy attribution

All original acceptance and trace rows compare candidate to native and independently compare both to FP32. This avoids assuming a native output is ground truth. The MetaX-specific two large-shape failures are not transferred as a MUSA diagnosis. MUSA has no corresponding failure in these frozen datasets.

| Set / precision | Candidate vs FP32 max RMS / peak | Native vs FP32 max RMS / peak |
|---|---:|---:|
| acceptance / int4 | 0.3814% / 0.6561% | 0.3811% / 0.6561% |
| acceptance / fp8 | 0.3799% / 0.9145% | 0.3787% / 0.9145% |
| trace / int4 | 0.3858% / 0.7132% | 0.3856% / 0.7132% |
| trace / fp8 | 0.4208% / 0.7619% | 0.4089% / 0.7619% |

## Routing capacity and redundant CTA measurement

The production grid uses the allocated sorted-route capacity, with CTA early return beyond the device-computed padded route count. The TLE-free aligner uses an upper bound `R + E*(BM-1)` (tightened for R<E) and `(E+1)^2` INT32 scratch: 264,196 bytes for E=256. Each workspace token chunk aligns only its own routes. There is no MetaX `fuse_sorted` path repeatedly launching the entire original-batch capacity.

An external fixed-input diagnostic precomputes each chunk's exact active block count on CPU, verifies it against the device-produced padded count outside timing, and replaces only the GEMM grid dimension while preserving routing/GEMM math/output allocation. Five alternating graph rounds compare capacity and exact grids. It is not a production fix: frozen host counts would be wrong after changing routing IDs during graph replay; transferring the device count to host inside production would introduce synchronization/capture problems. Raw rounds and code are included.

All 16 shape/precision comparisons, plus the one recheck, match exactly (RMS and peak error zero). Empty block percentages apply to both projection grids; the count of CTAs also depends on projection width.

| Precision | Shape (M,E,K,N,topk) | Empty blocks per chunk | Capacity ms | Frozen exact ms | Capacity/exact |
|---|---|---|---:|---:|---:|
| int4 | (8, 8, 4096, 14336, 2) | 56.25% | 4.881409 | 4.501556 | 1.084383x |
| int4 | (64, 256, 2048, 128, 8) | 18.75% | 0.656569 | 0.662281 | 0.991375x |
| int4 | (32, 256, 4096, 256, 6) | 29.17% | 1.813240 | 1.775546 | 1.021229x |
| int4 | (40, 256, 4096, 256, 6) | 33.33% | 2.114356 | 1.949831 | 1.084379x |
| int4 | (80, 256, 4096, 256, 6) | 19.63% | 2.638279 | 2.558482 | 1.031189x |
| int4 | (2048, 256, 4096, 256, 6) | 41.67% | 3.625411 | 3.626917 | 0.999585x |
| int4 | (16384, 256, 4096, 256, 6) | 6.94% | 22.124804 | 22.040161 | 1.003840x |
| int4 | (65536, 256, 4096, 256, 6) | 2.88%, 5.22% | 87.100098 | 87.390759 | 0.996674x |
| fp8 | (8, 8, 4096, 14336, 2) | 56.25% | 3.178357 | 3.089541 | 1.028747x |
| fp8 | (64, 256, 2048, 128, 8) | 18.75% | 0.371595 | 0.369308 | 1.006191x |
| fp8 | (32, 256, 4096, 256, 6) | direct GEMV | 0.818042 | 0.815933 | 1.002584x |
| fp8 | (40, 256, 4096, 256, 6) | direct GEMV | 1.023846 | 1.027472 | 0.996471x |
| fp8 | (80, 256, 4096, 256, 6) | 18.52% | 1.863730 | 1.668331 | 1.117122x |
| fp8 | (2048, 256, 4096, 256, 6) | 41.67% | 2.612038 | 2.545778 | 1.026028x |
| fp8 | (16384, 256, 4096, 256, 6) | 7.49% | 15.432734 | 15.528853 | 0.993810x |
| fp8 | (65536, 256, 4096, 256, 6) | 3.02%, 5.18% | 60.719721 | 60.966698 | 0.995949x |

The FP8 `(8,8,4096,14336,2)` first run had large timing variance (all raw rounds retained). A separate five-round recheck gives 1.048081x (3.323420 ms capacity, 3.170957 ms exact). It does not replace the first run or any full-matrix row.

Interpretation:

- FP8 M=32/40 uses direct GEMV, so grouped-grid redundancy cannot explain these native-baseline gaps.
- At FP8 M=80, eliminating excess capacity in this frozen-input experiment improves latency by 1.117x; at M=2048, 41.67% empty blocks correspond to only 1.026x. Empty CTA fraction is not elapsed-time fraction.
- At FP8 M=16384, the frozen-grid ratio is 0.994x; at M=65536, 0.996x. These measurements do not support redundant CTAs as the dominant explanation for the dense FP8 slowdown.
- A graph-safe device-side work queue, compact scheduling or another tighter proven static bound may be worth a separate measured kernel change. No CPU-precomputed route counts are installed in production. A static bound must preserve arbitrary updated routing, and timing gains from this diagnostic alone are insufficient to select a new implementation.

## Reproduce the added evidence

Use the same vendor environment and checkout layout as README.md. From this evidence directory:

```sh
python measure_route_capacity.py --precision int4 --scope acceptance --output evidence/recheck-routing-int4.json
python measure_route_capacity.py --precision fp8 --scope acceptance --output evidence/recheck-routing-fp8.json
python measure_route_capacity.py --precision fp8 --scope acceptance --indices 0 --output evidence/recheck-routing-fp8-small.json
cd source
python -m pytest -q tests/test_fused_marlin_moe.py --junitxml=../evidence/recheck-review-tests.xml
```
