# MUSA Marlin MoE: #835 follow-up, PPU-aligned validation

Latest supplement: [review-driven aggregation, interface tests and CTA analysis](review-followup.md). The measurements below retain their original tested revision and statistical definitions.

Code: `c3c14b502b48e3c450b64a6c1f90719c49e10ad0`. Base: `29b82b3d8aa6a1f906c61f9caa20dfd59c9d8a36`. Hardware runs: 2026-11-27.

## Scope and outcome

Bound MUSA INT4/FP8 temporary storage to 2 GiB by token batching, preserve output/inplace/shifted-alias semantics, and reuse the existing shared small-route expert grouping for large repeated-expert projections. Native MUSA benchmark imports register the vendor platform and fused_experts implementation. No NVIDIA production changes. Existing public exports, quantization layouts, and tuning configurations remain applicable.

Validation is complete for the frozen sets below. Performance remains limited on the explicitly listed FP8 shapes relative to native BF16; this is not a claim that every shape exceeds 0.9x. Paired measurements distinguish those inherited gaps from this change.

## Environment and measurement contract

- MTT S5000; torch 2.11.0.post2+musa5.2.0; Triton 3.6.0; native vLLM 0.32.0. Vendor image: `registry.mthreads.com/mcconline/inference/vllm:v0.32.0-hybrid-cache-fc928696-pd-20261105`. The environment uses the existing MUSA compiler overlay. This is not vLLM 0.33.
- INT4 baseline: native quantized fused_experts. FP8 baseline: native BF16 fused_experts on identical offline-decoded E4M3 values, following the PPU decoded-weight comparison convention; this is not a native FP8-kernel comparison. Do not disable the MUSA platform plugin with `VLLM_PLUGINS=""`.
- BF16 activations, group size 128, seed 0; quantized tensors are shared by candidate and reference. Trace records supply geometry/count, not original tensor values. FP32 reference separately decodes the effective BF16 weights and computes expert matmuls, SiLU, routing weights, and reduction in FP32 without intermediate A16 rounding.
- Candidate output allocation is inside every timed call. SpeedUp = native latency / candidate latency. Acceptance uses repository KERNEL timing (1000 ms warmup, 100 ms repetition); trace uses repository graph timing, three alternating rounds, median latency. Raw rounds are retained.
- Matrices and memory measurements started at d6ec7a4. Upstream moved to the base above during the run. The MUSA implementation, benchmark modules, shared aligner, MUSA reduction, tuning, and timing dependencies were unchanged by that upstream update. Production/benchmark hashes in each JSON match the published code. The complete functional suite and five-round paired controls were repeated after rebasing; the upstream test change only adds an Enflame-specific MXFP4 quantizer branch.
- Devices were on a shared host; no unrelated workload was stopped. Small timing differences should not be overinterpreted.

## Coverage and aggregate results

| Matrix | INT4 | FP8 | Aggregation |
|---|---:|---:|---|
| acceptance | 42 shapes, 1.649755x | 38 shapes, 2.065282x | Arithmetic mean; original #835 target >= 1.2x |
| trace | 53 shapes, 1.426678x | 53 shapes, 1.701389x | Call-count weighted mean, 10,105 calls |

The full #835 set is 42 INT4 / 38 FP8, not just the checked-in 32/28 rows. The additional original acceptance geometries are specified in `validate_835.py`; all rows are below. The 53-shape trace and counts are from [FlagOSTune f24bc0f](https://github.com/Vincent-Xiao/FlagOSTune/blob/f24bc0f3cbd42700f6d36c66aa43280b6962baef/shape-config/DeepSeek-V4-Flash-p32768d1024.txt), git blob `c7087e6b025499f63a2af6a0a31836734c170e53`. All rows, including slow ones, contribute to the mean.

## Accuracy

Every one of the 42/38 acceptance rows, 53 trace rows per precision, and three memory rows per precision passes both 1% relative RMS and 2% relative global peak against native and the independent FP32 reference. Metrics are RMS(error)/RMS(reference) and max(abs(error))/max(abs(reference)); they are not per-element relative-error bounds. Memory cases also pass after mutating the captured input and replaying.

| Precision / set | Max RMS vs FP32 | Max global peak vs FP32 |
|---|---:|---:|
| int4 / acceptance | 0.3814% | 0.6561% |
| int4 / trace | 0.3858% | 0.7132% |
| int4 / memory | 0.3755% | 0.7638% |
| fp8 / acceptance | 0.3799% | 0.9145% |
| fp8 / trace | 0.4208% | 0.7619% |
| fp8 / memory | 0.3757% | 0.8503% |

## Eager and graph memory

Geometry `(M, E=256, H=4096, I=256, topk=6)`. Steady-state allocated increments above resident inputs/weights, after JIT warmup. Peak includes the newly allocated final output; scratch subtracts only that output. Graph capture/replay overhead is included. Budget is 2,147,483,648 bytes.

| Type | M | Eager extra peak B | Graph extra peak B | Output B | Eager scratch B | Graph scratch B |
|---|---:|---:|---:|---:|---:|---:|
| int4 | 16384 | 990321152 | 990322176 | 134217728 | 856103424 | 856104448 |
| int4 | 32768 | 1980576256 | 1980577280 | 268435456 | 1712140800 | 1712141824 |
| int4 | 65536 | 2683561984 | 2683679744 | 536870912 | 2146691072 | 2146808832 |
| fp8 | 16384 | 990321152 | 990322176 | 134217728 | 856103424 | 856104448 |
| fp8 | 32768 | 1980576256 | 1980577280 | 268435456 | 1712140800 | 1712141824 |
| fp8 | 65536 | 2683561984 | 2683679744 | 536870912 | 2146691072 | 2146808832 |

This aligns the budget definition and stress geometries with PPU; it does not claim equal byte usage or equal latency across devices. MUSA keeps its existing quantized GEMMs and projection buffers, so its M=65536 scratch is near the 2 GiB cap.

## Tests and implementation audit

- Latest-base complete operator suite: `python -m pytest -q tests/test_fused_marlin_moe.py --junitxml=../evidence/accuracy-latest.xml`: **189 passed, 456 platform/capability-gated skipped**, 96.14 s. Skips are not counted as passes.
- New coverage: FP16/BF16 small grouping and router-weight placement; forced 257-token chunks with a three-token tail; allocation/output/inplace/shifted aliases and measured workspace; inference-mode INT4/FP8 mutations of w1/w2/s1/s2 followed by graph replay, both single and forced multi-chunk cases.
- CI helper unit tests: 84 passed. Workflow-policy unit tests: 10 passed; workflow policy, 13 export modules and exported-op registry checks pass. Black, isort, flake8 and git diff whitespace checks pass for all four changed files. Production no-torch-compute-fallback audit: one changed implementation checked, zero violations. Its torch calls are tensor allocations, dtype constants and metadata/no-copy views.
- No new compute kernel or tuning candidate set is introduced. Existing MUSA libtuner configuration is reused. No NVIDIA autotune changes apply to this MUSA host-dispatch and workspace change.
- Forward inference only; existing unsupported types/layouts remain unsupported. Inputs and caller-visible output are excluded from the temporary budget. Weight/scale version caches are not introduced.

## Paired controls and remaining performance gaps

Five alternating graph rounds compare the exact upstream MUSA implementation to the candidate on the same tensors, including all sub-0.9x native FP8 geometries, the small-group route, and the largest memory case. All numeric comparisons pass. No control point beyond the grouped route changes by more than 0.7%. These controls do not replace or filter the complete native-baseline matrices.

| Shape (M,E,H,I,topk) | INT4 before/after | FP8 before/after |
|---|---:|---:|
| (8, 8, 4096, 14336, 2) | 1.824934x | 1.757164x |
| (64, 256, 2048, 128, 8) | 1.000732x | 0.999373x |
| (32, 256, 4096, 256, 6) | 0.999990x | 0.999833x |
| (40, 256, 4096, 256, 6) | 0.993571x | 1.004394x |
| (80, 256, 4096, 256, 6) | 1.000387x | 0.994596x |
| (2048, 256, 4096, 256, 6) | 1.005118x | 1.000224x |
| (16384, 256, 4096, 256, 6) | 0.998845x | 0.999541x |
| (65536, 256, 4096, 256, 6) | 0.994262x | 1.005351x |

Below 0.9x native BF16:

- acceptance: `(64, 256, 2048, 128, 8)` = 0.765922x.
- trace: `(32, 256, 4096, 256, 6)` = 0.822064x.
- trace: `(40, 256, 4096, 256, 6)` = 0.753712x.
- trace: `(80, 256, 4096, 256, 6)` = 0.896480x.
- trace: `(2048, 256, 4096, 256, 6)` = 0.842865x.
- trace: `(16384, 256, 4096, 256, 6)` = 0.504945x.

These are inherited native-baseline gaps, not resolved by the workspace/routing change. The dispatch/control evidence replaces an unavailable MUSA profiler for this scoped change; no claim of full compute-kernel optimization is made. `optimization.md` and `deep_opt.md` referenced by workflow.md are absent from this checkout. The remaining gaps require a separate compute-path optimization effort; keep the bounded workspace and measured small-route improvement.

## Reproduce

Place a checkout named `source` inside this evidence directory, at the code SHA above. Use the vendor environment and its required compiler overlay; add `source/src` to PYTHONPATH, keep native MUSA platform registration enabled, set `VLLM_MUSA_ENABLE_MOE_TRITON=1`, `OMP_NUM_THREADS=1`, and choose an idle device using MUSA_VISIBLE_DEVICES. From the evidence directory:

```sh
python validate_835.py --precision int4 --scope acceptance --output evidence/recheck-acceptance-int4.json
python validate_835.py --precision fp8 --scope acceptance --output evidence/recheck-acceptance-fp8.json
python validate_835.py --precision int4 --scope trace --output evidence/recheck-trace-int4.json
python validate_835.py --precision fp8 --scope trace --output evidence/recheck-trace-fp8.json
python validate_835.py --precision int4 --scope memory --output evidence/recheck-memory-int4.json
python validate_835.py --precision fp8 --scope memory --output evidence/recheck-memory-fp8.json
python paired_controls.py --precision int4 --scope acceptance --output evidence/recheck-controls-int4.json
python paired_controls.py --precision fp8 --scope acceptance --output evidence/recheck-controls-fp8.json
```

## Complete timing rows

Times are milliseconds. Every acceptance and trace row is included.

### acceptance: int4

| Shape (M,E,H,I,topk) | Count | Native ms | Candidate ms | SpeedUp |
|---|---:|---:|---:|---:|
| (1, 8, 4096, 14336, 2) | 1 | 2.747520 | 1.415720 | 1.940723x |
| (4, 8, 4096, 14336, 2) | 1 | 5.331360 | 4.949800 | 1.077086x |
| (8, 8, 4096, 14336, 2) | 1 | 7.268160 | 4.669740 | 1.556438x |
| (16, 8, 4096, 14336, 2) | 1 | 8.324400 | 5.706800 | 1.458681x |
| (32, 8, 4096, 14336, 2) | 1 | 9.507920 | 7.176160 | 1.324931x |
| (64, 8, 4096, 14336, 2) | 1 | 8.409281 | 6.428040 | 1.308218x |
| (128, 8, 4096, 14336, 2) | 1 | 8.626800 | 7.458700 | 1.156609x |
| (256, 8, 4096, 14336, 2) | 1 | 12.625360 | 7.626400 | 1.655481x |
| (1, 256, 7168, 2048, 8) | 1 | 2.297840 | 1.281560 | 1.793002x |
| (4, 256, 7168, 2048, 8) | 1 | 7.833640 | 5.045880 | 1.552482x |
| (8, 256, 7168, 2048, 8) | 1 | 14.311161 | 7.936640 | 1.803176x |
| (16, 256, 7168, 2048, 8) | 1 | 25.352200 | 13.951040 | 1.817226x |
| (32, 256, 7168, 2048, 8) | 1 | 46.578640 | 22.440719 | 2.075630x |
| (64, 256, 7168, 2048, 8) | 1 | 56.715321 | 30.463240 | 1.861763x |
| (128, 256, 7168, 2048, 8) | 1 | 65.307922 | 35.174959 | 1.856660x |
| (256, 256, 7168, 2048, 8) | 1 | 66.455521 | 36.213060 | 1.835126x |
| (1, 512, 4096, 1024, 10) | 1 | 0.968600 | 0.544800 | 1.777900x |
| (4, 512, 4096, 1024, 10) | 1 | 2.647440 | 1.970720 | 1.343387x |
| (8, 512, 4096, 1024, 10) | 1 | 4.995300 | 3.314800 | 1.506969x |
| (16, 512, 4096, 1024, 10) | 1 | 8.929739 | 5.592760 | 1.596661x |
| (32, 512, 4096, 1024, 10) | 1 | 17.998440 | 9.344180 | 1.926166x |
| (64, 512, 4096, 1024, 10) | 1 | 24.199941 | 14.660680 | 1.650670x |
| (128, 512, 4096, 1024, 10) | 1 | 30.877760 | 18.499401 | 1.669122x |
| (256, 512, 4096, 1024, 10) | 1 | 34.043161 | 20.482700 | 1.662045x |
| (1, 256, 4096, 2048, 6) | 1 | 1.130240 | 0.636400 | 1.775990x |
| (4, 256, 4096, 2048, 6) | 1 | 3.101000 | 2.247280 | 1.379890x |
| (8, 256, 4096, 2048, 6) | 1 | 5.848760 | 3.883220 | 1.506162x |
| (16, 256, 4096, 2048, 6) | 1 | 10.967160 | 6.650120 | 1.649167x |
| (32, 256, 4096, 2048, 6) | 1 | 20.826940 | 11.179740 | 1.862918x |
| (64, 256, 4096, 2048, 6) | 1 | 27.387199 | 15.971920 | 1.714709x |
| (128, 256, 4096, 2048, 6) | 1 | 32.733681 | 19.468800 | 1.681340x |
| (256, 256, 4096, 2048, 6) | 1 | 34.785799 | 20.583160 | 1.690013x |
| (512, 8, 4096, 14336, 2) | 1 | 20.204280 | 15.023460 | 1.344849x |
| (1, 256, 2048, 128, 8) | 1 | 0.354080 | 0.087040 | 4.068014x |
| (16, 256, 2048, 128, 8) | 1 | 0.448520 | 0.467760 | 0.958868x |
| (64, 256, 2048, 128, 8) | 1 | 0.955840 | 0.632580 | 1.511018x |
| (512, 256, 2048, 128, 8) | 1 | 1.099000 | 0.789960 | 1.391210x |
| (1035, 256, 2048, 128, 8) | 1 | 1.141380 | 0.850280 | 1.342358x |
| (1024, 8, 4096, 14336, 2) | 1 | 37.969219 | 25.372120 | 1.496494x |
| (1024, 256, 7168, 2048, 8) | 1 | 68.803078 | 41.731062 | 1.648726x |
| (1024, 512, 4096, 1024, 10) | 1 | 34.791380 | 22.755280 | 1.528937x |
| (1024, 256, 4096, 2048, 6) | 1 | 35.497480 | 23.157440 | 1.532876x |

### acceptance: fp8

| Shape (M,E,H,I,topk) | Count | Native ms | Candidate ms | SpeedUp |
|---|---:|---:|---:|---:|
| (1, 8, 4096, 14336, 2) | 1 | 1.804720 | 0.489560 | 3.686412x |
| (16, 8, 4096, 14336, 2) | 1 | 6.309900 | 3.844580 | 1.641246x |
| (64, 8, 4096, 14336, 2) | 1 | 7.436980 | 3.837280 | 1.938086x |
| (256, 8, 4096, 14336, 2) | 1 | 5.893000 | 5.014040 | 1.175300x |
| (1024, 8, 4096, 14336, 2) | 1 | 28.367800 | 18.168480 | 1.561374x |
| (4096, 8, 4096, 14336, 2) | 1 | 94.540199 | 67.018326 | 1.410662x |
| (16384, 8, 4096, 14336, 2) | 1 | 359.890076 | 262.565979 | 1.370665x |
| (1, 256, 7168, 2048, 8) | 1 | 1.638920 | 0.467640 | 3.504662x |
| (16, 256, 7168, 2048, 8) | 1 | 19.020479 | 6.379040 | 2.981715x |
| (64, 256, 7168, 2048, 8) | 1 | 49.482180 | 24.319639 | 2.034659x |
| (256, 256, 7168, 2048, 8) | 1 | 32.060600 | 31.075359 | 1.031705x |
| (1024, 256, 7168, 2048, 8) | 1 | 85.339005 | 28.897360 | 2.953177x |
| (4096, 256, 7168, 2048, 8) | 1 | 136.719711 | 73.862122 | 1.851013x |
| (16384, 256, 7168, 2048, 8) | 1 | 401.177277 | 268.594543 | 1.493617x |
| (1, 512, 4096, 1024, 10) | 1 | 0.399500 | 0.193040 | 2.069519x |
| (16, 512, 4096, 1024, 10) | 1 | 7.747660 | 2.400760 | 3.227169x |
| (64, 512, 4096, 1024, 10) | 1 | 20.218081 | 9.506700 | 2.126719x |
| (256, 512, 4096, 1024, 10) | 1 | 27.837440 | 13.367599 | 2.082456x |
| (1024, 512, 4096, 1024, 10) | 1 | 28.452999 | 13.440000 | 2.117039x |
| (4096, 512, 4096, 1024, 10) | 1 | 29.945320 | 29.184000 | 1.026087x |
| (16384, 512, 4096, 1024, 10) | 1 | 89.604439 | 93.291725 | 0.960476x |
| (1, 256, 4096, 2048, 6) | 1 | 0.759520 | 0.229440 | 3.310321x |
| (16, 256, 4096, 2048, 6) | 1 | 8.245440 | 2.777480 | 2.968677x |
| (64, 256, 4096, 2048, 6) | 1 | 26.522520 | 10.761680 | 2.464534x |
| (256, 256, 4096, 2048, 6) | 1 | 18.298279 | 13.821000 | 1.323948x |
| (1024, 256, 4096, 2048, 6) | 1 | 48.346460 | 14.134060 | 3.420564x |
| (4096, 256, 4096, 2048, 6) | 1 | 55.211639 | 31.091400 | 1.775785x |
| (16384, 256, 4096, 2048, 6) | 1 | 184.670517 | 110.210640 | 1.675614x |
| (4, 8, 4096, 14336, 2) | 1 | 4.092680 | 1.633720 | 2.505129x |
| (128, 8, 4096, 14336, 2) | 1 | 6.620960 | 4.713860 | 1.404573x |
| (512, 8, 4096, 14336, 2) | 1 | 9.806641 | 10.733320 | 0.913663x |
| (4, 256, 7168, 2048, 8) | 1 | 6.274680 | 1.652120 | 3.797957x |
| (128, 256, 7168, 2048, 8) | 1 | 48.684219 | 28.477160 | 1.709588x |
| (1, 256, 2048, 128, 8) | 1 | 0.373520 | 0.095160 | 3.925179x |
| (16, 256, 2048, 128, 8) | 1 | 0.423240 | 0.240080 | 1.762912x |
| (64, 256, 2048, 128, 8) | 1 | 0.423800 | 0.553320 | 0.765922x |
| (512, 256, 2048, 128, 8) | 1 | 0.654160 | 0.535760 | 1.220994x |
| (1035, 256, 2048, 128, 8) | 1 | 0.759820 | 0.588280 | 1.291596x |

### trace: int4

| Shape (M,E,H,I,topk) | Count | Native ms | Candidate ms | SpeedUp |
|---|---:|---:|---:|---:|
| (1, 256, 4096, 256, 6) | 172 | 0.369888 | 0.115731 | 3.196102x |
| (2, 256, 4096, 256, 6) | 172 | 0.359054 | 0.211254 | 1.699635x |
| (4, 256, 4096, 256, 6) | 172 | 0.525441 | 0.329390 | 1.595191x |
| (8, 256, 4096, 256, 6) | 172 | 0.795255 | 0.657218 | 1.210032x |
| (16, 256, 4096, 256, 6) | 172 | 1.405514 | 1.179534 | 1.191584x |
| (24, 256, 4096, 256, 6) | 172 | 2.329192 | 1.357901 | 1.715288x |
| (32, 256, 4096, 256, 6) | 172 | 2.943892 | 1.843557 | 1.596854x |
| (40, 256, 4096, 256, 6) | 172 | 3.311183 | 2.140293 | 1.547070x |
| (48, 256, 4096, 256, 6) | 172 | 3.042487 | 2.080965 | 1.462056x |
| (56, 256, 4096, 256, 6) | 172 | 3.274569 | 2.414968 | 1.355947x |
| (64, 256, 4096, 256, 6) | 172 | 3.397737 | 2.475305 | 1.372654x |
| (72, 256, 4096, 256, 6) | 172 | 3.542342 | 2.523557 | 1.403710x |
| (80, 256, 4096, 256, 6) | 172 | 3.730047 | 2.695239 | 1.383939x |
| (88, 256, 4096, 256, 6) | 172 | 3.889716 | 2.653076 | 1.466116x |
| (96, 256, 4096, 256, 6) | 172 | 4.003160 | 2.734107 | 1.464157x |
| (104, 256, 4096, 256, 6) | 172 | 4.035326 | 2.719540 | 1.483827x |
| (112, 256, 4096, 256, 6) | 172 | 4.123092 | 3.009033 | 1.370238x |
| (120, 256, 4096, 256, 6) | 172 | 4.168129 | 3.068410 | 1.358400x |
| (128, 256, 4096, 256, 6) | 172 | 4.134659 | 3.046026 | 1.357395x |
| (136, 256, 4096, 256, 6) | 172 | 4.163082 | 3.070088 | 1.356014x |
| (144, 256, 4096, 256, 6) | 172 | 4.260147 | 3.084158 | 1.381300x |
| (152, 256, 4096, 256, 6) | 172 | 4.267590 | 3.093901 | 1.379356x |
| (160, 256, 4096, 256, 6) | 172 | 4.272880 | 3.259346 | 1.310962x |
| (168, 256, 4096, 256, 6) | 172 | 4.276890 | 3.111559 | 1.374517x |
| (176, 256, 4096, 256, 6) | 172 | 4.284707 | 3.106992 | 1.379053x |
| (184, 256, 4096, 256, 6) | 172 | 4.284198 | 3.094404 | 1.384499x |
| (192, 256, 4096, 256, 6) | 172 | 4.309638 | 3.121416 | 1.380668x |
| (200, 256, 4096, 256, 6) | 172 | 4.294159 | 3.088703 | 1.390279x |
| (208, 256, 4096, 256, 6) | 172 | 4.307549 | 3.117094 | 1.381912x |
| (216, 256, 4096, 256, 6) | 172 | 4.310373 | 3.100630 | 1.390160x |
| (224, 256, 4096, 256, 6) | 172 | 4.316336 | 3.129410 | 1.379281x |
| (232, 256, 4096, 256, 6) | 172 | 4.400819 | 3.119462 | 1.410762x |
| (240, 256, 4096, 256, 6) | 172 | 4.400920 | 3.171593 | 1.387606x |
| (248, 256, 4096, 256, 6) | 172 | 4.403524 | 3.128054 | 1.407752x |
| (256, 256, 4096, 256, 6) | 172 | 4.405479 | 3.150694 | 1.398257x |
| (272, 256, 4096, 256, 6) | 172 | 4.407163 | 3.112606 | 1.415908x |
| (288, 256, 4096, 256, 6) | 172 | 4.408439 | 3.150849 | 1.399127x |
| (304, 256, 4096, 256, 6) | 172 | 4.399476 | 3.128300 | 1.406347x |
| (320, 256, 4096, 256, 6) | 172 | 4.410267 | 3.346795 | 1.317758x |
| (336, 256, 4096, 256, 6) | 172 | 4.407593 | 3.173223 | 1.388996x |
| (352, 256, 4096, 256, 6) | 172 | 4.412788 | 3.178859 | 1.388167x |
| (368, 256, 4096, 256, 6) | 172 | 4.408026 | 3.166149 | 1.392236x |
| (384, 256, 4096, 256, 6) | 172 | 4.419298 | 3.173710 | 1.392471x |
| (400, 256, 4096, 256, 6) | 172 | 4.414284 | 3.208106 | 1.375978x |
| (416, 256, 4096, 256, 6) | 172 | 4.423986 | 3.196074 | 1.384194x |
| (432, 256, 4096, 256, 6) | 172 | 4.426202 | 3.183215 | 1.390482x |
| (448, 256, 4096, 256, 6) | 172 | 4.430699 | 3.184375 | 1.391387x |
| (464, 256, 4096, 256, 6) | 172 | 4.428466 | 3.230732 | 1.370731x |
| (480, 256, 4096, 256, 6) | 172 | 4.426985 | 3.394710 | 1.304083x |
| (496, 256, 4096, 256, 6) | 344 | 4.428423 | 3.318417 | 1.334499x |
| (512, 256, 4096, 256, 6) | 344 | 4.425703 | 3.319376 | 1.333293x |
| (2048, 256, 4096, 256, 6) | 43 | 4.753738 | 3.632988 | 1.308493x |
| (16384, 256, 4096, 256, 6) | 946 | 30.224440 | 22.122655 | 1.366221x |

### trace: fp8

| Shape (M,E,H,I,topk) | Count | Native ms | Candidate ms | SpeedUp |
|---|---:|---:|---:|---:|
| (1, 256, 4096, 256, 6) | 172 | 0.178435 | 0.059842 | 2.981779x |
| (2, 256, 4096, 256, 6) | 172 | 0.180891 | 0.072290 | 2.502305x |
| (4, 256, 4096, 256, 6) | 172 | 0.265021 | 0.130024 | 2.038250x |
| (8, 256, 4096, 256, 6) | 172 | 0.889961 | 0.250848 | 3.547805x |
| (16, 256, 4096, 256, 6) | 172 | 0.704841 | 0.435626 | 1.617996x |
| (24, 256, 4096, 256, 6) | 172 | 1.805552 | 0.625683 | 2.885731x |
| (32, 256, 4096, 256, 6) | 172 | 0.673893 | 0.819757 | 0.822064x |
| (40, 256, 4096, 256, 6) | 172 | 0.783157 | 1.039067 | 0.753712x |
| (48, 256, 4096, 256, 6) | 172 | 2.741449 | 1.387058 | 1.976448x |
| (56, 256, 4096, 256, 6) | 172 | 2.862731 | 1.475909 | 1.939640x |
| (64, 256, 4096, 256, 6) | 172 | 1.596812 | 1.590039 | 1.004259x |
| (72, 256, 4096, 256, 6) | 172 | 1.661585 | 1.759927 | 0.944122x |
| (80, 256, 4096, 256, 6) | 172 | 1.710972 | 1.908544 | 0.896480x |
| (88, 256, 4096, 256, 6) | 172 | 3.428391 | 1.964655 | 1.745035x |
| (96, 256, 4096, 256, 6) | 172 | 3.439930 | 1.942743 | 1.770656x |
| (104, 256, 4096, 256, 6) | 172 | 3.540408 | 1.970705 | 1.796518x |
| (112, 256, 4096, 256, 6) | 172 | 3.554993 | 1.985578 | 1.790408x |
| (120, 256, 4096, 256, 6) | 172 | 3.616756 | 2.005064 | 1.803811x |
| (128, 256, 4096, 256, 6) | 172 | 3.621985 | 2.010280 | 1.801731x |
| (136, 256, 4096, 256, 6) | 172 | 3.629367 | 2.017918 | 1.798570x |
| (144, 256, 4096, 256, 6) | 172 | 3.643168 | 2.015172 | 1.807869x |
| (152, 256, 4096, 256, 6) | 172 | 3.706175 | 1.990586 | 1.861851x |
| (160, 256, 4096, 256, 6) | 172 | 3.720660 | 2.065723 | 1.801142x |
| (168, 256, 4096, 256, 6) | 172 | 3.726751 | 2.020080 | 1.844853x |
| (176, 256, 4096, 256, 6) | 172 | 3.707630 | 2.009605 | 1.844954x |
| (184, 256, 4096, 256, 6) | 172 | 3.743108 | 2.059804 | 1.817216x |
| (192, 256, 4096, 256, 6) | 172 | 3.761455 | 2.040415 | 1.843476x |
| (200, 256, 4096, 256, 6) | 172 | 3.778250 | 2.072997 | 1.822603x |
| (208, 256, 4096, 256, 6) | 172 | 3.866156 | 2.073772 | 1.864311x |
| (216, 256, 4096, 256, 6) | 172 | 3.872100 | 2.076316 | 1.864890x |
| (224, 256, 4096, 256, 6) | 172 | 3.871830 | 2.069482 | 1.870918x |
| (232, 256, 4096, 256, 6) | 172 | 3.871922 | 2.089443 | 1.853088x |
| (240, 256, 4096, 256, 6) | 172 | 3.862467 | 2.118534 | 1.823180x |
| (248, 256, 4096, 256, 6) | 172 | 3.868665 | 2.099335 | 1.842805x |
| (256, 256, 4096, 256, 6) | 172 | 3.868634 | 2.099856 | 1.842333x |
| (272, 256, 4096, 256, 6) | 172 | 3.877914 | 2.087729 | 1.857479x |
| (288, 256, 4096, 256, 6) | 172 | 3.882412 | 2.105080 | 1.844306x |
| (304, 256, 4096, 256, 6) | 172 | 3.874498 | 2.088169 | 1.855453x |
| (320, 256, 4096, 256, 6) | 172 | 3.871734 | 2.225011 | 1.740097x |
| (336, 256, 4096, 256, 6) | 172 | 3.885359 | 2.098721 | 1.851298x |
| (352, 256, 4096, 256, 6) | 172 | 3.881346 | 2.129827 | 1.822376x |
| (368, 256, 4096, 256, 6) | 172 | 3.878679 | 2.110667 | 1.837655x |
| (384, 256, 4096, 256, 6) | 172 | 3.881613 | 2.105995 | 1.843125x |
| (400, 256, 4096, 256, 6) | 172 | 3.880382 | 2.150704 | 1.804238x |
| (416, 256, 4096, 256, 6) | 172 | 3.882647 | 2.129105 | 1.823605x |
| (432, 256, 4096, 256, 6) | 172 | 3.885765 | 2.146125 | 1.810595x |
| (448, 256, 4096, 256, 6) | 172 | 3.891165 | 2.123696 | 1.832261x |
| (464, 256, 4096, 256, 6) | 172 | 3.892912 | 2.148731 | 1.811726x |
| (480, 256, 4096, 256, 6) | 172 | 3.889729 | 2.149297 | 1.809768x |
| (496, 256, 4096, 256, 6) | 344 | 3.893363 | 2.089378 | 1.863408x |
| (512, 256, 4096, 256, 6) | 344 | 3.898838 | 2.121778 | 1.837533x |
| (2048, 256, 4096, 256, 6) | 43 | 2.236719 | 2.653708 | 0.842865x |
| (16384, 256, 4096, 256, 6) | 946 | 7.891336 | 15.628100 | 0.504945x |
