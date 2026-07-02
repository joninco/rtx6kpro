# W4A8-MX Tiny-Decode Kernel — rp-native Triton M=1 MoE path (work in progress)

Working log of the new tiny-decode kernel that closes the DS4 A8-vs-A16 decode gap by
replacing the b12x dynamic grouped kernel at M=1 with two Triton kernels that read the
**N256/K128 in-place-repacked weights directly** (zero extra weight memory). Written so
Luke can pick this up / work in parallel — every iteration below is measured, and the
open questions are flagged.

> **State (2026-07-02 late, updated):** kernel is correct (cos vs fp32 oracle
> **0.999999**, better than the production w4a8 dynamic kernel's 0.9990), fast in
> isolation (**28.8 µs/layer at M=1** vs dynamic 34.8 µs, a16 fused 22.5 µs), and
> **E2E-reproducible at 138.7–138.8 tok/s** DS4 TP2 A8 decode cc1 (baseline 133.7,
> A16 reference 140.5). The §6 "restart variance" was solved: `sem='relaxed'` on the
> atomic scatters — isolated-neutral but −7% E2E — was the regression; removed
> (vLLM commit `8e6e417c`).

## Code access (everything Luke needs)

- **vLLM branch (integrated)**: https://github.com/local-inference-lab/vllm/tree/fable/b12x-w4a8mx-tiny-decode-20260702 —
  head `8e6e417c` (`fix: drop relaxed atomics`), kernel commit `6896d418`. Files:
  [`vllm/model_executor/layers/fused_moe/b12x_tiny_decode.py`](https://github.com/local-inference-lab/vllm/blob/fable/b12x-w4a8mx-tiny-decode-20260702/vllm/model_executor/layers/fused_moe/b12x_tiny_decode.py)
  + a 10-line hook at the top of `_run_b12x_moe_fp4` in `b12x_moe.py`.
- **Standalone kit (this repo)**: [`optimization/code/tiny-decode/`](code/tiny-decode/) —
  final kernels, the fp32-oracle/timing driver, and the abandoned fused variant.
- **Docker image (code baked in, ready to serve)**:
  `voipmonitor/vllm:eldritch-enlightenment-v8e6e417c-b12x77bd50e-tinymoe-overlay-cu132-20260702`
  (pushed to Docker Hub; base v3f65c52-b12x77bd50e + the branch's vllm tree).
  Enable with `-e VLLM_B12X_W4A8_MX_TINY_DECODE=1` on the hybrid DS4 launch — no bind
  mounts needed.
- **Mappings prerequisite**: `lukealonso/b12x` PR #21 (`123a16f`,
  `tests/test_w4a8_rp_inverse_mapping.py`).

Measured with the baked image + env flag: decode cc1 **138.7–138.8 tok/s**, prefill
unchanged (12,202 tok/s @128k vs 12,182 hybrid reference), 30k coherence clean.

## Branches / commits

| What | Where |
|---|---|
| vLLM kernel + integration | `local-inference-lab/vllm` branch **`fable/b12x-w4a8mx-tiny-decode-20260702`**, commit `6896d418` — `vllm/model_executor/layers/fused_moe/b12x_tiny_decode.py` (kernels + `maybe_run_tiny_w4a8mx_moe`) and the hook at the top of `_run_b12x_moe_fp4` in `b12x_moe.py`. Branch is stacked on `9dbad237` (dual-path linear). |
| Verified inverse mappings (prerequisite) | `lukealonso/b12x` **PR #21**, commit `123a16f` — `tests/test_w4a8_rp_inverse_mapping.py` (weights + sfb grids, w13-rotated + w2 orientations, bit-exact vs the real prep kernels) |
| Related dense-linear work | PR #21 commits `a67e5bd` (tiled quant, 8–35×) + `49b453c` (SF stage-bulk/SFB k-reuse); vLLM `9dbad237` (dual-path linear) |
| Enable flag | `VLLM_B12X_W4A8_MX_TINY_DECODE=1` (default off) |

## 1. Design

Program = (routed row, w13 rp tile `nt`, k tile `kt`) for FC1; (routed row, out-row tile,
k tile) for FC2. Two kernels, FC1 stages SiLU inputs as fp32 in a small gmem scratch
([M·topk, 2n] = 48 KB at M=1), FC2 applies `silu(gate)·up` inline and scatters
router-weighted fp32 atomics into the output ([M, K] fp32 scratch → one bf16 copy).

Key decisions, all measured (§4):
- **BF16 activations, no input quantization** — at M=1 activations are 8 KB; skipping the
  MXFP8 quant both removes work and *improves numerics over the production path*
  (cos vs fp32 oracle 0.999999 vs 0.9990).
- **Whole-tile flat loads**: one rp (nt,kt) tile = 4096 contiguous int32 words whose flat
  index is `k32<<10 | n8c<<7 | r8<<4 | cgrp<<2 | n8i`. Load with a plain
  `tl.arange(0, 4096)` (provably contiguous ⇒ Triton vectorizes to 128-bit loads) and
  `tl.reshape` to `(k32, n8c, r8, cgrp, n8i)`; logical coords fall out of the axes:
  `row = n8c*32 + n8i*8 + r8`, `k = kt*128 + (k32*4+cgrp)*8 + j`, scale col `= kt*4+k32`.
- **SFU-free decodes**: e2m1 nibble → fp32 by direct bit assembly
  (`e>0: ((e+126)<<23)|(m<<22); e==0: m*(126<<23)`, plus sign bit);
  e8m0 scale byte → fp32 by `byte<<23` bitcast. No `exp2` anywhere.
- **Activation loads hoisted** out of the nibble loop: one contiguous 128-wide load per
  k-tile, per-j extraction via an 8-wide mask-sum (registers only).
- **Gate/up tile pairing**: after the prep rotation, rp tiles `[0, N/256)` hold the "up"
  rows and `[N/256, 2N/256)` the "gate" rows of the same channels **for both** the vLLM
  `w31` layout (rot=N) and the bench's up-first layout (rot=0) — the rotation exists
  exactly to normalize this. FC1 writes inter rows via `r_log = (p + N) % 2N`.
- `KT_PER_PROG=1` (one 16 KB tile per program; more programs = better latency hiding),
  `num_warps=8` (16 elems/thread; 4 warps ⇒ catastrophic spills, §4), relaxed-semantics
  atomics (kernel boundary is the only ordering needed; adds commute).

## 2. Correctness gates (all green)

- `cos(tiny, fp32 oracle) = 0.999999` at M=1/4/8 (dynamic kernel: 0.9990).
- `cos(tiny, dynamic kernel) = 0.999` (the residual is the dynamic path's MXFP8
  activation quantization).
- E2E serve: 30k-ctx coherence clean (CJK 0), sane completions.
- The gate/up pairing was additionally confirmed by breaking it (oracle cos drops to ~0.5).

Repro driver: scratchpad `tiny_moe_test.py` (synthetic experts through the *real* b12x
prep, fp32 oracle, dynamic-kernel cross-check, graph-replay timing). Ask fable/see the
journal for the file; it belongs in `benchmarks/` once this lands.

## 3. Isolated performance (graph replay, µs/layer, DS4 TP2 shapes, GPU 6)

| M | tiny (final) | b12x dynamic | a16 fused | notes |
|---:|---:|---:|---:|---|
| 1 | **28.7** | 34.8 (+3.0 wrapper) | 22.5 | tiny also has ~2 µs wrapper (2 zeros + 1 copy) |
| 4 | 195 | 88.0 | 75.6 | tiny re-reads weights per routed row — **do not use at M>1** (integration gates on M==1) |
| 8 | 390 | 210 | 190 | same |

All numbers share the same flattering condition (replayed identical routing ⇒ weights
partially L2-resident); comparisons are apples-to-apples.

## 4. Optimization journey (what moved the needle and what did not)

| Iteration | M=1 µs | Lesson |
|---|---:|---|
| v1: per-element computed-index loads | 108.1 | Triton emits scalar 4 B gathers for any index math it cannot prove contiguous |
| v2: SFU-free decodes + 2D accumulators | 92.2 | exp2-per-nibble was real but not dominant |
| v3: **flat `arange(4096)` loads + reshape** | **36.9** | the big one: provable contiguity ⇒ vectorized loads |
| fused single-kernel FC1+FC2 (3 variants incl. axis-separable affine + `<<`→`*` + num_stages) | 96–111 | per-program 512 B strips + register pressure; abandoned in favor of the 2-kernel shape |
| v7: hoisted activation loads + `KT_PER_PROG` 4→2/1 | 30.7 | fewer per-j gathers; more programs |
| `KT_PER_PROG=1` (FC1 grid = rt×8×32) | **28.8** | shorter streams win |
| relaxed atomics | 28.7 | isolated noise-level, **but −7% E2E in serving (131 vs 138.7) — do not use** |
| `num_stages=2/4` on the kt loop | 97–101 | staged 16 KB tile buffers ⇒ spills — **do not pipeline this loop** |
| `num_warps=4` | 10,966 (!) | 32 elems/thread ⇒ total spill catastrophe |
| `cache_modifier='.cs'` | n/a | not supported by this Triton build |
| **PDL overlap** (FC1 `gdc_launch_dependents` at entry; FC2 loads w2+scales, `gdc_wait`, then inter reads; `launch_pdl=True` on FC2; Triton 3.7 intrinsics) | 28.7 isolated, **131.3 E2E** | compiles + correct, but same −7% E2E as relaxed atomics — suspicious that both exotic variants land at exactly ~131.3 (a graph-replay fast path being disabled?). Reverted; E2E re-confirmed 138.69 after revert. Worth a Luke-level look at what PDL/relaxed do to the captured graph. |

NCU (isolated, M=1): FC1 27.5 µs @ ~57% DRAM, 80 regs/thread; FC2 18.6 µs @ ~42%,
77 regs/thread; no spills. Headroom to a16's 22.5 exists mainly in FC2 utilization and
FC1/FC2 overlap (they serialize at the kernel boundary; a16 is one fused kernel).

## 5. E2E serving (DS4 TP2 A8, hybrid DG-linear base = 133.7 tok/s)

Server: stock hybrid launch + `VLLM_B12X_W4A8_MX_TINY_DECODE=1` + bind-mounted
`b12x_moe.py`/`b12x_tiny_decode.py`. Warmup engages the path eagerly (log line
`b12x w4a8_mx tiny-decode path engaged`), buffers pre-allocate before graph capture,
decode graphs capture the Triton kernels fine.

| Run | decode cc1 ITL | tok/s |
|---|---:|---:|
| baseline (dynamic MoE) | 7.477 ms | 133.7 |
| tiny v1 module, first boot | 7.208 ms | 138.7 |
| tiny + fused-zero variant + relaxed atomics, after restart | 7.616 ms | 131.3 |
| tiny + relaxed atomics only, after restart | 7.635 ms | 131.0 |
| **tiny final (default-sem atomics), fresh restart, run 1/2** | **7.204 / 7.212 ms** | **138.81 / 138.66** |

## 6. SOLVED — the "restart variance" was relaxed atomics

Reverting `sem='relaxed'` → default semantics on both `tl.atomic_add` scatters restored
138.81/138.66 tok/s across two runs on a fresh restart. Lesson for kernel authors: the
isolated graph-replay microbench (identical routing every replay ⇒ L2-warm weights)
completely masked a 7% E2E effect — always confirm atomic/caching-semantics changes
end-to-end. Root cause hypothesis: relaxed semantics change the RED instruction/L2
policy Triton emits, which matters under real mixed traffic but not with a warm L2.

## 7. Next steps (parallel-friendly)

- ~~Resolve §6~~ done — 138.7 reproducible (4 measurements across 3 restarts:
  138.81/138.66/138.69 + first-boot 138.74).
- The remaining −1.8 t/s to the A16 reference (140.5) needs FC1/FC2 overlap that does
  NOT go through PDL or relaxed atomics (both measured −7% E2E, see §4) — i.e. a true
  single fused kernel; the fused CTA-shape that failed for us (§4, 512 B strips) might
  work with CuTe-DSL-style manual smem staging instead of Triton.
- FC2 utilization (42% DRAM): candidates — split FC2 differently (out-tile × expert),
  vectorize the inter reads across kt, or fold FC2 into FC1's tail for its own expert.
- M=2–4 support (per-expert row batching instead of per-routed-row weight re-reads) —
  only matters for MTP/batched decode; cc1 serving is M=1.
- Productionization: move kernels into b12x proper (or vllm officially), add the
  correctness driver to `benchmarks/`, wire a regression test vs the fp32 oracle.
- The same design trivially retargets **w4a8_nvfp4 / GLM shapes** if wanted later
  (mappings are format-generic; only scale decode differs).

## 8. Repro

```bash
# isolated (GPU 6): correctness + graph-replay timing
docker run --rm --gpus all --runtime nvidia --ipc host \
  -e CUDA_VISIBLE_DEVICES=6 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUTE_DSL_ARCH=sm_120a \
  -v <b12x>/benchmarks:/bench:ro -v <scratch>:/work:ro --entrypoint /bin/sh \
  voipmonitor/vllm:eldritch-enlightenment-v3f65c52-b12x77bd50e-overlay-cu132-20260702 \
  -lc 'python3 /work/tiny_moe_test.py'

# E2E: hybrid DS4 launch (drop --linear-backend b12x, VLLM_USE_B12X_FP8_GEMM=0)
#   + -e VLLM_B12X_W4A8_MX_TINY_DECODE=1
#   + bind-mount vllm/model_executor/layers/fused_moe/{b12x_moe.py,b12x_tiny_decode.py}
# then: llm_decode_bench.py --concurrency 1 --contexts 0 --max-tokens 512
```

Related pages: `b12x-dense-fp8-gemm-vs-deepgemm.md` (the full A8/A16 decode gap analysis
that motivated this kernel, incl. why the micro-port and all dynamic-kernel knobs were
dead ends).
