# DeepSeek-V4-Flash v2 Black Benediction PR11

Measured on 2026-06-08 and 2026-06-09 on the local 16x RTX PRO 6000 Blackwell
host. This page records the Black Benediction image with B12X PR11 for
DeepSeek-V4-Flash TP2/TP4, MTP on/off decode speed, and 30-run profile quality
checks.

Status: TP2 and TP4 start with B12X MLA sparse attention, B12X MoE/linear,
V2 model runner, CUDA graphs, `max_num_seqs=64`, and MTP2. The PR11 B12X
compressed MLA fix allows MTP full graph capture through 192 rows.

## Image

```text
voipmonitor/vllm:black-benediction-b12xpr11-vllmbb6c5b7-b12xd90d89c-fi3395b41aa8d-dg324aced12c-cu132-20260608
```

Pinned digest:

```text
voipmonitor/vllm@sha256:ce23a9b075bd7138ce3b12ee29609b98606e5050e2def4a29bbb917ad96e5997
```

Local image ID:

```text
sha256:da1c3e883628cf4f5fcd507e8e906851f744820259393d4d1b4e13919e37f326
```

Relevant source state:

| Component | Revision |
|---|---|
| CUDA | `13.2.1` |
| cuBLAS package | `13.4.1.2-1` |
| cuDNN package | `9.22.0.52-1` |
| NCCL runtime | `2.30.4`, `local-inference-lab/nccl-canonical` |
| PyTorch | `2.12.0+cu132` |
| vLLM branch | `dev/black-benediction` |
| vLLM commit | `bb6c5b7351fceb9d524e0d43b957415ffefcb981` |
| B12X branch | `refs/pull/11/head` |
| B12X commit | `d90d89c8353adabb56cc84bd3924ef811ef8d877` |
| FlashInfer branch | `refs/pull/3395/head` |
| FlashInfer commit | `b41aa8dd2fb93c49b1c6134bd1953040f8089d51` |
| DeepGEMM branch | `refs/pull/324/head` |
| DeepGEMM commit | `aced12c2c8882a945c568ace9d4a7e5778aae410` |

## Model

Local snapshot used for these measurements:

```text
/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136
```

Served model name:

```text
DeepSeek-V4-Flash
```

## Runtime Profile

Local helper used for the measurements:

```text
/root/run-ds4-flash-black-pr11
```

Important defaults:

| Setting | Value |
|---|---|
| GPUs | set by `CUDA_VISIBLE_DEVICES_VALUE` |
| Tensor parallel | `TP_SIZE=2` or `TP_SIZE=4` |
| MTP | set by `MTP=0|1` |
| MTP tokens | `2` |
| MTP draft sampling | `probabilistic` |
| MTP local argmax reduction | `true` |
| Max num seqs | `64` |
| Max batched tokens | `4096` |
| CUDA graph cap | `64` no-MTP, `192` MTP |
| Max model len | `130000` for speed matrix, `262144` for profile farm |
| KV cache dtype | `fp8` |
| Attention backend | `B12X_MLA_SPARSE` |
| MoE backend | `b12x` |
| Linear backend | `b12x` |
| GPU memory utilization | `0.875` speed matrix, `0.90` profile farm |

Important: unset empty NCCL graph variables before `vllm serve`:

```bash
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
```

Example launches:

```bash
# TP4 with MTP2.
CUDA_VISIBLE_DEVICES_VALUE=0,1,2,3 TP_SIZE=4 MTP=1 PORT=5329 /root/run-ds4-flash-black-pr11

# TP4 without MTP.
CUDA_VISIBLE_DEVICES_VALUE=0,1,2,3 TP_SIZE=4 MTP=0 PORT=5329 /root/run-ds4-flash-black-pr11

# TP2 with MTP2.
CUDA_VISIBLE_DEVICES_VALUE=0,1 TP_SIZE=2 MTP=1 PORT=5329 /root/run-ds4-flash-black-pr11
```

MTP config used by the helper:

```json
{
  "method": "mtp",
  "num_speculative_tokens": 2,
  "draft_sample_method": "probabilistic",
  "moe_backend": "b12x",
  "use_local_argmax_reduction": true
}
```

Readiness check:

```bash
curl -fsS http://127.0.0.1:5329/v1/models
docker logs ds4-flash-black-pr11 2>&1 | grep -E 'GPU KV cache size|Maximum concurrency|Graph capturing finished|Application startup complete' | tail -20
```

## Correctness Smoke

All four speed-matrix services passed the local smoke test with coherent output
and `chinese_count=0` before decode measurement:

```text
/root/bench-results/ds4-black-pr11-20260608/smoke/
```

## Decode Speed

Benchmark:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --host 127.0.0.1 \
  --port PORT \
  --model DeepSeek-V4-Flash \
  --contexts 0 \
  --concurrency 1,2,4,8,16,32,64 \
  --duration 30 \
  --max-tokens 2048 \
  --skip-prefill \
  --kv-budget KV_TOKENS \
  --display-mode plain \
  --output OUT.json
```

Aggregate decode tok/s:

| TP | MTP | C1 | C2 | C4 | C8 | C16 | C32 | C64 | Accept avg |
|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TP2 | off | 131.7 | 220.4 | 359.7 | 541.6 | 780.3 | 1,091.0 | 1,486.6 | 0.000 |
| TP2 | on | 222.5 | 355.4 | 521.7 | 738.8 | 1,006.6 | 1,369.5 | 1,786.6 | 0.687 |
| TP4 | off | 159.2 | 279.7 | 472.1 | 759.5 | 1,135.4 | 1,656.6 | 2,299.8 | 0.000 |
| TP4 | on | 285.4 | 470.9 | 724.5 | 1,071.1 | 1,504.3 | 1,996.9 | 2,544.7 | 0.706 |

Result JSONs:

```text
/root/bench-results/ds4-black-pr11-20260608/decode-sweep/
```

For TP4, use the `*-fullcc.json` files for the headline because the first run
auto-skipped some high-concurrency cells before the explicit KV budget rerun.

## Profile Farm

The 30-run quality farm used eight TP2+MTP2 replicas on ports `5500-5507`,
one GPU pair per replica, `MAX_MODEL_LEN=262144`, `GPU_MEMORY_UTILIZATION=0.90`,
and `max_num_seqs=64`.

Results:

```text
/root/bench-results/ds4-black-pr11-20260608/profiles-262k/
```

Profile summary:

| Profile | Samples | Score | Output tok avg | Output tok p50 | Elapsed avg | Gen tok/s avg |
|---|---:|---|---:|---:|---:|---:|
| estonia | 30 | PASS 11 / FAIL 19 | 161.2 | 148.0 | 1.4s | 216.1 |
| lavd-test | 30 | EXACT 2 / NEAR 2 / FAIL 26 | 2,290.7 | 275.5 | 15.2s | 156.2 |
| hotel | 30 | FAIL 30 | 3,407.5 | 2,953.5 | 24.5s | 137.5 |

## Notes

- The speed matrix is healthy and coherent for the short smoke tests.
- The profile quality results are weak; they are recorded as measured, not as a
  pass/fail endorsement of the model configuration.
- The key B12X PR11 fix is the compressed MLA decode split threshold for MTP
  full graph rows up to `64 * (1 + 2) = 192`.
