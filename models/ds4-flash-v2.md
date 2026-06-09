# DeepSeek-V4-Flash v2 Black Benediction PR11

Measured on 2026-06-08 and 2026-06-09 on the local 16x RTX PRO 6000 Blackwell
host. This page records the Black Benediction image with B12X PR11 for
DeepSeek-V4-Flash TP2/TP4, MTP on/off decode speed, prefill speed, H200 KLD
comparison, and profile quality checks.

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

Local helpers used for the measurements:

```text
/root/run-ds4-flash-black-pr11
/root/vllm/tmp/run_ds4_pr11_server.sh
```

Important defaults:

| Setting | Value |
|---|---|
| GPUs | set by `CUDA_VISIBLE_DEVICES_VALUE` |
| Tensor parallel | `TP_SIZE=2` or `TP_SIZE=4` |
| MTP | set by `MTP=0|1` |
| MTP tokens | `2` |
| MTP draft sampling | `probabilistic`; greedy measured separately |
| MTP local argmax reduction | `true` |
| Max num seqs | `64` |
| Max batched tokens | `4096` for the original speed matrix, `8192` for the 2026-06-09 reruns |
| CUDA graph cap | `64` no-MTP, `192` MTP |
| Max model len | `130000` for the original speed matrix, `262144` for profile/prefill/greedy reruns |
| KV cache dtype | `fp8` |
| Attention backend | `B12X_MLA_SPARSE` |
| MoE backend | `b12x` |
| Linear backend | `b12x` |
| GPU memory utilization | `0.875` speed matrix, `0.88` prefill/greedy reruns, `0.90` profile farm |
| DS4 chat kwargs for quality rerun | `{"thinking": true, "reasoning_effort": "high"}` |

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

For the greedy rerun only `draft_sample_method` changed to `greedy`; local
argmax reduction stayed enabled and the smoke test remained coherent.

Readiness check:

```bash
curl -fsS http://127.0.0.1:5329/v1/models
docker logs ds4-flash-black-pr11 2>&1 | grep -E 'GPU KV cache size|Maximum concurrency|Graph capturing finished|Application startup complete' | tail -20
```

## Correctness Smoke

All speed-matrix and greedy services passed the local smoke test with coherent
output and `chinese_count=0` before decode measurement:

```text
/root/bench-results/ds4-black-pr11-20260608/smoke/
/root/bench-results/ds4-black-pr11-20260609/greedy-tp2-smoke.txt
/root/bench-results/ds4-black-pr11-20260609/greedy-tp4-smoke.txt
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

| TP | MTP | Draft sampling | C1 | C2 | C4 | C8 | C16 | C32 | C64 | Accept avg |
|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TP2 | off | none | 131.7 | 220.4 | 359.7 | 541.6 | 780.3 | 1,091.0 | 1,486.6 | 0.000 |
| TP2 | on | probabilistic | 222.5 | 355.4 | 521.7 | 738.8 | 1,006.6 | 1,369.5 | 1,786.6 | 0.687 |
| TP2 | on | greedy | 189.3 | 294.3 | 433.7 | 596.5 | 821.3 | 1,055.6 | 1,311.4 | 0.547 |
| TP4 | off | none | 159.2 | 279.7 | 472.1 | 759.5 | 1,135.4 | 1,656.6 | 2,299.8 | 0.000 |
| TP4 | on | probabilistic | 285.4 | 470.9 | 724.5 | 1,071.1 | 1,504.3 | 1,996.9 | 2,544.7 | 0.706 |
| TP4 | on | greedy | 247.3 | 404.8 | 607.7 | 915.4 | 1,203.5 | 1,689.8 | 2,032.2 | 0.538 |

Result JSONs:

```text
/root/bench-results/ds4-black-pr11-20260608/decode-sweep/
/root/bench-results/ds4-black-pr11-20260609/decode-greedy/
```

For TP4, use the `*-fullcc.json` files for the headline because the first run
auto-skipped some high-concurrency cells before the explicit KV budget rerun.

## Prefill Speed

Benchmark:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --host 127.0.0.1 \
  --port PORT \
  --model DeepSeek-V4-Flash \
  --prefill-only \
  --prefill-contexts 8k,64k,128k \
  --prefill-duration 10 \
  --max-tokens 16 \
  --output OUT.json
```

Warm prefill rerun, client prompt tokens / TTFT:

| TP | MTP | ctx | tokens | TTFT s | tok/s | samples |
|:---:|:---:|---:|---:|---:|---:|---:|
| TP2 | on | 8k | 8,194 | 1.174 | 6,978 | 7 |
| TP2 | on | 64k | 64,561 | 9.718 | 6,644 | 1 |
| TP2 | on | 128k | 128,994 | 20.962 | 6,154 | 1 |
| TP4 | on | 8k | 8,195 | 0.995 | 8,236 | 10 |
| TP4 | on | 64k | 64,562 | 8.288 | 7,790 | 2 |
| TP4 | on | 128k | 128,995 | 17.973 | 7,177 | 1 |

The first TP2 8k run was a cold/warmup outlier: `3,987 tok/s` on rerun 1 and
`6,978 tok/s` on rerun 2. TP4 8k was stable across the two reruns
(`8,350` then `8,236 tok/s`).

Result JSONs:

```text
/root/bench-results/ds4-black-pr11-20260609/prefill-rerun/
```

## KLD vs H200

Reference:

```text
/root/vllm/artifacts/ds4_flash_2xh200_ref_logits_20260607/nomtp_tp2
```

Current B12X PR11 capture:

```text
/root/vllm/artifacts/ds4_flash_local_black_pr11_logits_20260609/nomtp_tp2_black_pr11_b12x_rowmeta_fullrows_fullcalls128
```

KLD summary, lower is better:

| Variant | Matched rows | Mean | Median | p90 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| B12X PR11 current TP2 no-MTP | 216 | 0.04048 | 0.00654 | 0.11784 | 0.40691 | 0.43863 |
| lucifer cutlass rowmeta fullrows | 350 | 0.02777 | 0.00699 | 0.07651 | 0.21563 | 0.43425 |
| b12x attention/linear + cutlass MoE default | n/a | 0.02719 | 0.00526 | 0.08437 | 0.24788 | 0.50644 |
| b12x attention/linear + cutlass MoE piecewise nobreak | n/a | 0.02884 | 0.00420 | 0.08437 | 0.30857 | 0.50644 |
| b12x offline spawn rowmeta fullrows | n/a | 0.02894 | 0.00692 | 0.08168 | 0.33458 | 0.34684 |
| cstechdev default rowmeta fullrows | n/a | 0.02920 | 0.00534 | 0.09020 | 0.30459 | 0.36811 |

The current B12X PR11 mean is worse than the best historical good runs, but it
is far from the previously broken variants with mean KLD around `1.15+`.
Matched rows are still limited by capture metadata/global-row alignment, not by
missing H200 reference logits.

Result JSON:

```text
/root/vllm/artifacts/ds4_flash_local_black_pr11_logits_20260609/nomtp_tp2_black_pr11_b12x_rowmeta_fullrows_fullcalls128/kld_vs_h200_ref_global_rows_allprompts.json
```

## Profile Farm

The 30-run quality farm used eight TP2+MTP2 replicas on ports `5500-5507`,
one GPU pair per replica, `MAX_MODEL_LEN=262144`, and `max_num_seqs=64`.
The estonia rerun used `thinking=true` and `reasoning_effort=high`. Because the
prompt is about `134k` tokens and each TP2 replica had only about `320k` KV
tokens, the rerun used one request per replica per wave.

Results:

```text
/root/bench-results/ds4-black-pr11-20260608/profiles-262k/
/root/bench-results/ds4-black-pr11-20260609/estonia-thinking-high/
```

Profile summary:

| Profile | Samples | Score | Output tok avg | Output tok p50 | Elapsed avg | Gen tok/s avg |
|---|---:|---|---:|---:|---:|---:|
| estonia, initial/default kwargs | 30 | PASS 11 / FAIL 19 | 161.2 | 148.0 | 1.4s | 216.1 |
| estonia, `thinking=true`, `reasoning_effort=high` | 30 | PASS 30 / FAIL 0 | 2,111.0 | 1,510.0 | 11.0s | 204.0 |
| lavd-test | 30 | EXACT 2 / NEAR 2 / FAIL 26 | 2,290.7 | 275.5 | 15.2s | 156.2 |
| hotel | 30 | FAIL 30 | 3,407.5 | 2,953.5 | 24.5s | 137.5 |

## Notes

- The speed matrix is healthy and coherent for the short smoke tests.
- The original profile quality results are weak without the DS4 thinking/high
  override; estonia becomes stable at `PASS 30/30` with the override.
- Greedy MTP works with local argmax enabled, but was slower than probabilistic
  MTP in this matrix.
- The key B12X PR11 fix is the compressed MLA decode split threshold for MTP
  full graph rows up to `64 * (1 + 2) = 192`.
