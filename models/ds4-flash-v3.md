# DeepSeek-V4-Flash v3 Standard Lucifer Image

Updated on 2026-06-09. This page supersedes the Lucifer/Cutlass parts of
`ds4-flash-v2.md`: the recommended Lucifer image is now built by our standard
`blackwell-llm-docker` recipe, not by the external/procr Dockerfile. The runtime
backend is still the Lucifer FlashInfer/CUTLASS path:

```text
--attention-backend SPARSE_MLA_SM120
--kernel-config.moe_backend flashinfer_cutlass
```

This is not the B12X MoE backend. B12X is still used only as the comparison
variant below.

## Recommended Image

```text
voipmonitor/vllm:lucifer
voipmonitor/vllm:lucifer-vllm7c6bbf4-fi3395b41aa8d-dg324aced12c-tk9801a7-cu132-20260609
```

Pinned digest:

```text
voipmonitor/vllm@sha256:76f5f2cb4942d5b175908192ac07be81df077fe28cd5d3f8c7c92611895e14d4
```

Source state:

| Component | Revision |
|---|---|
| Docker recipe repo | `local-inference-lab/blackwell-llm-docker` |
| Docker recipe commit | `7e54b18` |
| vLLM branch | `local-inference-lab/vllm:lucifer` |
| vLLM commit | `7c6bbf4c5a482e100af886c5b6eb4303746cc3ba` |
| CUDA | `13.2.1` |
| cuBLAS package | `13.4.1.2-1` |
| cuDNN package | `9.22.0.52-1` |
| NCCL runtime | `2.30.4`, `local-inference-lab/nccl-canonical` |
| PyTorch | `2.12.0+cu132` |
| FlashInfer branch | `refs/pull/3395/head` |
| FlashInfer commit | `b41aa8dd2fb93c49b1c6134bd1953040f8089d51` |
| DeepGEMM branch | `refs/pull/324/head` |
| DeepGEMM commit | `aced12c2c8882a945c568ace9d4a7e5778aae410` |
| B12X package | PR11 `d90d89c8353adabb56cc84bd3924ef811ef8d877`, installed but not used for Lucifer MoE |
| CUTLASS commit | `d80a4e53b52b42550659a8696dab32705265e324` |
| Triton kernels commit | `9801a7afbaea43a085db2016eadddd631555ae13` |

Sanity check from the built image:

```text
torch 2.12.0+cu132 13.2
flashinfer 0.6.12+cu132
deep_gemm 2.5.0
vllm 0.11.2.dev279+lucifer.cu132.20260609
import_ok flashinfer.sparse_mla_sm120
import_ok vllm.third_party.triton_kernels.matmul_ogs
import_ok triton_kernels.matmul_ogs
```

Important: do not set `NCCL_GRAPH_FILE=` to an empty value. The current build
recipe and Lucifer branch contain no `NCCL_GRAPH_FILE` default. Runtime launch
commands below still explicitly unset graph/all-reduce override variables before
`exec vllm serve` so inherited image or shell env cannot leak in.

## Build

Exact rebuild:

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 7e54b18

IMAGE=voipmonitor/vllm:lucifer-vllm7c6bbf4-fi3395b41aa8d-dg324aced12c-tk9801a7-cu132-20260609 \
ALIAS_IMAGE=voipmonitor/vllm:lucifer \
./build-lucifer-cu132.sh
```

Push:

```bash
docker push voipmonitor/vllm:lucifer-vllm7c6bbf4-fi3395b41aa8d-dg324aced12c-tk9801a7-cu132-20260609
docker push voipmonitor/vllm:lucifer
```

The build helper pins FlashInfer PR3395, DeepGEMM PR324, B12X PR11, the Lucifer
vLLM branch, CUTLASS, and the legacy Triton kernels source hook required by the
Lucifer code path. It uses the same CUDA 13.2 base and patched NCCL stack as the
B12X images.

## Model

Model ID:

```text
deepseek-ai/DeepSeek-V4-Flash
```

Served model name used by the new Lucifer compose:

```text
DeepSeek-V4-Flash
```

The current local TP2 validation used the Hugging Face snapshot resolved by the
server:

```text
/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/553034d7dd9e06c2eeaee68cf85a17d6d4754cf0
```

## Docker Compose

Default GPU scope follows the current host convention: use the last four GPUs.
TP4 defaults to `12,13,14,15`; for TP2 set
`CUDA_VISIBLE_DEVICES=14,15 TP_SIZE=2`.

```yaml
services:
  ds4-lucifer:
    image: ${IMAGE:-voipmonitor/vllm:lucifer}
    container_name: ${CONTAINER_NAME:-ds4-lucifer}
    network_mode: host
    gpus: all
    runtime: nvidia
    ipc: host
    shm_size: 32g
    ulimits:
      memlock: -1
      stack: 67108864
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
      - ${LUCIFER_CACHE_DIR:-/root/.cache/lucifer-vllm7c6bbf4}:/cache
    environment:
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-12,13,14,15}
      CUTE_DSL_ARCH: sm_120a
      HF_HUB_OFFLINE: ${HF_HUB_OFFLINE:-1}
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      NCCL_IB_DISABLE: "1"
      MODEL_ID: ${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash}
      SERVED_MODEL_NAME: ${SERVED_MODEL_NAME:-DeepSeek-V4-Flash}
      PORT: ${PORT:-8000}
      TP_SIZE: ${TP_SIZE:-4}
      MTP: ${MTP:-1}
      SPECULATIVE_TOKENS: ${SPECULATIVE_TOKENS:-2}
      DRAFT_SAMPLE_METHOD: ${DRAFT_SAMPLE_METHOD:-probabilistic}
      GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION:-0.90}
      MAX_MODEL_LEN: ${MAX_MODEL_LEN:-262144}
      MAX_NUM_SEQS: ${MAX_NUM_SEQS:-64}
      MAX_NUM_BATCHED_TOKENS: ${MAX_NUM_BATCHED_TOKENS:-8192}
      MAX_CUDAGRAPH_CAPTURE_SIZE: ${MAX_CUDAGRAPH_CAPTURE_SIZE:-}
    entrypoint: ["/bin/bash", "-lc"]
    command: |
      set -euo pipefail

      unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE
      unset VLLM_ENABLE_PCIE_ALLREDUCE VLLM_PCIE_ALLREDUCE_BACKEND
      unset VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS
      unset VLLM_RTX6K_FUSED_ALLREDUCE_ADD VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER
      unset VLLM_CACHE_DIR

      GRAPH_CAP="$${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
      if [ -z "$${GRAPH_CAP}" ]; then
        if [ "$${MTP:-1}" = "1" ]; then GRAPH_CAP=192; else GRAPH_CAP=64; fi
      fi

      SPEC_ARGS=()
      if [ "$${MTP:-1}" = "1" ]; then
        SPEC_ARGS=(
          --speculative-config.method mtp
          --speculative-config.num_speculative_tokens "$${SPECULATIVE_TOKENS}"
          --speculative-config.draft_sample_method "$${DRAFT_SAMPLE_METHOD}"
        )
      fi

      exec vllm serve "$${MODEL_ID}" \
        --served-model-name "$${SERVED_MODEL_NAME}" \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port "$${PORT}" \
        --load-format auto \
        --tensor-parallel-size "$${TP_SIZE}" \
        --kv-cache-dtype fp8 \
        --block-size 256 \
        --gpu-memory-utilization "$${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "$${MAX_MODEL_LEN}" \
        --max-num-seqs "$${MAX_NUM_SEQS}" \
        --max-num-batched-tokens "$${MAX_NUM_BATCHED_TOKENS}" \
        --max-cudagraph-capture-size "$${GRAPH_CAP}" \
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
        --async-scheduling \
        --no-scheduler-reserve-full-isl \
        --enable-chunked-prefill \
        --enable-flashinfer-autotune \
        --enable-prefix-caching \
        --attention-backend SPARSE_MLA_SM120 \
        --kernel-config.moe_backend flashinfer_cutlass \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --default-chat-template-kwargs.thinking=true \
        --default-chat-template-kwargs.reasoning_effort=high \
        "$${SPEC_ARGS[@]}"
```

Equivalent TP2 launch used for the local validation:

```bash
docker rm -f ds4-lucifer-tp2 >/dev/null 2>&1 || true
docker run -d --name ds4-lucifer-tp2 \
  --gpus all --runtime nvidia --ipc host --shm-size 32g --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/.cache/lucifer-vllm7c6bbf4:/cache \
  -e CUDA_VISIBLE_DEVICES=14,15 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e HF_HUB_OFFLINE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e NCCL_IB_DISABLE=1 \
  voipmonitor/vllm:lucifer \
  /bin/bash -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_ENABLE_PCIE_ALLREDUCE VLLM_PCIE_ALLREDUCE_BACKEND VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS VLLM_RTX6K_FUSED_ALLREDUCE_ADD VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER VLLM_CACHE_DIR; exec vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --served-model-name DeepSeek-V4-Flash \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --load-format auto \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 262144 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --max-cudagraph-capture-size 192 \
    --compilation-config="{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}" \
    --async-scheduling \
    --no-scheduler-reserve-full-isl \
    --enable-chunked-prefill \
    --enable-flashinfer-autotune \
    --enable-prefix-caching \
    --attention-backend SPARSE_MLA_SM120 \
    --kernel-config.moe_backend flashinfer_cutlass \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --default-chat-template-kwargs.thinking=true \
    --default-chat-template-kwargs.reasoning_effort=high \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 2 \
    --speculative-config.draft_sample_method probabilistic'
```

Startup checks from this run:

```text
attention_backend='SPARSE_MLA_SM120'
kernel_config.moe_backend='flashinfer_cutlass'
SpeculativeConfig(method='mtp', num_spec_tokens=2)
Using 'FLASHINFER_CUTLASS_MXFP4_MXFP8' Mxfp4 MoE backend.
Using ['CUSTOM', 'PYNCCL'] all-reduce backends for group 'tp:0'
GPU KV cache size: 404,165 tokens
Maximum concurrency for 262,144 tokens per request: 1.54x
```

The actual vLLM process environment had no `NCCL_GRAPH_FILE`, no cpp all-reduce
selector variables, and no `VLLM_CACHE_DIR`; only `CUDA_VISIBLE_DEVICES=14,15`
remained from the checked variables.

## Validation

Local smoke test:

```bash
python3 /mnt/test.py --port 8000 -L
```

Result: 9 coherent loop iterations, `CJK characters in output: 0`.
Generation-only throughput was about `212-229 tok/s`.

Warm TP2 MTP probabilistic decode validation:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 8000 \
  --concurrency 1,2,4,8,16,32,64 \
  --contexts 0k \
  --duration 30 \
  --max-tokens 2048 \
  --skip-prefill \
  --display-mode plain \
  --output /root/bench-results/ds4-lucifer-clean-20260609/tp2_mtp_prob_decode_ctx0_30s_warm.json
```

Comparison against the previous `hg436/vllm-public:lucifer-9d9a0a0` TP2 MTP
probabilistic run from v2:

| C | New standard image tok/s | Previous Lucifer tok/s | Delta |
|---:|---:|---:|---:|
| 1 | 199.5 | 207.1 | -3.7% |
| 2 | 333.5 | 346.4 | -3.7% |
| 4 | 397.2 | 400.4 | -0.8% |
| 8 | 787.1 | 787.1 | +0.0% |
| 16 | 1,185.5 | 1,153.1 | +2.8% |
| 32 | 1,870.6 | 1,796.3 | +4.1% |
| 64 | 2,815.4 | 2,752.5 | +2.3% |

Conclusion: the new standard image is speed-equivalent to the previously tested
Lucifer image for the TP2 MTP probabilistic decode path. The full TP2/TP4 speed
matrix below is therefore carried forward from v2 instead of being rerun.

## Profile Quality

Profile quality results are carried forward from `ds4-flash-v2.md`. Both farms
used 30 full invocations per profile with `thinking=true` and
`reasoning_effort=high`.

| Profile | Samples | B12X score | B12X success | Lucifer score | Lucifer success | Delta |
|---|---:|---|---:|---|---:|---:|
| estonia | 900 / 900 | PASS 879 / FAIL 21 | 97.7% | PASS 878 / FAIL 22 | 97.6% | -0.1 pp |
| lavd-test | 300 / 300 | EXACT 266 / NEAR 21 / FAIL 13 | 95.7% | EXACT 272 / NEAR 19 / FAIL 9 | 97.0% | +1.3 pp |
| hotel-lights | 900 / 900 | EXACT 816 / FAIL 84 | 90.7% | EXACT 814 / FAIL 86 | 90.4% | -0.2 pp |

Profile speed and latency:

| Profile | B12X gen tok/s avg | Lucifer gen tok/s avg | Lucifer/B12X | B12X elapsed avg | Lucifer elapsed avg | Elapsed ratio |
|---|---:|---:|---:|---:|---:|---:|
| estonia | 41.1 | 80.6 | 1.96x | 69.8s | 45.4s | 0.65x |
| lavd-test | 84.0 | 129.2 | 1.54x | 226.8s | 141.7s | 0.62x |
| hotel-lights | 46.2 | 93.1 | 2.02x | 448.4s | 221.6s | 0.49x |

Result roots:

```text
/root/bench-results/ds4-black-pr11-20260609/profile-30x-thinking-high/
/root/bench-results/ds4-lucifer-cutlass-20260609/profile-30x-thinking-high/
```

## Decode Speed

The B12X matrix is carried forward from v2. The Lucifer matrix is also carried
forward from v2 and treated as representative for the new standard image based
on the TP2 validation above.

B12X aggregate decode tok/s:

| TP | MTP | Draft sampling | C1 | C2 | C4 | C8 | C16 | C32 | C64 | Accept avg |
|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TP2 | off | none | 131.7 | 220.4 | 359.7 | 541.6 | 780.3 | 1,091.0 | 1,486.6 | 0.000 |
| TP2 | on | probabilistic | 222.5 | 355.4 | 521.7 | 738.8 | 1,006.6 | 1,369.5 | 1,786.6 | 0.687 |
| TP2 | on | greedy | 189.3 | 294.3 | 433.7 | 596.5 | 821.3 | 1,055.6 | 1,311.4 | 0.547 |
| TP4 | off | none | 159.2 | 279.7 | 472.1 | 759.5 | 1,135.4 | 1,656.6 | 2,299.8 | 0.000 |
| TP4 | on | probabilistic | 285.4 | 470.9 | 724.5 | 1,071.1 | 1,504.3 | 1,996.9 | 2,544.7 | 0.706 |
| TP4 | on | greedy | 247.3 | 404.8 | 607.7 | 915.4 | 1,203.5 | 1,689.8 | 2,032.2 | 0.538 |

Lucifer aggregate decode tok/s:

| TP | MTP | Draft sampling | C1 | C2 | C4 | C8 | C16 | C32 | C64 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| TP2 | off | none | 123.8 | 205.1 | 350.2 | 565.5 | 827.3 | 1,237.7 | 1,924.2 |
| TP2 | on | probabilistic | 207.1 | 346.4 | 400.4 | 787.1 | 1,153.1 | 1,796.3 | 2,752.5 |
| TP4 | off | none | 146.8 | 260.4 | 452.6 | 745.7 | 1,178.8 | 1,809.8 | 2,739.4 |
| TP4 | on | probabilistic | 257.2 | 439.8 | 583.4 | 1,129.2 | 1,707.6 | 2,686.7 | 3,932.4 |

Decode speedup is Lucifer/B12X. Values above `1.00x` mean Lucifer is faster.

| TP | MTP | Lucifer draft sampling | C1 | C2 | C4 | C8 | C16 | C32 | C64 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| TP2 | off | none | 0.94x | 0.93x | 0.97x | 1.04x | 1.06x | 1.13x | 1.29x |
| TP2 | on | probabilistic | 0.93x | 0.97x | 0.77x | 1.07x | 1.15x | 1.31x | 1.54x |
| TP4 | off | none | 0.92x | 0.93x | 0.96x | 0.98x | 1.04x | 1.09x | 1.19x |
| TP4 | on | probabilistic | 0.90x | 0.93x | 0.81x | 1.05x | 1.14x | 1.35x | 1.55x |

## Prefill Speed

B12X warm prefill rerun, MTP on:

| TP | MTP | 8k tok/s | 64k tok/s | 128k tok/s |
|:---:|:---:|---:|---:|---:|
| TP2 | on | 6,978 | 6,644 | 6,154 |
| TP4 | on | 8,236 | 7,790 | 7,177 |

Lucifer prefill:

| TP | MTP | 8k tok/s | 64k tok/s | 128k tok/s |
|---|---|---:|---:|---:|
| TP2 | off | 13,409 | 12,712 | 11,670 |
| TP2 | on | 12,956 | 12,348 | 11,318 |
| TP4 | off | 15,593 | 14,770 | 13,475 |
| TP4 | on | 15,054 | 14,329 | 13,142 |

Prefill speedup is available only for MTP-on, because B12X no-MTP prefill was
not remeasured in the comparable rerun.

| TP | MTP | Lucifer draft sampling | 8k | 64k | 128k |
|---|---|---|---:|---:|---:|
| TP2 | on | probabilistic | 1.86x | 1.86x | 1.84x |
| TP4 | on | probabilistic | 1.83x | 1.84x | 1.83x |

## Notes

- The new standard Lucifer image removes the dependency on external Dockerfiles
  while preserving the Lucifer FlashInfer/CUTLASS runtime path.
- Runtime JIT can occur on the first launch because the `/cache` volume may be
  empty. Reusing the same cache volume avoids repeating most of the warmup cost.
- The current recommended MTP draft sampling is `probabilistic`.
- No full sweep was rerun for v3. Only TP2 MTP probabilistic was revalidated on
  the new image, and it matched the previous Lucifer image within normal
  measurement variance.
