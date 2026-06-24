# DeepSeek-V4-Flash v5 Eldritch Backend Matrix

This page tracks DeepSeek-V4-Flash on the shared Eldritch vLLM image. It is a
backend matrix page, not yet a replacement for the known-good v3/v4 DS4F
production recipes.

## Current Image

```text
voipmonitor/vllm:eldritch-enlightenment-v9e8b0ab-b12x21e5cd4-cu132-20260624
voipmonitor/vllm@sha256:0158205fd17403dc755dbe34f76133edbbf2caa5c5bec4580aea99af2916bf9e
```

| Component | Revision |
|---|---|
| vLLM | `codex/eldritch-all-experiments-tp6fix-20260624 @ 9e8b0abcfd1986415812f30cbcffe37e346e7755` |
| B12X | `codex/eldritch-b12x-pr14-pr16-20260623 @ 21e5cd4d420b5ad5a68491416ae452599dbe0b5f` |
| FlashInfer | `b3baedbbef2686df91b6dc43818ee56fe26ceba2` |
| DeepGEMM | `14073b4e1e706506e193231209738c848d092a1f` |
| CUDA | `13.2.1` |

## Model

```text
deepseek-ai/DeepSeek-V4-Flash
/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136
```

## Backend Variants

| Variant | Attention | MoE / linear | Status on v5 image |
|---|---|---|---|
| B12X | `B12X_MLA_SPARSE` | `--moe-backend=b12x`, `--linear-backend=b12x` | Startup currently fails in B12X DS4 cache-page view. |
| FlashInfer/CUTLASS | `FLASHINFER_MLA_SPARSE_DSV4` | `--kernel-config.moe_backend flashinfer_cutlass` | Model load reaches `FLASHINFER_CUTLASS_MXFP4_MXFP8`, then worker died after graph capture in this v5 smoke. |
| Legacy Lucifer/CUTLASS | old v3 image path | `flashinfer_cutlass` | Still the known-good DS4F CUTLASS reference path from v3/v4. |

The old backend name `SPARSE_MLA_SM120` is rejected by this image; use
`FLASHINFER_MLA_SPARSE_DSV4`.

## Known v5 Failures

B12X TP2/MTP-off on the v5 image failed during startup:

```text
_b12x_cache_page_view -> torch.as_strided out of bounds
sizes [7166, 1039680], storage offset 181440, requires 7450528320, storage 7450346880
```

FlashInfer/CUTLASS TP2/MTP-off confirmed these log markers:

```text
Using FLASHINFER_MLA_SPARSE_DSV4
Using FLASHINFER_CUTLASS_MXFP4_MXFP8 MoE backend
Using DeepGemmFp8BlockScaledMMKernel
```

With AOT enabled it failed inside Dynamo/TileLang around
`mhc_pre_tilelang -> is_deep_gemm_supported`. With AOT disabled it loaded and
captured graphs, then the worker exited with `RuntimeError: cancelled`.

## B12X Compose Template

```yaml
services:
  ds4-b12x:
    image: ${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v9e8b0ab-b12x21e5cd4-cu132-20260624}
    container_name: ${NAME:-ds4-v5-b12x}
    network_mode: host
    ipc: host
    shm_size: 32g
    gpus: all
    ulimits:
      memlock: -1
      nofile: 1048576
      stack: 67108864
    volumes:
      - /root/.cache/huggingface:/root/.cache/huggingface
      - ${CACHE_ROOT:-/root/.cache/vllm-ds4-v5-b12x}:/cache
    environment:
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1}
      CUTE_DSL_ARCH: sm_120a
      NCCL_IB_DISABLE: "1"
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      VLLM_PREFIX_CACHE_RETENTION_INTERVAL: "4096"
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_USE_FLASHINFER_SAMPLER: "1"
      VLLM_USE_B12X_MOE: "1"
      VLLM_USE_B12X_FP8_GEMM: "1"
      VLLM_USE_B12X_MHC: "1"
      VLLM_USE_B12X_SPARSE_INDEXER: "1"
      VLLM_ENABLE_PCIE_ALLREDUCE: "1"
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
      B12X_W4A16_TC_DECODE: "1"
      MODEL_PATH: ${MODEL_PATH:-/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136}
      PORT: ${PORT:-8000}
    command:
      - /bin/bash
      - -lc
      - |
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
        exec vllm serve "$${MODEL_PATH}" \
          --served-model-name DeepSeek-V4-Flash \
          --host 0.0.0.0 \
          --port "$${PORT}" \
          --trust-remote-code \
          --kv-cache-dtype fp8 \
          --block-size 256 \
          --tensor-parallel-size 2 \
          --gpu-memory-utilization 0.86 \
          --max-model-len 128000 \
          --max-num-seqs 1 \
          --max-num-batched-tokens 8192 \
          --max-cudagraph-capture-size 4 \
          --attention-backend B12X_MLA_SPARSE \
          --moe-backend b12x \
          --linear-backend b12x \
          --async-scheduling \
          --no-scheduler-reserve-full-isl \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --enable-flashinfer-autotune \
          --tokenizer-mode deepseek_v4 \
          --tool-call-parser deepseek_v4 \
          --reasoning-parser deepseek_v4 \
          --enable-auto-tool-choice \
          --default-chat-template-kwargs.thinking=true \
          --default-chat-template-kwargs.reasoning_effort=high
```

## FlashInfer/CUTLASS Compose Template

```yaml
services:
  ds4-cutlass:
    image: ${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v9e8b0ab-b12x21e5cd4-cu132-20260624}
    container_name: ${NAME:-ds4-v5-cutlass}
    network_mode: host
    ipc: host
    shm_size: 32g
    gpus: all
    ulimits:
      memlock: -1
      nofile: 1048576
      stack: 67108864
    volumes:
      - /root/.cache/huggingface:/root/.cache/huggingface
      - ${CACHE_ROOT:-/root/.cache/vllm-ds4-v5-cutlass}:/cache
    environment:
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1}
      CUTE_DSL_ARCH: sm_120a
      NCCL_IB_DISABLE: "1"
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      VLLM_PREFIX_CACHE_RETENTION_INTERVAL: "4096"
      VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS: "1"
      VLLM_USE_V2_MODEL_RUNNER: "1"
      MODEL_PATH: ${MODEL_PATH:-/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136}
      PORT: ${PORT:-8000}
    command:
      - /bin/bash
      - -lc
      - |
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
        exec vllm serve "$${MODEL_PATH}" \
          --served-model-name DeepSeek-V4-Flash \
          --host 0.0.0.0 \
          --port "$${PORT}" \
          --trust-remote-code \
          --kv-cache-dtype fp8 \
          --block-size 256 \
          --tensor-parallel-size 2 \
          --gpu-memory-utilization 0.86 \
          --max-model-len 128000 \
          --max-num-seqs 1 \
          --max-num-batched-tokens 8192 \
          --max-cudagraph-capture-size 4 \
          --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \
          --kernel-config.moe_backend flashinfer_cutlass \
          --async-scheduling \
          --no-scheduler-reserve-full-isl \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --enable-flashinfer-autotune \
          --tokenizer-mode deepseek_v4 \
          --tool-call-parser deepseek_v4 \
          --reasoning-parser deepseek_v4 \
          --enable-auto-tool-choice \
          --default-chat-template-kwargs.thinking=true \
          --default-chat-template-kwargs.reasoning_effort=high
```

## Validation Status

| Variant | TP | MTP | Startup | Decode | Prefill |
|---|---:|---:|---|---:|---:|
| B12X v5 image | 2 | off | fails in B12X cache-page view | not measured | not measured |
| FlashInfer/CUTLASS v5 image | 2 | off | loads then worker exits after graph capture | not measured | not measured |
| Legacy Lucifer/CUTLASS v3/v4 | 2/4 | off/on | known-good reference | see v3/v4 | see v3/v4 |

Use v3/v4 for production DS4F until the B12X cache-page OOB and current
CUTLASS worker-exit issue are fixed on the Eldritch image.
