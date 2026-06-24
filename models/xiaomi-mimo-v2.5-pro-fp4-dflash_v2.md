# Xiaomi MiMo V2.5 Pro FP4-DFlash v2

This page documents the baked Eldritch vLLM path for
`XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash` on RTX 6000 Pro Blackwell. The v1/v2
overlay is no longer required; the MiMo padded-V DFlash fix is included in the
image.

## Image

```text
voipmonitor/vllm:eldritch-enlightenment-v9e8b0ab-b12x21e5cd4-cu132-20260624
voipmonitor/vllm@sha256:0158205fd17403dc755dbe34f76133edbbf2caa5c5bec4580aea99af2916bf9e
```

| Component | Revision |
|---|---|
| vLLM | `codex/eldritch-all-experiments-tp6fix-20260624 @ 9e8b0abcfd1986415812f30cbcffe37e346e7755` |
| MiMo fix | local PR #43 equivalent, baked into the vLLM branch |
| B12X | `21e5cd4d420b5ad5a68491416ae452599dbe0b5f` |
| FlashInfer | `b3baedbbef2686df91b6dc43818ee56fe26ceba2` |
| DeepGEMM | `14073b4e1e706506e193231209738c848d092a1f` |

## Model

```text
XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash
/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro-FP4-DFlash/snapshots/b754e6c86008bdb5cc901308dda5a38173ec7276
```

Do not pass `--max-model-len` for the standard launch. The model reports
`max_model_len=1048576`.

## Backend Notes

MiMo has asymmetric K/V head dimensions:

```text
head_dim=192
v_head_dim=128
```

The fast DFlash path keeps target and draft on standard `TRITON_ATTN`, pads the
V cache to the K head size, and slices the output back before `o_proj`. The slow
path is the automatic `TRITON_ATTN_DIFFKV` / `FLASH_ATTN_DIFFKV` selection.

Expected fast markers:

```text
Using AttentionBackendEnum.TRITON_ATTN backend.
kernel_unified_attention
```

Bad slow marker:

```text
kernel_unified_attention_diffkv
```

## Docker Compose

```yaml
services:
  mimo25-dflash:
    image: ${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v9e8b0ab-b12x21e5cd4-cu132-20260624}
    container_name: ${NAME:-mimo25-dflash-v2}
    network_mode: host
    ipc: host
    shm_size: 32g
    gpus: all
    ulimits:
      memlock:
        soft: -1
        hard: -1
      nofile:
        soft: 1048576
        hard: 1048576
      stack: 67108864
    volumes:
      - /root/.cache/huggingface:/root/.cache/huggingface
      - ${CACHE_ROOT:-/root/.cache/vllm-mimo25-dflash-v2}:/cache
    environment:
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUTE_DSL_ARCH: sm_120a
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      SAFETENSORS_FAST_GPU: "1"
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: "1"
      VLLM_CACHE_DIR: /cache/jit/vllm
      TORCH_EXTENSIONS_DIR: /cache/torch_extensions
      TORCHINDUCTOR_CACHE_DIR: /cache/torchinductor
      TRITON_CACHE_DIR: /cache/triton
      XDG_CACHE_HOME: /cache
      FLASHINFER_WORKSPACE_BASE: /cache/jit/flashinfer
      NCCL_P2P_LEVEL: SYS
      NCCL_IB_DISABLE: "1"
      NCCL_PROTO: LL,LL128,Simple
      VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS: "1"
      MODEL_PATH: ${MODEL_PATH:-/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro-FP4-DFlash/snapshots/b754e6c86008bdb5cc901308dda5a38173ec7276}
      PORT: ${PORT:-8000}
    command:
      - /bin/sh
      - -lc
      - |
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS \
          VLLM_USE_B12X_MOE B12X_MOE_FORCE_A16 VLLM_USE_B12X_FP8_GEMM \
          VLLM_PCIE_ALLREDUCE_BACKEND VLLM_ENABLE_PCIE_ALLREDUCE \
          VLLM_DISABLED_KERNELS VLLM_CPP_AR_1STAGE_NCCL_CUTOFF \
          VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS
        exec vllm serve "$$MODEL_PATH" \
          --served-model-name mimo-v25-pro-fp4-dflash \
          --host 0.0.0.0 \
          --port "$$PORT" \
          --trust-remote-code \
          --kv-cache-dtype fp8 \
          --block-size 64 \
          --tensor-parallel-size 8 \
          --gpu-memory-utilization 0.90 \
          --max-num-seqs 64 \
          --max-num-batched-tokens 16384 \
          --max-cudagraph-capture-size 128 \
          --attention-backend TRITON_ATTN \
          --kernel-config.moe_backend flashinfer_cutlass \
          --kernel-config.linear_backend b12x \
          --reasoning-parser mimo \
          --tool-call-parser mimo \
          --enable-auto-tool-choice \
          --compilation-config '{"cudagraph_mode":"PIECEWISE","custom_ops":["all"]}' \
          --async-scheduling \
          --no-scheduler-reserve-full-isl \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --speculative-config "{\"model\":\"$$MODEL_PATH/dflash\",\"method\":\"dflash\",\"num_speculative_tokens\":7}"
```

## Single Docker Run

```bash
IMAGE=voipmonitor/vllm:eldritch-enlightenment-v9e8b0ab-b12x21e5cd4-cu132-20260624
MODEL=/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro-FP4-DFlash/snapshots/b754e6c86008bdb5cc901308dda5a38173ec7276
CACHE=/root/.cache/vllm-mimo25-dflash-v2

docker rm -f mimo25-dflash-v2 2>/dev/null || true
mkdir -p "$CACHE"

docker run -d --name mimo25-dflash-v2 \
  --gpus all --ipc=host --shm-size=32g --network=host --init \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 --ulimit stack=67108864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$CACHE":/cache \
  -e MODEL_PATH="$MODEL" \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS=1 \
  "$IMAGE" /bin/sh -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS VLLM_USE_B12X_MOE B12X_MOE_FORCE_A16 VLLM_USE_B12X_FP8_GEMM VLLM_PCIE_ALLREDUCE_BACKEND VLLM_ENABLE_PCIE_ALLREDUCE; exec vllm serve "$MODEL_PATH" --served-model-name mimo-v25-pro-fp4-dflash --host 0.0.0.0 --port 8000 --trust-remote-code --kv-cache-dtype fp8 --block-size 64 --tensor-parallel-size 8 --gpu-memory-utilization 0.90 --max-num-seqs 64 --max-num-batched-tokens 16384 --max-cudagraph-capture-size 128 --attention-backend TRITON_ATTN --kernel-config.moe_backend flashinfer_cutlass --kernel-config.linear_backend b12x --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice --compilation-config "{\"cudagraph_mode\":\"PIECEWISE\",\"custom_ops\":[\"all\"]}" --async-scheduling --no-scheduler-reserve-full-isl --enable-chunked-prefill --enable-prefix-caching --speculative-config "{\"model\":\"$MODEL_PATH/dflash\",\"method\":\"dflash\",\"num_speculative_tokens\":7}"'
```

## Validation

Measured on 8x RTX 6000 Pro Blackwell, TP8, DFlash 7, FP8 KV,
`flashinfer_cutlass` MoE:

| Test | Result |
|---|---:|
| `/mnt/test.py -L` generation-only steady range | 240-280 tok/s |
| Decode cc1 ctx0 | 154.8 tok/s |
| Decode TTFT / ITL | 82 ms / 6.4 ms |
| Prefill 8k | 7,971 tok/s |
| Prefill 64k | 5,778 tok/s |
| Startup KV cache | 1,248,002 tokens |

Commands:

```bash
python3 /mnt/test.py --port 8000 -L
python3 /root/llm-inference-bench/llm_decode_bench.py --port 8000 --concurrency 1 --contexts 0k --duration 30 --skip-prefill
python3 /root/llm-inference-bench/llm_decode_bench.py --port 8000 --prefill-only --standalone-prefill --prefill-contexts 8k,64k --prefill-duration 10
```
