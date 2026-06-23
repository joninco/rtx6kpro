# Xiaomi MiMo V2.5 Pro FP4-DFlash v2

Status: validated overlay runbook. The final Docker tag is still pending; once
the image is rebuilt with the MiMo padded-V patch baked in, replace the
`IMAGE` value below and remove the `mimo_v2.py` bind mount.

This page documents the fast vLLM path for
`XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash` on RTX 6000 Pro Blackwell.

## Model

```text
XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash
```

Validated local snapshot:

```text
/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro-FP4-DFlash/snapshots/b754e6c86008bdb5cc901308dda5a38173ec7276
```

Do not pass `--max-model-len` for the standard launch. The model reports
`max_model_len=1048576`.

## Required Patch

MiMo has asymmetric K/V head dimensions:

```text
head_dim=192
v_head_dim=128
```

The slow path is the newer automatic `TRITON_ATTN_DIFFKV` /
`FLASH_ATTN_DIFFKV` selection. That path is useful for target-only asymmetric
K/V serving because it avoids padding V cache to K head size. For MiMo DFlash
long-context serving it is much slower, because the target and draft paths no
longer stay on the same unified attention kernels.

Use PR:

```text
https://github.com/local-inference-lab/vllm/pull/43
```

The patch keeps MiMo on standard `TRITON_ATTN`, pads V to `head_dim` in the KV
cache, then slices the output back before `o_proj`. B12X paged attention remains
unchanged.

Expected log marker after the patch:

```text
Using AttentionBackendEnum.TRITON_ATTN backend.
kernel_unified_attention
```

Bad/slow marker:

```text
kernel_unified_attention_diffkv
```

## Current Image

Validated base image:

```text
voipmonitor/vllm:glm52-eldritch-enlightenment-vllmfa32916-b12x21e5cd4-pr14pr16-cu132-20260623
```

Final baked image:

```text
TODO: replace with final Docker tag after rebuilding with PR #43
```

## Expected Speed

Validation command:

```bash
python3 /mnt/test.py --port 8000 -c 100000 --quiet --json-summary -
```

Observed on 8x RTX 6000 Pro Blackwell, TP8, DFlash 7, FP8 KV,
`flashinfer_cutlass` MoE:

| Stack | Attention kernel | Long-context generation-only speed |
|---|---|---:|
| latest image before PR #43 | `kernel_unified_attention_diffkv` | ~93 tok/s |
| old known-good black-benediction overlay | `kernel_unified_attention` | ~215 tok/s |
| latest image + PR #43 overlay | `kernel_unified_attention` | ~227 tok/s |

Short-context coding smoke with `/mnt/test.py -L` is expected around
`260-270 tok/s` generation-only after warmup. Long-context first request may
show high TTFT if Triton kernels are still JIT compiling; rerun once before
recording steady-state speed.

## Docker Compose

Create a local checkout for the patch overlay:

```bash
git clone https://github.com/local-inference-lab/vllm.git /root/vllm-mimo-padv
cd /root/vllm-mimo-padv
git fetch origin pull/43/head:pr43-mimo-padv
git checkout pr43-mimo-padv
```

Compose:

```yaml
services:
  mimo25-dflash:
    image: ${IMAGE:-voipmonitor/vllm:glm52-eldritch-enlightenment-vllmfa32916-b12x21e5cd4-pr14pr16-cu132-20260623}
    container_name: ${NAME:-mimo25-dflash}
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
      # Remove this mount once the final Docker image contains PR #43.
      - ${VLLM_PATCH_ROOT:-/root/vllm-mimo-padv}/vllm/model_executor/models/mimo_v2.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/mimo_v2.py:ro
    environment:
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUTE_DSL_ARCH: sm_120a
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      SAFETENSORS_FAST_GPU: "1"
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
          --max-num-seqs 128 \
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

Start:

```bash
cat > /tmp/mimo25-dflash-v2.compose.yaml <<'EOF'
# paste the compose service above here
EOF

IMAGE=voipmonitor/vllm:glm52-eldritch-enlightenment-vllmfa32916-b12x21e5cd4-pr14pr16-cu132-20260623 \
NAME=mimo25-dflash \
CACHE_ROOT=/root/.cache/vllm-mimo25-dflash-v2 \
VLLM_PATCH_ROOT=/root/vllm-mimo-padv \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PORT=8000 \
docker compose -f /tmp/mimo25-dflash-v2.compose.yaml up -d
```

Stop:

```bash
docker compose -f /tmp/mimo25-dflash-v2.compose.yaml down
```

## Single Docker Run

```bash
IMAGE=voipmonitor/vllm:glm52-eldritch-enlightenment-vllmfa32916-b12x21e5cd4-pr14pr16-cu132-20260623
MODEL=/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro-FP4-DFlash/snapshots/b754e6c86008bdb5cc901308dda5a38173ec7276
CACHE=/root/.cache/vllm-mimo25-dflash-v2
VLLM_PATCH_ROOT=/root/vllm-mimo-padv

docker rm -f mimo25-dflash 2>/dev/null || true
mkdir -p "$CACHE"

docker run -d \
  --name mimo25-dflash \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$CACHE:/cache" \
  -v "$VLLM_PATCH_ROOT/vllm/model_executor/models/mimo_v2.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/mimo_v2.py:ro" \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUTE_DSL_ARCH=sm_120a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_CACHE_DIR=/cache/jit/vllm \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e XDG_CACHE_HOME=/cache \
  -e FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS=1 \
  "$IMAGE" \
  /bin/sh -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS VLLM_USE_B12X_MOE B12X_MOE_FORCE_A16 VLLM_USE_B12X_FP8_GEMM VLLM_PCIE_ALLREDUCE_BACKEND VLLM_ENABLE_PCIE_ALLREDUCE VLLM_DISABLED_KERNELS VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS; exec vllm serve "$@"' -- \
  "$MODEL" \
  --served-model-name mimo-v25-pro-fp4-dflash \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 64 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 \
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
  --speculative-config "{\"model\":\"$MODEL/dflash\",\"method\":\"dflash\",\"num_speculative_tokens\":7}"
```

## Validation

Readiness:

```bash
curl -fsS http://127.0.0.1:8000/v1/models | jq .
```

Backend log checks:

```bash
docker logs mimo25-dflash 2>&1 | grep -E 'TRITON_ATTN|DIFFKV|kernel_unified_attention|FLASHINFER_CUTLASS|GPU KV cache|Graph capturing|SpecDecoding' | tail -n 80
```

Smoke and long-context speed:

```bash
python3 /mnt/test.py --port 8000 -L
python3 /mnt/test.py --port 8000 -c 100000 --quiet --json-summary -
```

Expected:

```text
short context: ~260-270 tok/s generation-only
100k context: ~215-230 tok/s generation-only after warmup
chinese_count: 0
```

If long-context speed is around `90 tok/s`, inspect logs for
`kernel_unified_attention_diffkv`; the MiMo padded-V patch is missing or the
wrong file was mounted.

