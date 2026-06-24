# Kimi-K2.7-Code v2 on Eldritch

This page documents Kimi-K2.7-Code on the shared Eldritch image with the
Kimi-K2.6 DFlash draft. This is the successor to `kimi-k27-code.md`.

## Image

```text
voipmonitor/vllm:eldritch-enlightenment-v9e8b0ab-b12x21e5cd4-cu132-20260624
voipmonitor/vllm@sha256:0158205fd17403dc755dbe34f76133edbbf2caa5c5bec4580aea99af2916bf9e
```

| Component | Revision |
|---|---|
| vLLM | `codex/eldritch-all-experiments-tp6fix-20260624 @ 9e8b0abcfd1986415812f30cbcffe37e346e7755` |
| B12X | `21e5cd4d420b5ad5a68491416ae452599dbe0b5f` |
| FlashInfer | `b3baedbbef2686df91b6dc43818ee56fe26ceba2` |
| DeepGEMM | `14073b4e1e706506e193231209738c848d092a1f` |

## Models

Target:

```text
moonshotai/Kimi-K2.7-Code
```

DFlash draft:

```text
/root/.cache/huggingface/hub/models--SubSir--Kimi-K2.6-DFlash-tmp/snapshots/171a2d3e68ec4050abe66c298477056b2fc2d40a
```

## Runtime

| Setting | Value |
|---|---|
| TP / DCP | `8 / 4` |
| Target attention | `TRITON_MLA` |
| Draft attention | `TRITON_ATTN` |
| KV cache | `fp8` |
| Runner | V2 |
| DFlash tokens | `7` |
| Tool parser | `kimi_k2` |
| Reasoning parser | `kimi_k2` |
| Custom allreduce | disabled for this profile |

Important: with DFlash `num_speculative_tokens=7`, CUDA graph capture sizes
must include a multiple of `8`. `--max-cudagraph-capture-size=4` fails with:

```text
No valid cudagraph sizes after rounding to multiple of 8
```

Use `--max-cudagraph-capture-size=8` or higher. For normal testing use
`max_num_seqs=64` and graph cap `64`; `max_num_seqs=1` is only a one-client
debug profile and can crash if two benchmark clients are started at once.

## Docker Compose

```yaml
services:
  kimi:
    image: ${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v9e8b0ab-b12x21e5cd4-cu132-20260624}
    container_name: ${NAME:-kimi-k27-code-v2}
    init: true
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
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k27-code-v2}:/cache
    environment:
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUTE_DSL_ARCH: sm_120a
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
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
      USE_NCCL_XML: "0"
      VLLM_ENABLE_PCIE_ALLREDUCE: "0"
      SAFETENSORS_FAST_GPU: "1"
      TARGET: ${TARGET:-moonshotai/Kimi-K2.7-Code}
      DRAFT: ${DRAFT:-/root/.cache/huggingface/hub/models--SubSir--Kimi-K2.6-DFlash-tmp/snapshots/171a2d3e68ec4050abe66c298477056b2fc2d40a}
      PORT: ${PORT:-8000}
    command:
      - /bin/sh
      - -lc
      - |
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS VLLM_USE_B12X_FP8_GEMM
        exec vllm serve "$$TARGET" \
          --served-model-name Kimi-K2.7-Code \
          --host 0.0.0.0 \
          --port "$$PORT" \
          --trust-remote-code \
          --tensor-parallel-size 8 \
          --decode-context-parallel-size 4 \
          --kv-cache-dtype fp8 \
          --attention-backend TRITON_MLA \
          --gpu-memory-utilization 0.94 \
          --max-model-len 262144 \
          --max-num-seqs 64 \
          --max-num-batched-tokens 8192 \
          --max-cudagraph-capture-size 64 \
          --mm-processor-cache-gb 0 \
          --mm-encoder-tp-mode weights \
          --reasoning-parser kimi_k2 \
          --tool-call-parser kimi_k2 \
          --enable-auto-tool-choice \
          --async-scheduling \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --load-format fastsafetensors \
          --speculative-config "{\"model\":\"$$DRAFT\",\"method\":\"dflash\",\"num_speculative_tokens\":7,\"attention_backend\":\"TRITON_ATTN\"}"
```

## Single Docker Run

```bash
IMAGE=voipmonitor/vllm:eldritch-enlightenment-v9e8b0ab-b12x21e5cd4-cu132-20260624
DRAFT=/root/.cache/huggingface/hub/models--SubSir--Kimi-K2.6-DFlash-tmp/snapshots/171a2d3e68ec4050abe66c298477056b2fc2d40a
CACHE=/root/.cache/vllm-kimi-k27-code-v2

docker rm -f kimi-k27-code-v2 2>/dev/null || true
mkdir -p "$CACHE"

docker run -d --name kimi-k27-code-v2 \
  --gpus all --ipc=host --shm-size=32g --network=host --init \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 --ulimit stack=67108864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$CACHE":/cache \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  "$IMAGE" /bin/sh -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS VLLM_USE_B12X_FP8_GEMM; exec vllm serve moonshotai/Kimi-K2.7-Code --served-model-name Kimi-K2.7-Code --host 0.0.0.0 --port 8000 --trust-remote-code --tensor-parallel-size 8 --decode-context-parallel-size 4 --kv-cache-dtype fp8 --attention-backend TRITON_MLA --gpu-memory-utilization 0.94 --max-model-len 262144 --max-num-seqs 64 --max-num-batched-tokens 8192 --max-cudagraph-capture-size 64 --mm-processor-cache-gb 0 --mm-encoder-tp-mode weights --reasoning-parser kimi_k2 --tool-call-parser kimi_k2 --enable-auto-tool-choice --async-scheduling --enable-chunked-prefill --enable-prefix-caching --load-format fastsafetensors --speculative-config "{\"model\":\"'$DRAFT'\",\"method\":\"dflash\",\"num_speculative_tokens\":7,\"attention_backend\":\"TRITON_ATTN\"}"'
```

## Validation

Startup and short-context generation are validated. The `max_num_seqs=64`,
graph-cap-64 profile reported KV cache budget `1,604,480` tokens
(`25070` blocks x `16`; local `401,120` x DCP4).

A debug run with graph cap `4` failed because DFlash 7 requires graph sizes
rounded to a multiple of 8. A separate debug run with `max_num_seqs=1` was
coherent but crashed after two benchmark clients were launched concurrently;
use the `max_num_seqs=64`, graph-cap-64 profile above for normal testing.

Measured on 8x RTX 6000 Pro Blackwell, TP8/DCP4, DFlash 7:

| Test | Result |
|---|---:|
| `/mnt/test.py -L` generation-only smoke | 180-250 tok/s, CJK 0 |
| Decode cc1 ctx0 | 107.9 tok/s |
| Decode TTFT / ITL | 61 ms / 9 ms |
| Prefill 8k | 7,493 tok/s |
| Prefill 64k | 6,143 tok/s |

Commands:

```bash
python3 /mnt/test.py --port 8000 -L
python3 /root/llm-inference-bench/llm_decode_bench.py --port 8000 --concurrency 1 --contexts 0k --duration 30 --skip-prefill
python3 /root/llm-inference-bench/llm_decode_bench.py --port 8000 --prefill-only --standalone-prefill --prefill-contexts 8k,64k --prefill-duration 10
```
