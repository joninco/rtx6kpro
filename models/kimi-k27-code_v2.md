# Kimi-K2.7-Code v2 on Eldritch

This page documents Kimi-K2.7-Code on the shared Eldritch final image with the
Kimi-K2.6 DFlash draft. This is the successor to `kimi-k27-code.md`.

## Image

```text
voipmonitor/vllm:eldritch-final-vfcc6141-b12x284a2ea-cu132-20260626
voipmonitor/vllm@sha256:dd41066fc2bd00fbc9446a78a386a3fe3700d42a4553ddf7a5bcb304ba200f86
```

| Component | Revision |
|---|---|
| vLLM | `codex/eldritch-final-20260626 @ fcc614141e5e9ab18cb304c476f7feed2a9552e3` |
| B12X | `284a2eae83754ee1abd31c37b9ca66b68e20b8a8` |
| FlashInfer | `25dd814e03791e370f96c3148242f0dc8de504ac` |
| DeepGEMM | `2073ddb2814892014c33ef4cd1c7d4c148baf1fe` |

See [`eldritch-final-docker.md`](./eldritch-final-docker.md) for the full
Docker build recipe and component pins.

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
| TP / DCP | `8 / 1` |
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

Use `--max-cudagraph-capture-size=8` or higher. For production testing use
`max_num_seqs=64` and graph cap `64` or higher. `max_num_seqs=1` plus graph cap
`8` is only a fast one-client debug profile.

## Docker Compose

```yaml
services:
  kimi:
    image: ${IMAGE:-voipmonitor/vllm:eldritch-final-vfcc6141-b12x284a2ea-cu132-20260626}
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
          --decode-context-parallel-size 1 \
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
IMAGE=voipmonitor/vllm:eldritch-final-vfcc6141-b12x284a2ea-cu132-20260626
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
  "$IMAGE" /bin/sh -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS VLLM_USE_B12X_FP8_GEMM; exec vllm serve moonshotai/Kimi-K2.7-Code --served-model-name Kimi-K2.7-Code --host 0.0.0.0 --port 8000 --trust-remote-code --tensor-parallel-size 8 --decode-context-parallel-size 1 --kv-cache-dtype fp8 --attention-backend TRITON_MLA --gpu-memory-utilization 0.94 --max-model-len 262144 --max-num-seqs 64 --max-num-batched-tokens 8192 --max-cudagraph-capture-size 64 --mm-processor-cache-gb 0 --mm-encoder-tp-mode weights --reasoning-parser kimi_k2 --tool-call-parser kimi_k2 --enable-auto-tool-choice --async-scheduling --enable-chunked-prefill --enable-prefix-caching --load-format fastsafetensors --speculative-config "{\"model\":\"'$DRAFT'\",\"method\":\"dflash\",\"num_speculative_tokens\":7,\"attention_backend\":\"TRITON_ATTN\"}"'
```

## Validation

Startup and short-context generation are validated on the final image.

A debug run with graph cap `4` fails because DFlash 7 requires graph sizes
rounded to a multiple of 8. Use graph cap `8` or higher.

Measured on 8x RTX 6000 Pro Blackwell, TP8/DCP1, final image. These are
one-client smoke measurements with `max_num_seqs=1` and graph cap `8`; rerun a
full sweep with `max_num_seqs=64` before treating them as production throughput.

| Test | Result |
|---|---:|
| No-spec short smoke | 104.8 tok/s, CJK 0, `finish=stop` |
| No-spec 30k-context smoke | 91.4 tok/s, CJK 0, `finish=stop` |
| DFlash7 short smoke | 219.5 tok/s, CJK 0, `finish=stop` |
| DFlash7 30k-context smoke | 210.6 tok/s, CJK 0, `finish=stop` |
| DFlash7 KV cache budget | 384,320 tokens |
| DFlash7 acceptance, 30k smoke | about `0.89/0.75/0.57/0.49/0.41/0.34/0.28` |

Commands:

```bash
python3 /mnt/test.py --port 8000 -L
python3 /mnt/test.py --port 8000 -c 30000
python3 /root/llm-inference-bench/llm_decode_bench.py --port 8000 --concurrency 1 --contexts 0k --duration 30 --skip-prefill
python3 /root/llm-inference-bench/llm_decode_bench.py --port 8000 --prefill-only --standalone-prefill --prefill-contexts 8k,64k --prefill-duration 10
```

Final image artifacts:

```text
/root/bench-results/final-eldritch-20260626/kimi27-final-dcp1-none-short.json
/root/bench-results/final-eldritch-20260626/kimi27-final-dcp1-none-30k.json
/root/bench-results/final-eldritch-20260626/kimi27-final-dcp1-dflash7-short.json
/root/bench-results/final-eldritch-20260626/kimi27-final-dcp1-dflash7-30k.json
```
