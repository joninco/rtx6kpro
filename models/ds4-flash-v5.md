# DeepSeek-V4-Flash v5 Eldritch Packed-KV

This page documents the Eldritch DS4F image with B12X support for the upstream
DeepSeek-V4 packed KV-cache layout. This is the replacement for the earlier v5
notes that marked B12X as broken on Eldritch.

## Docker Image

```text
voipmonitor/vllm:eldritch-packedkv-vb923949-b12xf81d898-cu132-20260624
voipmonitor/vllm@sha256:a4a69ce28298c17a9b3fc4fc4e42fbfc33cacd001a6355ab7841c744596cfc4d
```

Build recipe:

```text
local-inference-lab/blackwell-llm-docker: build-eldritch-packedkv-cu132.sh
```

| Component | Revision |
|---|---|
| vLLM | `local-inference-lab/vllm codex/eldritch-all-experiments-packedkv-20260624 @ b9239499bd9305c55cdfc37e97dde518804c1920` |
| B12X | `voipmonitor/b12x codex/ds4-packedkv-stride-20260624 @ f81d8985e2c387d0dec5f9f310a4e7a45be72adc` |
| FlashInfer | `b3baedbbef2686df91b6dc43818ee56fe26ceba2` |
| DeepGEMM | `14073b4e1e706506e193231209738c848d092a1f` |
| CUTLASS | `d80a4e53b52b42550659a8696dab32705265e324` |
| CUDA / torch | CUDA `13.2`, torch `2.12.0+cu132` |

The image includes the DS4 packed-KV fix split across two PRs:

| PR | Purpose |
|---|---|
| `local-inference-lab/vllm#45` | vLLM DS4 B12X adapter exposes correctly offset packed page views and caches the static view wrapper per module. |
| `lukealonso/b12x#17` | B12X compressed MLA accepts strided packed page views and uses the physical byte span without materializing contiguous copies. |

## Model

```text
deepseek-ai/DeepSeek-V4-Flash
/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136
```

## Recommended B12X Compose

```yaml
services:
  ds4-b12x:
    image: ${IMAGE:-voipmonitor/vllm:eldritch-packedkv-vb923949-b12xf81d898-cu132-20260624}
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
      VLLM_USE_AOT_COMPILE: "1"
      VLLM_USE_BREAKABLE_CUDAGRAPH: "0"
      VLLM_USE_MEGA_AOT_ARTIFACT: "1"
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_USE_FLASHINFER_SAMPLER: "1"
      VLLM_USE_B12X_WO_PROJECTION: "1"
      VLLM_USE_B12X_MHC: "1"
      VLLM_USE_B12X_FP8_GEMM: "1"
      VLLM_USE_B12X_MOE: "1"
      VLLM_USE_B12X_SPARSE_INDEXER: "1"
      VLLM_ENABLE_PCIE_ALLREDUCE: "1"
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
      B12X_MLA_SM120_UNIFIED: "1"
      B12X_MHC_MAX_TOKENS: "16384"
      B12X_DENSE_SPLITK_TURBO: "1"
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
          --gpu-memory-utilization 0.88 \
          --max-model-len 262144 \
          --max-num-seqs 64 \
          --max-num-batched-tokens 8192 \
          --max-cudagraph-capture-size 192 \
          --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
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

## Single-Command Launch

```bash
docker rm -f ds4-v5-b12x 2>/dev/null || true
docker run -d \
  --name ds4-v5-b12x \
  --gpus all \
  --ipc host \
  --shm-size 32g \
  --network host \
  --ulimit memlock=-1 \
  --ulimit nofile=1048576 \
  --ulimit stack=67108864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/.cache/vllm-ds4-v5-b12x:/cache \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096 \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_WO_PROJECTION=1 \
  -e VLLM_USE_B12X_MHC=1 \
  -e VLLM_USE_B12X_FP8_GEMM=1 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e B12X_MLA_SM120_UNIFIED=1 \
  -e B12X_MHC_MAX_TOKENS=16384 \
  -e B12X_DENSE_SPLITK_TURBO=1 \
  -e B12X_W4A16_TC_DECODE=1 \
  voipmonitor/vllm:eldritch-packedkv-vb923949-b12xf81d898-cu132-20260624 \
  /bin/bash -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS && exec vllm serve /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136 \
    --served-model-name DeepSeek-V4-Flash \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.88 \
    --max-model-len 262144 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --max-cudagraph-capture-size 192 \
    --compilation-config '\''{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'\'' \
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
    --default-chat-template-kwargs.reasoning_effort=high'
```

For quick smoke/debug runs, use `--max-num-seqs 1` and
`--max-cudagraph-capture-size 8` to avoid long graph capture.

## Validation

Clean-image functional smoke, TP2/MTP off/B12X:

```text
/mnt/test.py --port 5627 -L -c 1200
CJK=0, coherent Python output, generation-only about 125 tok/s
Startup KV cache: 445,153 tokens
B12X PCIe all-reduce active
```

## Packed KV vs OldKV A/B

The oldKV layout was a temporary diagnostic overlay that restores the pre-packed
DS4 KV allocation. It is not recommended for production. The packed layout is
the upstream vLLM allocation from DS4 KV packing.

Decode was measured on the same clean image and same host GPUs `8,9`. The
monitor process ran inside a helper container restricted to those GPUs, so PCIe
rx/tx only includes that pair.

| Layout | C1 tok/s | C2 tok/s | C4 tok/s | C8 tok/s |
|---|---:|---:|---:|---:|
| Packed KV | 119.2 | 190.1 | 305.4 | 448.8 |
| OldKV overlay | 119.8 | 190.7 | 308.0 | 450.3* |

`*` The oldKV C8 cell was marked capacity-limited by the benchmark.

| Layout | C1 PCIe rx/tx MB/s | C2 PCIe rx/tx MB/s | C4 PCIe rx/tx MB/s | C8 PCIe rx/tx MB/s |
|---|---:|---:|---:|---:|
| Packed KV | 398 / 376 | 619 / 610 | 837 / 809 | 1284 / 1292 |
| OldKV overlay | 425 / 404 | 638 / 627 | 830 / 823 | 1089 / 1071* |

Prefill:

| Layout | 8k tok/s | 8k PCIe rx/tx MB/s | 64k tok/s | 64k PCIe rx/tx MB/s |
|---|---:|---:|---:|---:|
| Packed KV | 8,632 | 15191 / 15982 | 8,151 | 14681 / 15420 |
| OldKV overlay | 8,748 | 13255 / 13103 | 8,148 | 15089 / 14940 |

Delta summary:

| Metric | Packed vs oldKV |
|---|---:|
| Decode C1 tok/s | `-0.47%` |
| Decode C2 tok/s | `-0.34%` |
| Decode C4 tok/s | `-0.84%` |
| Decode C8 tok/s | `-0.34%` |
| Prefill 8k tok/s | `-1.33%` |
| Prefill 64k tok/s | `+0.04%` |

Interpretation: the packed-KV B12X path restores DS4 B12X throughput to the
oldKV range without reverting vLLM's packed allocation. Decode throughput is
within normal run noise. Prefill is unchanged at 64k and about 1.3% lower at 8k
in this sample. The PCIe columns are coarse `nvidia-smi dmon` diagnostics, not
per-kernel NCCL profiling.

Benchmark JSON artifacts:

```text
/root/bench-results/ds4-packedkv-f81d898-decode-cc1-8-gpu89-container.json
/root/bench-results/ds4-packedkv-f81d898-prefill-8k64k-gpu89-container.json
/root/bench-results/ds4-oldkv-f81d898-decode-cc1-8-gpu89-container.json
/root/bench-results/ds4-oldkv-f81d898-prefill-8k64k-gpu89-container.json
```

## Notes

- Use `B12X_MLA_SPARSE` for B12X attention.
- Use `FLASHINFER_MLA_SPARSE_DSV4` for the current FlashInfer DS4 attention
  backend name. The old `SPARSE_MLA_SM120` name is rejected by this image.
- Do not set `NCCL_GRAPH_FILE` to an empty string. Leave it unset unless a real
  XML graph file is provided.
