# GLM-5.2 v12 NVFP4 / B12X Dark Devotion

This page documents the reproducible GLM-5.2 NVFP4 serving stack built from
`local-inference-lab/vllm dev/dark-devotion` with the DCP global-top-k MTP fix
from PR #31, the B12X GLM odd-16-head prefill split needed for TP6/DCP3 and
TP6/DCP6, and B12X PR #15 for packed W4A16 ModelOpt scale reuse. The intended
default runtime is TP8 with DCP selectable at launch, B12X sparse MLA attention,
B12X NvFP4 MoE forced to A16, FP8 KV cache, and MTP3.

## Image

```text
voipmonitor/vllm:glm52-dark-devotion-release-vllmec65667-b12xaaf1891-scale-fix-cu132-20260622
voipmonitor/vllm@sha256:dc786d933582f77c421b18759411faf5b7d50863259a052e63cd72719604
```

Verified package versions:

| Component | Version / revision |
|---|---|
| vLLM package | `0.11.2.dev279+dark.devotion.release.ec65667.b12xaaf1891.fi9c5ed7c.cu132.20260622` |
| vLLM base | `local-inference-lab/vllm codex/dark-devotion-release-20260622 @ ec656676100a756912d6966c4232ea436c55d792` |
| vLLM fixes | dark-devotion release stack including the DCP global-top-k/MTP path |
| B12X | `voipmonitor/b12x codex/dark-devotion-pr14-pr15-20260622 @ aaf1891861ab86e78561326f13156d69a51a3ed8` |
| FlashInfer | `9c5ed7c194e7412780862491742fc655daaad6ac` |
| PyTorch | `2.12.0+cu132` |
| CUDA / cuBLAS | CUDA `13.2.x`, cuBLAS runtime `13.4.1.2` |
| NCCL | local inference NCCL `2.30.4` |
| Docker build repo | `local-inference-lab/blackwell-llm-docker @ 9eeb368feb3b1db7a9a0e2379cc443de2495509f` |

The image is a clean Docker build, not a runtime overlay.

## Docker Build

Build repository:

```text
https://github.com/local-inference-lab/blackwell-llm-docker @ 9eeb368feb3b1db7a9a0e2379cc443de2495509f
```

Build script used:

```text
/root/vllm/blackwell-llm-docker/build-dark-devotion-release-20260622-cu132.sh
```

Exact build invocation:

```bash
cd /root/vllm/blackwell-llm-docker

IMAGE=voipmonitor/vllm:glm52-dark-devotion-release-vllmec65667-b12xaaf1891-scale-fix-cu132-20260622 \
BUILD_BASE_IMAGE=0 PUSH_BASE_IMAGE=0 \
SYSTEM_BASE_IMAGE=voipmonitor/vllm:glm-kimi-cu132-system-base-20260608 \
BUILD_BASE_IMAGE_TAG=voipmonitor/vllm:glm-kimi-cu132-build-base-20260608 \
FLASHINFER_REF=main \
FLASHINFER_COMMIT=9c5ed7c194e7412780862491742fc655daaad6ac \
B12X_REPO=https://github.com/voipmonitor/b12x.git \
B12X_REF=codex/dark-devotion-pr14-pr15-20260622 \
B12X_COMMIT=aaf1891861ab86e78561326f13156d69a51a3ed8 \
VLLM_REF=codex/dark-devotion-release-20260622 \
VLLM_COMMIT=ec656676100a756912d6966c4232ea436c55d792 \
LAUNCHER_REF=codex/dark-devotion-release-20260622 \
LAUNCHER_COMMIT=ec656676100a756912d6966c4232ea436c55d792 \
VLLM_BUILD_VERSION=0.11.2.dev279+dark.devotion.release.ec65667.b12xaaf1891.fi9c5ed7c.cu132.20260622 \
./build-dark-devotion-release-20260622-cu132.sh
```

## Model

Default checkpoint:

```text
lukealonso/GLM-5.2-NVFP4
/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522
```

GLM-5.2 requires the index cache sparsity override until upstream vLLM picks the
pattern from model config:

```json
{"use_index_cache": true, "index_topk_pattern": "FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}
```

## Runtime Defaults

These are the important switches:

| Setting | Value |
|---|---|
| `--attention-backend` | `B12X_MLA_SPARSE` |
| `--moe-backend` | `b12x` |
| `--quantization` | `modelopt_fp4` |
| `--kv-cache-dtype` | `fp8` |
| `B12X_MOE_FORCE_A16` | `1` |
| `B12X_W4A16_TC_DECODE` | `1` |
| `VLLM_USE_B12X_SPARSE_INDEXER` | `1` |
| `VLLM_DCP_GLOBAL_TOPK` | `1` |
| `VLLM_DCP_SHARD_DRAFT` | `1` |
| `MTP` default | `3`, probabilistic draft sampling |
| DCP default | choose `1`, `2`, `4`, or `8` at launch |

`VLLM_DCP_GLOBAL_TOPK=1` is default in this code path, but it is set explicitly
below for reproducibility. The top-k path is global across DCP ranks and uses
the B12X sparse indexer. `VLLM_DCP_SHARD_DRAFT=1` is required so MTP draft KV is
not replicated per DCP rank.

Do not set `NCCL_GRAPH_FILE` to an empty string. If no XML topology file is
used, unset it before starting vLLM.

Do not set `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` for GLM-5.2. This vLLM build
rejects it for models without a sliding-window KV cache group.

## Docker Compose

This compose file is parameterized. Change `DCP_SIZE` and `MTP_TOKENS` without
editing the command.

```yaml
services:
  glm52:
    image: ${IMAGE:-voipmonitor/vllm:glm52-dark-devotion-release-vllmec65667-b12xaaf1891-scale-fix-cu132-20260622}
    container_name: ${NAME:-glm52-v12}
    network_mode: host
    ipc: host
    shm_size: 32g
    gpus: all
    init: true
    ulimits:
      memlock:
        soft: -1
        hard: -1
      nofile:
        soft: 1048576
        hard: 1048576
      stack: 67108864
    volumes:
      - ${HF_CACHE:-/root/.cache/huggingface}:/root/.cache/huggingface
      - ${CACHE_ROOT:-/root/.cache/vllm-glm52-v12}:/cache
    environment:
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUDA_DEVICE_MAX_CONNECTIONS: "32"
      CUTE_DSL_ARCH: sm_120a
      OMP_NUM_THREADS: "16"
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      SAFETENSORS_FAST_GPU: "1"
      NCCL_IB_DISABLE: "1"
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      VLLM_WORKER_MULTIPROC_METHOD: spawn
      VLLM_USE_AOT_COMPILE: "1"
      VLLM_USE_BREAKABLE_CUDAGRAPH: "0"
      VLLM_USE_MEGA_AOT_ARTIFACT: "1"
      VLLM_USE_FLASHINFER_SAMPLER: "1"
      VLLM_USE_B12X_FP8_GEMM: "1"
      VLLM_USE_B12X_MOE: "1"
      VLLM_USE_B12X_SPARSE_INDEXER: "1"
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_ENABLE_PCIE_ALLREDUCE: "1"
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
      VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: 64KB
      USES_B12X: "True"
      B12X_DENSE_SPLITK_TURBO: "1"
      B12X_W4A16_TC_DECODE: "1"
      B12X_MOE_FORCE_A16: ${B12X_MOE_FORCE_A16:-1}
      VLLM_DCP_GLOBAL_TOPK: "1"
      VLLM_DCP_SHARD_DRAFT: "1"
      VLLM_DCP_GLOBAL_TOPK_PREFILL_ONLY: "0"
      VLLM_DCP_TOPK_FORCE_DEEPGEMM: "0"
      VLLM_CACHE_DIR: /cache/jit/vllm
      TRITON_CACHE_DIR: /cache/jit/triton
      TORCH_EXTENSIONS_DIR: /cache/jit/torch_extensions
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/torchinductor
      FLASHINFER_WORKSPACE_BASE: /cache/jit/flashinfer
      XDG_CACHE_HOME: /cache/jit
      MODEL: ${MODEL:-/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522}
      PORT: ${PORT:-5543}
      TP_SIZE: ${TP_SIZE:-8}
      DCP_SIZE: ${DCP_SIZE:-4}
      MTP_TOKENS: ${MTP_TOKENS:-3}
      GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION:-0.955}
      MAX_MODEL_LEN: ${MAX_MODEL_LEN:-256000}
      MAX_NUM_SEQS: ${MAX_NUM_SEQS:-32}
      MAX_NUM_BATCHED_TOKENS: ${MAX_NUM_BATCHED_TOKENS:-8192}
      MAX_CUDAGRAPH_CAPTURE_SIZE: ${MAX_CUDAGRAPH_CAPTURE_SIZE:-128}
      GLM52_INDEX_TOPK_PATTERN: FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS
    command:
      - /bin/bash
      - -lc
      - |
        set -euo pipefail
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
        HF_OVERRIDES="$(printf '{"use_index_cache":true,"index_topk_pattern":"%s"}' "$$GLM52_INDEX_TOPK_PATTERN")"
        SPEC_ARGS=""
        if [ "$$MTP_TOKENS" != "0" ]; then
          SPEC_ARGS="--speculative-config {\"model\":\"$$MODEL\",\"method\":\"mtp\",\"num_speculative_tokens\":$$MTP_TOKENS,\"moe_backend\":\"b12x\",\"draft_sample_method\":\"probabilistic\"}"
        fi
        exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve "$$MODEL" \
          --served-model-name GLM-5.2-NVFP4 \
          --trust-remote-code \
          --host 0.0.0.0 \
          --port "$$PORT" \
          --tensor-parallel-size "$$TP_SIZE" \
          --pipeline-parallel-size 1 \
          --max-model-len "$$MAX_MODEL_LEN" \
          --decode-context-parallel-size "$$DCP_SIZE" \
          --dcp-comm-backend ag_rs \
          --dcp-kv-cache-interleave-size 1 \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --load-format fastsafetensors \
          --async-scheduling \
          -cc.pass_config.fuse_allreduce_rms=True \
          --gpu-memory-utilization "$$GPU_MEMORY_UTILIZATION" \
          --max-num-batched-tokens "$$MAX_NUM_BATCHED_TOKENS" \
          --max-num-seqs "$$MAX_NUM_SEQS" \
          --max-cudagraph-capture-size "$$MAX_CUDAGRAPH_CAPTURE_SIZE" \
          --attention-backend B12X_MLA_SPARSE \
          --moe-backend b12x \
          --kv-cache-dtype fp8 \
          --tool-call-parser glm47 \
          --enable-auto-tool-choice \
          --reasoning-parser glm45 \
          --hf-overrides "$$HF_OVERRIDES" \
          --quantization modelopt_fp4 \
          $$SPEC_ARGS
```

Examples:

```bash
# DCP1, no MTP
DCP_SIZE=1 MTP_TOKENS=0 PORT=5541 docker compose -f glm52-v12.compose.yaml up -d

# DCP4, default MTP3
DCP_SIZE=4 MTP_TOKENS=3 PORT=5543 docker compose -f glm52-v12.compose.yaml up -d
```

## Single Docker Run

```bash
docker rm -f glm52-v12 2>/dev/null || true
mkdir -p /root/.cache/vllm-glm52-v12

docker run -d \
  --init \
  --name glm52-v12 \
  --gpus all --runtime nvidia \
  --ipc host --shm-size 32g --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/.cache/vllm-glm52-v12:/cache \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e OMP_NUM_THREADS=16 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_FP8_GEMM=1 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB \
  -e USES_B12X=True \
  -e B12X_DENSE_SPLITK_TURBO=1 \
  -e B12X_W4A16_TC_DECODE=1 \
  -e B12X_MOE_FORCE_A16=1 \
  -e VLLM_DCP_GLOBAL_TOPK=1 \
  -e VLLM_DCP_SHARD_DRAFT=1 \
  -e VLLM_DCP_GLOBAL_TOPK_PREFILL_ONLY=0 \
  -e VLLM_DCP_TOPK_FORCE_DEEPGEMM=0 \
  -e VLLM_CACHE_DIR=/cache/jit/vllm \
  -e TRITON_CACHE_DIR=/cache/jit/triton \
  -e TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/jit/torchinductor \
  -e FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer \
  -e XDG_CACHE_HOME=/cache/jit \
  --entrypoint bash \
  voipmonitor/vllm:glm52-dark-devotion-release-vllmec65667-b12xaaf1891-scale-fix-cu132-20260622 \
  -lc 'set -euo pipefail
    unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS VLLM_PREFIX_CACHE_RETENTION_INTERVAL
    MODEL=/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522
    PATTERN=FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS
    exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve "$MODEL" \
      --served-model-name GLM-5.2-NVFP4 \
      --trust-remote-code --host 0.0.0.0 --port 5543 \
      --tensor-parallel-size 8 --pipeline-parallel-size 1 \
      --max-model-len 256000 \
      --decode-context-parallel-size 4 \
      --dcp-comm-backend ag_rs \
      --dcp-kv-cache-interleave-size 1 \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --load-format fastsafetensors \
      --async-scheduling \
      -cc.pass_config.fuse_allreduce_rms=True \
      --gpu-memory-utilization 0.955 \
      --max-num-batched-tokens 8192 \
      --max-num-seqs 32 \
      --max-cudagraph-capture-size 128 \
      --attention-backend B12X_MLA_SPARSE \
      --moe-backend b12x \
      --kv-cache-dtype fp8 \
      --tool-call-parser glm47 \
      --enable-auto-tool-choice \
      --reasoning-parser glm45 \
      --hf-overrides "{\"use_index_cache\":true,\"index_topk_pattern\":\"$PATTERN\"}" \
      --quantization modelopt_fp4 \
      --speculative-config "{\"model\":\"$MODEL\",\"method\":\"mtp\",\"num_speculative_tokens\":3,\"moe_backend\":\"b12x\",\"draft_sample_method\":\"probabilistic\"}"'
```

For no-MTP runs, remove `--speculative-config ...` or set `MTP_TOKENS=0` in the
compose recipe.

## Speed Results

The full sweep tables below are the current v12 baseline for the PR31 DCP
global-top-k path. The release image above adds the B12X packed-scale memory fix
and keeps the same runtime semantics; the clean-image smoke and KV checks below
were rerun on the release image.

Full sweep settings used for the tables:

```text
TP=8, max_model_len=256000, max_num_seqs=32, max_cudagraph_capture_size=128,
max_num_batched_tokens=8192, gpu_memory_utilization=0.88, FP8 KV, B12X A16 MoE.
```

The release image B12X update reuses packed W4A16 ModelOpt E4M3/K16 scale
layout instead of retaining native NVFP4 scale grids for the forced-A16 path.
It also keeps the legacy E4M3 scale decode path for non-A16 mode. Clean-image
TP8/DCP1/no-MTP smoke on the release image with `max_num_seqs=1`, graph capture
`4`, and `gpu_memory_utilization=0.94` produced coherent output for both
`B12X_MOE_FORCE_A16=0` and `B12X_MOE_FORCE_A16=1`. The A16 path loaded at
`55.01 GiB/GPU` with `649,600` KV tokens and seven `/mnt/test.py -L` samples:
mean `77.41 tok/s`, median `77.36 tok/s`, min/max `77.32/77.53 tok/s`, CJK
`0/7`.

The production launch profile now uses `gpu_memory_utilization=0.955` for a
larger KV cache. This is intentionally aggressive: TP8/DCP4/MTP3/A16 leaves
roughly 0.7 GiB/GPU after graph capture on RTX PRO 6000 Blackwell. Use `0.94`
if the host has extra background VRAM use or you need more operational margin.

Benchmark command:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py --port 5543
```

### Decode ctx0 Aggregate tok/s

| DCP | MTP | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 | cc128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | off | 78.0 | 132.4 | 217.0 | 327.5 | 508.0 | 703.1 | - | - |
| 1 | 3 | 104.0 | 172.6 | 306.5 | 503.9 | 787.4 | 1,204 | - | - |
| 2 | off | 66.3 | 107.2 | 174.2 | 280.1 | 437.6 | 608.9 | 626.4 | - |
| 2 | 3 | 102.7 | 170.2 | 298.5 | 474.2 | 724.3 | 1,087 | 1,013 | - |
| 4 | off | 62.6 | 98.8 | 153.6 | 239.0 | 344.6 | 454.0 | 462.2 | 465.4 |
| 4 | 3 | 90.4 | 163.4 | 279.5 | 445.3 | 657.6 | 985.7 | 915.1 | 904.1 |
| 8 | off | 55.2 | 84.2 | 125.7 | 179.1 | 238.0 | 295.2 | 300.3 | 301.5 |
| 8 | 3 | 93.7 | 163.4 | 260.1 | 406.2 | 568.7 | 770.6 | 741.6 | 745.6 |

### Prefill And 128k Decode

| DCP | MTP | 8k prefill | 64k prefill | 128k prefill | 128k cc1 | 128k cc2 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | off | 4,721 | 4,454 | 4,128 | 73.2 | 121.9 |
| 1 | 3 | 4,485 | 4,364 | 4,035 | 88.3 | 142.0 |
| 2 | off | 4,051 | 4,098 | 3,947 | 63.6 | 103.3 |
| 2 | 3 | 3,958 | 4,008 | 3,861 | 89.6 | 157.0 |
| 4 | off | 3,081 | 3,111 | 3,065 | 61.3 | 97.0 |
| 4 | 3 | 3,005 | 3,043 | 2,994 | 92.5 | 152.1 |
| 8 | off | 2,063 | 2,078 | 2,064 | 53.5 | 83.4 |
| 8 | 3 | 2,023 | 2,038 | 2,024 | 95.7 | 154.9 |

### Startup KV Cache

Release-image TP8 startup profile:

```text
TP=8, MTP=3, B12X_MOE_FORCE_A16=1, max_model_len=256000,
max_num_seqs=32, max_cudagraph_capture_size=128,
max_num_batched_tokens=8192, gpu_memory_utilization=0.955, FP8 KV.
```

| TP | DCP | MTP | `gpu_memory_utilization` | GPU KV cache tokens | Max concurrency at 256k | Source |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 1 | 3 | 0.955 | ~642,944 | ~2.51x | estimated from DCP4 and measured DCP ratios |
| 8 | 2 | 3 | 0.955 | ~1,279,616 | ~5.00x | estimated from DCP4 and measured DCP ratios |
| 8 | 4 | 3 | 0.955 | 2,559,232 | 10.00x | measured |
| 8 | 8 | 3 | 0.955 | ~5,052,416 | ~19.74x | estimated from DCP4 and measured DCP ratios |

The TP8/DCP4/MTP3/A16 `0.955` run completed CUDA graph capture and an explicit
240k prompt request without OOM:

```text
prompt_tokens=240,025, completion_tokens=1, TTFT=84.971s
```

The safer `0.94` TP8/DCP4/MTP3/A16 profile measured `2,448,128` KV tokens and
also completed the 240k prompt request. Its 128k prefill benchmark was:

```text
128,879 prompt tokens, TTFT=44.24s, prefill=2,913 tok/s
```

Historical `0.88` startup measurements from the earlier v12 profile are useful
when a host needs more VRAM headroom:

| TP | DCP | MTP | `gpu_memory_utilization` | GPU KV cache tokens | Max concurrency at 256k |
|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 3 | 0.88 | 508,224 | 1.99x |
| 8 | 2 | 3 | 0.88 | 1,011,456 | 3.95x |
| 8 | 4 | 3 | 0.88 | 2,022,912 | 7.90x |
| 8 | 8 | 3 | 0.88 | 3,993,600 | 15.60x |

### Coding Peak

This is the `/mnt/test.py -L` Sieve-of-Eratosthenes coding prompt, measured on
DCP1/MTP5 with the previous v12 PR31/`tp6odd16` image and `max_num_seqs=32`,
graph capture `128`.

| Profile | Samples | Generation-only tok/s mean | Median | Max | CJK |
|---|---:|---:|---:|---:|---:|
| DCP1 MTP5 coding prompt | 30 | 159.1 | 159.0 | 178.9 | 0/30 |

The benchmark harness also has a default-off `--coding-peak` mode for future
runs. It sends the same coding prompt without forcing temperature and records
five runs by default.

### TP6 Launch Profiles

TP6 supports DCP values that divide TP6: `1`, `2`, `3`, and `6`. Use host GPUs
`8-13` in these examples only to keep them isolated from an existing TP8
service. On a normal host, use any six visible GPUs.

```bash
# TP6 / DCP6 / MTP3 KV-discovery profile
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13 \
TP_SIZE=6 DCP_SIZE=6 MTP_TOKENS=3 \
PORT=5538 \
GPU_MEMORY_UTILIZATION=0.97 \
MAX_MODEL_LEN=-1 \
MAX_NUM_SEQS=32 \
MAX_NUM_BATCHED_TOKENS=8192 \
MAX_CUDAGRAPH_CAPTURE_SIZE=128 \
docker compose -f glm52-v12.compose.yaml up -d

# TP6 / DCP6 / MTP1
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13 \
TP_SIZE=6 DCP_SIZE=6 MTP_TOKENS=1 \
PORT=5538 \
GPU_MEMORY_UTILIZATION=0.97 \
MAX_MODEL_LEN=-1 \
MAX_NUM_SEQS=32 \
MAX_NUM_BATCHED_TOKENS=8192 \
MAX_CUDAGRAPH_CAPTURE_SIZE=128 \
docker compose -f glm52-v12.compose.yaml up -d

# Same run without forced A16 MoE
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13 \
TP_SIZE=6 DCP_SIZE=6 MTP_TOKENS=3 \
B12X_MOE_FORCE_A16=0 \
PORT=5538 \
GPU_MEMORY_UTILIZATION=0.97 \
MAX_MODEL_LEN=-1 \
MAX_NUM_SEQS=32 \
MAX_NUM_BATCHED_TOKENS=8192 \
MAX_CUDAGRAPH_CAPTURE_SIZE=128 \
docker compose -f glm52-v12.compose.yaml up -d
```

For DCP1/DCP2/DCP3, change only `DCP_SIZE`. For no-MTP runs, set
`MTP_TOKENS=0`. For production serving, replace `MAX_MODEL_LEN=-1` with the
context length you want to expose.

Release-image TP6/MTP3/A16 KV budget measurement, host GPUs `8-13`, FP8 KV,
`max_model_len=-1`, `max_num_seqs=32`, `max_cudagraph_capture_size=128`,
`max_num_batched_tokens=8192`, `gpu_memory_utilization=0.97`:

| TP | DCP | MTP | `B12X_MOE_FORCE_A16` | Model load per GPU | GPU KV cache tokens |
|---:|---:|---:|---:|---:|---:|
| 6 | 1 | 3 | 1 | 81.65 GiB | 170,304 |
| 6 | 2 | 3 | 1 | 81.90 GiB | 335,744 |
| 6 | 3 | 3 | 1 | 82.02 GiB | 503,616 |
| 6 | 6 | 3 | 1 | 82.27 GiB | 1,007,232 |

Estimated TP6/MTP3/A16 KV budget at `gpu_memory_utilization=0.955`, extrapolated
from the measured `0.97` runs:

| TP | DCP | MTP | `gpu_memory_utilization` | Estimated GPU KV cache tokens |
|---:|---:|---:|---:|---:|
| 6 | 1 | 3 | 0.955 | ~141,632 |
| 6 | 2 | 3 | 0.955 | ~278,528 |
| 6 | 3 | 3 | 0.955 | ~417,792 |
| 6 | 6 | 3 | 0.955 | ~835,584 |

The earlier large A16 VRAM penalty is fixed in this release image. Current
TP6/DCP6/MTP3/A16 model load is about `82.27 GiB/GPU`, not the old
`89.5 GiB/GPU` value.

### Historical Reduced TP6 Debug Smoke

The previous `tp6odd16` image first validated the B12X GLM odd-16-head prefill
split needed for TP6 DCP layouts. This is historical smoke-test data; use the
release-image KV table above for current capacity planning.

```text
TP=6, DCP=3 or 6, MTP=3, max_model_len=24576, max_num_seqs=1,
max_cudagraph_capture_size=4, max_num_batched_tokens=2048,
gpu_memory_utilization=0.966, SAFETENSORS_FAST_GPU=0, load-format=auto.
```

Result:

| TP | DCP | GPU KV cache tokens | Max concurrency | Graph capture | Coherence smoke |
|---:|---:|---:|---:|---:|---|
| 6 | 3 | 88,512 | 3.60x | 17 s | 4 complete `/mnt/test.py -L -c 1000 --hide-reasoning` iterations before timeout; CJK 0; generation-only about 71-80 tok/s |
| 6 | 6 | 170,112 | 6.92x | 18 s | 5 iterations before timeout; CJK 0; no duplicated-code pattern; generation-only about 77-81 tok/s |

DCP3 MTP acceptance windows were roughly `0.84-0.98 / 0.63-0.96 / 0.48-0.90`.
DCP6 examples were about `0.92 / 0.80 / 0.68`.

Use the reduced scheduler settings only for fast debugging. Use the TP8 recipe
above for production DCP1/2/4/8 runs; TP6 still needs a full throughput sweep.

## KLD

Reference BF16 logits:

```text
/root/kld/glm52_refs/bf16-b12xmlasparse-w1-ctx2048-s512-20260618/logits_0.safetensors
/root/kld/glm52_refs/decode_teacher_bf16_ref_ctx2048_t17_20260618.safetensors
```

The optional logits-export plumbing is in:

```text
https://github.com/local-inference-lab/vllm/pull/32
```

It adds `SamplingParams.return_prompt_logits` and `return_sample_logits` so KLD
capture does not need ad-hoc local patches.

Current v12 NVFP4/B12X/A16 result:

| Checkpoint/runtime | Prefill mean KLD | Decode KL model->BF16 | Decode KL BF16->model | Decode JS | Token match |
|---|---:|---:|---:|---:|---:|
| v12 NVFP4 B12X A16 | 0.0705969 | 0.000005873 | 0.000007896 | 0.000001629 | 17/17 |
| v11 NVFP4 B12X A16 | 0.067532 | 0.000009892 | 0.000013952 | 0.000002845 | 17/17 |
| v11 official FP8 | 0.079041 | 0.000085952 | 0.000045639 | 0.000011247 | 17/17 |
| v11 MXFP8 | 0.018720 | 0.000012423 | 0.000015746 | 0.000003378 | 17/17 |

Local v12 KLD artifacts:

```text
/root/kld/glm52_v12_kld_nvfp4_b12x_a16_20260621_144826
```

The BF16 reference upload script is:

```text
/root/kld/upload_glm52_bf16_refs_to_hf.py
```

Upload to Hugging Face is currently blocked because the available token for
user `festr2` is read-only. A write token is required to create or update the
reference-logits repository.

## Smoke Tests

```bash
curl -s http://127.0.0.1:5543/v1/models | python3 -m json.tool | head
python3 /mnt/test.py --port 5543 -L
docker logs -f glm52-v12
```

For fast startup while debugging a single request, use smaller scheduler
settings:

```text
MAX_NUM_SEQS=1
MAX_CUDAGRAPH_CAPTURE_SIZE=4
```

Use the full `MAX_NUM_SEQS=32` and graph capture `128` profile for production
and benchmark numbers above.
