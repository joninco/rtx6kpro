# GLM-5.2 v12 NVFP4 / B12X Dark Devotion

This page documents the reproducible GLM-5.2 NVFP4 serving stack built from
`local-inference-lab/vllm dev/dark-devotion` with the DCP global-top-k MTP fix
from PR #31 and the B12X GLM odd-16-head prefill split needed for TP6/DCP3 and
TP6/DCP6. The intended default runtime is TP8 with DCP selectable at launch,
B12X sparse MLA attention, B12X NvFP4 MoE forced to A16, FP8 KV cache, and MTP3.

## Image

```text
voipmonitor/vllm:glm52-dark-devotion-pr31-tp6odd16-vllm79f154c-b12x1cfc6cf-cu132-20260621
voipmonitor/vllm@sha256:3ad851d7117613ecc8c76bcfb12adb794de0e78c6324b90b40f739f1138baba6
```

Verified package versions:

| Component | Version / revision |
|---|---|
| vLLM package | `0.11.2.dev279+dark.devotion.pr31.tp6odd16.79f154c.b12x1cfc6cf.fi9c5ed7c.cu132.20260621` |
| vLLM base | `local-inference-lab/vllm dev/dark-devotion @ 4e4a0b91a73d474374e8e5da528a24bb6a16b0eb` |
| vLLM fix | PR #31, `79f154c998acd315bd999c8909cfc24085c23f85` |
| B12X | `voipmonitor/b12x codex/glm-prefill-odd16-split-20260621 @ 1cfc6cffc0ccfd01d0d66c775a8de952eba12c09` |
| FlashInfer | `9c5ed7c194e7412780862491742fc655daaad6ac` |
| PyTorch | `2.12.0+cu132` |
| CUDA / cuBLAS | CUDA `13.2.x`, cuBLAS runtime `13.4.1.2` |
| NCCL | local inference NCCL `2.30.4` |
| Docker build repo | `local-inference-lab/blackwell-llm-docker @ 38e1c3f0654f6d1ff296cb7e9f860f089b6a115f` |

The image is a clean Docker build, not a runtime overlay.

## Docker Build

Build repository:

```text
https://github.com/local-inference-lab/blackwell-llm-docker @ 38e1c3f0654f6d1ff296cb7e9f860f089b6a115f
```

Build script used:

```text
/root/vllm/blackwell-llm-docker/build-dark-devotion-pr31-tp6odd16-cu132.sh
```

Exact build invocation:

```bash
cd /root/vllm/blackwell-llm-docker

IMAGE=voipmonitor/vllm:glm52-dark-devotion-pr31-tp6odd16-vllm79f154c-b12x1cfc6cf-cu132-20260621 \
BUILD_BASE_IMAGE=0 PUSH_BASE_IMAGE=0 \
SYSTEM_BASE_IMAGE=voipmonitor/vllm:glm-kimi-cu132-system-base-20260608 \
BUILD_BASE_IMAGE_TAG=voipmonitor/vllm:glm-kimi-cu132-build-base-20260608 \
FLASHINFER_REF=main \
FLASHINFER_COMMIT=9c5ed7c194e7412780862491742fc655daaad6ac \
B12X_REPO=https://github.com/voipmonitor/b12x.git \
B12X_REF=codex/glm-prefill-odd16-split-20260621 \
B12X_COMMIT=1cfc6cffc0ccfd01d0d66c775a8de952eba12c09 \
VLLM_REF=codex/dark-devotion-dcp4-mtp3-globaltopk-fix-20260621 \
VLLM_COMMIT=79f154c998acd315bd999c8909cfc24085c23f85 \
LAUNCHER_REF=codex/dark-devotion-dcp4-mtp3-globaltopk-fix-20260621 \
LAUNCHER_COMMIT=79f154c998acd315bd999c8909cfc24085c23f85 \
VLLM_BUILD_VERSION=0.11.2.dev279+dark.devotion.pr31.tp6odd16.79f154c.b12x1cfc6cf.fi9c5ed7c.cu132.20260621 \
./build-dark-devotion-pr31-tp6odd16-cu132.sh
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
    image: ${IMAGE:-voipmonitor/vllm:glm52-dark-devotion-pr31-tp6odd16-vllm79f154c-b12x1cfc6cf-cu132-20260621}
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
      B12X_MOE_FORCE_A16: "1"
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
      GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION:-0.88}
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
  voipmonitor/vllm:glm52-dark-devotion-pr31-tp6odd16-vllm79f154c-b12x1cfc6cf-cu132-20260621 \
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
      --gpu-memory-utilization 0.88 \
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

All speed results below used:

```text
TP=8, max_model_len=256000, max_num_seqs=32, max_cudagraph_capture_size=128,
max_num_batched_tokens=8192, gpu_memory_utilization=0.88, FP8 KV, B12X A16 MoE.
```

The `79f154c` rebuild only changes the DCP warmup barrier implementation from a
direct device-group/NCCL barrier to `dcp_group.barrier()`. A bounded smoke run on
DCP4/MTP3 with the debug scheduler profile (`max_num_seqs=4`,
`max_cudagraph_capture_size=24`) produced 15 complete `/mnt/test.py -L`
samples with generation-only mean `123.52 tok/s`, median `122.98 tok/s`, and
CJK `0/15`. The earlier same-profile reference was about `119.96 tok/s` mean /
`120.27 tok/s` median, so the barrier fix did not introduce a measurable decode
regression.

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

| DCP | MTP | GPU KV cache tokens | Max concurrency at 256k |
|---|---:|---:|---:|
| 1 | off | 431,168 | 1.68x |
| 1 | 3 | 403,328 | 1.58x |
| 2 | off | 856,832 | 3.35x |
| 2 | 3 | 798,848 | 3.12x |
| 4 | off | 1,713,664 | 6.69x |
| 4 | 3 | 1,597,696 | 6.24x |
| 8 | off | 3,411,968 | 13.33x |
| 8 | 3 | 3,075,072 | 12.01x |

### Coding Peak

This is the `/mnt/test.py -L` Sieve-of-Eratosthenes coding prompt, measured on
DCP1/MTP5 with the same image and `max_num_seqs=32`, graph capture `128`.

| Profile | Samples | Generation-only tok/s mean | Median | Max | CJK |
|---|---:|---:|---:|---:|---:|
| DCP1 MTP5 coding prompt | 30 | 159.1 | 159.0 | 178.9 | 0/30 |

The benchmark harness also has a default-off `--coding-peak` mode for future
runs. It sends the same coding prompt without forcing temperature and records
five runs by default.

### TP6 Debug Smoke

The `tp6odd16` image also includes the B12X GLM odd-16-head prefill split needed
for TP6 DCP layouts. This was validated as a startup/coherence smoke test, not a
full throughput sweep:

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

Use the same reduced scheduler settings for TP6/DCP6 debugging. Use the TP8
recipe above for production DCP1/2/4/8 runs until TP6 has a full sweep.

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
