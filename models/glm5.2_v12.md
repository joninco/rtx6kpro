# GLM-5.2 v12 NVFP4 on Dark Devotion

Measured and packaged on 2026-06-21 on the local 16x RTX PRO 6000
Blackwell host.

This page records the production GLM-5.2 v12 Docker image built from
`dev/dark-devotion` plus the GLM DCP global-top-k MTP fix from PR #31. The
main served profile is Luke's NVFP4 checkpoint with B12X MLA sparse attention,
B12X NvFP4 MoE forced to A16, FP8 KV cache, TP8, DCP4, and MTP3.

## Image

```text
voipmonitor/vllm:glm52-dark-devotion-pr31-dcpglobaltopk-mtp-cu132-20260621
```

Local image ID:

```text
sha256:f4917f67150958d18edf38ee2a5d36205075c1eac025d13d506e271e83ba370e
```

Pinned digest:

```text
voipmonitor/vllm@sha256:de96648aeade8a8189b3614d15d3bdec98f21dce3b3dc239262db5d06c52fc2e
```

Relevant source state:

| Component | Revision |
|---|---|
| CUDA runtime | `13.2.1` |
| cuBLAS runtime | `13.4.1.2` |
| NCCL local library | `2.30.4`, loaded via `/opt/libnccl-local-inference.so.2.30.4` |
| PyTorch | `2.12.0+cu132` |
| FlashInfer | `0.6.13+cu132` |
| vLLM base branch | `local-inference-lab/vllm dev/dark-devotion` |
| vLLM build base commit | `df8ad3b202c84937a23cfa9d93f7a3677da8ecde` |
| vLLM serving fix | PR #31, `000807e2b0e33277ac6b3ae51ae2e52d8472c9ab` |
| B12X branch | `lukealonso/b12x master` |
| B12X commit | `5af873a7b6c81fbf533ef96bede13fbf4744ad2a` |
| FlashInfer commit | `9c5ed7c194e7412780862491742fc655daaad6ac` |
| Docker build repo | `local-inference-lab/blackwell-llm-docker` |

The image verifies:

```text
vllm 0.11.2.dev279+dark.devotion.df8ad3b.b12x5af873a.fi9c5ed7c.dcpglobaltopk.mtptopkscores.cu132.20260621
torch 2.12.0+cu132
flashinfer 0.6.13+cu132
deepseek_mtp.py includes topk_scores_buffer for B12X DCP global-top-k
```

## What Changed

The current `dev/dark-devotion` stack already contains the B12X global top-k
sparse-indexer path. PR #31 adds the missing MTP-side support required to use
that path under DCP:

```text
https://github.com/local-inference-lab/vllm/pull/31
```

Runtime behavior:

| Setting | Status |
|---|---|
| `VLLM_DCP_GLOBAL_TOPK` | Defaults to enabled in code. Set to `1` explicitly in the recipe for reproducibility. |
| `VLLM_DCP_SHARD_DRAFT` | Must be set to `1`. This shards MTP draft KV/metadata for DCP instead of replicating it. |
| Sparse indexer | `VLLM_USE_B12X_SPARSE_INDEXER=1`, using B12X global top-k. |
| MoE | `--moe-backend b12x` with `B12X_MOE_FORCE_A16=1`. |
| CUDA graphs | Enabled, no `--enforce-eager`. |

vLLM prints these variables as "unknown environment variable" because they are
read directly by the GLM/DCP code paths rather than through the central env
registry. That warning is expected for this image.

## Docker Build

Build script:

```text
/root/vllm/blackwell-llm-docker/build-dark-devotion-cu132.sh
```

Build command used:

```bash
cd /root/vllm/blackwell-llm-docker

IMAGE=voipmonitor/vllm:dark-devotion-df8ad3b-b12x5af873a-mtptopkscores-cu132-20260621 \
BUILD_BASE_IMAGE=0 PUSH_BASE_IMAGE=0 \
SYSTEM_BASE_IMAGE=voipmonitor/vllm:glm-kimi-cu132-system-base-20260608 \
BUILD_BASE_IMAGE_TAG=voipmonitor/vllm:glm-kimi-cu132-build-base-20260608 \
FLASHINFER_REF=main \
FLASHINFER_COMMIT=9c5ed7c194e7412780862491742fc655daaad6ac \
B12X_REF=master \
B12X_COMMIT=5af873a7b6c81fbf533ef96bede13fbf4744ad2a \
VLLM_REF=df8ad3b202c84937a23cfa9d93f7a3677da8ecde \
VLLM_COMMIT=df8ad3b202c84937a23cfa9d93f7a3677da8ecde \
LAUNCHER_REF=df8ad3b202c84937a23cfa9d93f7a3677da8ecde \
LAUNCHER_COMMIT=df8ad3b202c84937a23cfa9d93f7a3677da8ecde \
VLLM_PATCH_URL=http://172.17.0.1:8199/0001-dark-devotion-mtp-topk-scores-b12x-dcp.patch \
VLLM_PATCH_SHA256=8231580f5c7cd8a9f17508058a818f4809a2a3dbff49e3b4a1014b4d634df504 \
VLLM_BUILD_VERSION=0.11.2.dev279+dark.devotion.df8ad3b.b12x5af873a.fi9c5ed7c.dcpglobaltopk.mtptopkscores.cu132.20260621 \
./build-dark-devotion-cu132.sh
```

Then tag the production image:

```bash
docker tag \
  voipmonitor/vllm:dark-devotion-df8ad3b-b12x5af873a-mtptopkscores-cu132-20260621 \
  voipmonitor/vllm:glm52-dark-devotion-pr31-dcpglobaltopk-mtp-cu132-20260621
```

## Model

Luke NVFP4 checkpoint:

```text
/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522
```

Required GLM-5.2 index-cache override:

```json
{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}
```

## Production DCP4 MTP3

This is the production profile intended for user testing. It uses TP8/DCP4,
MTP3, `max_num_seqs=32`, and graph capture size `128`.

```yaml
services:
  glm52:
    image: voipmonitor/vllm:glm52-dark-devotion-pr31-dcpglobaltopk-mtp-cu132-20260621
    container_name: glm52-v12-dcp4-mtp3
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
      - /root/.cache/vllm-glm52-v12-dcp4:/cache
    environment:
      CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
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
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_USE_FLASHINFER_SAMPLER: "1"
      VLLM_USE_B12X_MOE: "1"
      VLLM_USE_B12X_FP8_GEMM: "1"
      VLLM_USE_B12X_SPARSE_INDEXER: "1"
      VLLM_ENABLE_PCIE_ALLREDUCE: "1"
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
      VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: 64KB
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
    command:
      - /bin/bash
      - -lc
      - |
        set -euo pipefail
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
        exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve \
          /root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522 \
          --served-model-name GLM-5.2-NVFP4 \
          --trust-remote-code \
          --host 0.0.0.0 \
          --port 5543 \
          --tensor-parallel-size 8 \
          --pipeline-parallel-size 1 \
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
          --hf-overrides '{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}' \
          --quantization modelopt_fp4 \
          --speculative-config '{"model":"/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522","method":"mtp","num_speculative_tokens":3,"moe_backend":"b12x","draft_sample_method":"probabilistic"}'
```

Equivalent `docker run`:

```bash
docker rm -f glm52-v12-dcp4-mtp3 2>/dev/null || true
mkdir -p /root/.cache/vllm-glm52-v12-dcp4

docker run -d \
  --name glm52-v12-dcp4-mtp3 \
  --gpus '"device=0,1,2,3,4,5,6,7"' \
  --ipc=host --shm-size=32g --network=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/.cache/vllm-glm52-v12-dcp4:/cache \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e OMP_NUM_THREADS=16 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_FAST_GPU=1 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_FP8_GEMM=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB \
  -e B12X_DENSE_SPLITK_TURBO=1 \
  -e B12X_W4A16_TC_DECODE=1 \
  -e B12X_MOE_FORCE_A16=1 \
  -e VLLM_DCP_GLOBAL_TOPK=1 \
  -e VLLM_DCP_SHARD_DRAFT=1 \
  -e VLLM_DCP_GLOBAL_TOPK_PREFILL_ONLY=0 \
  -e VLLM_DCP_TOPK_FORCE_DEEPGEMM=0 \
  voipmonitor/vllm:glm52-dark-devotion-pr31-dcpglobaltopk-mtp-cu132-20260621 \
  /bin/bash -lc 'set -euo pipefail
    unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
    exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve \
      /root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522 \
      --served-model-name GLM-5.2-NVFP4 \
      --trust-remote-code --host 0.0.0.0 --port 5543 \
      --tensor-parallel-size 8 --pipeline-parallel-size 1 \
      --max-model-len 256000 \
      --decode-context-parallel-size 4 \
      --dcp-comm-backend ag_rs \
      --dcp-kv-cache-interleave-size 1 \
      --enable-chunked-prefill --enable-prefix-caching \
      --load-format fastsafetensors --async-scheduling \
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
      --hf-overrides '\''{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'\'' \
      --quantization modelopt_fp4 \
      --speculative-config '\''{"model":"/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522","method":"mtp","num_speculative_tokens":3,"moe_backend":"b12x","draft_sample_method":"probabilistic"}'\'''
```

Local validation launch:

| Field | Value |
|---|---|
| Container | `glm52-v12-pr31-dcp4-mtp3-prod-0to7` |
| Port | `5543` |
| GPUs | `0,1,2,3,4,5,6,7` |
| TP / DCP / MTP | `8 / 4 / 3` |
| `max_num_seqs` | `32` |
| `max_cudagraph_capture_size` | `128` |
| KV cache | `1,597,696` tokens |
| Max concurrency for 256k | `6.24x` |

## Debug DCP1 MTP3

Use this when validating functionality quickly. It keeps the same image and
model path but uses DCP1 with a smaller graph capture profile.

```bash
docker rm -f glm52-v12-dcp1-mtp3 2>/dev/null || true
mkdir -p /root/.cache/vllm-glm52-v12-dcp1

docker run -d \
  --name glm52-v12-dcp1-mtp3 \
  --gpus '"device=8,9,10,11,12,13,14,15"' \
  --ipc=host --shm-size=32g --network=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/.cache/vllm-glm52-v12-dcp1:/cache \
  -e CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e OMP_NUM_THREADS=16 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_FAST_GPU=1 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_FP8_GEMM=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB \
  -e B12X_DENSE_SPLITK_TURBO=1 \
  -e B12X_W4A16_TC_DECODE=1 \
  -e B12X_MOE_FORCE_A16=1 \
  -e VLLM_DCP_GLOBAL_TOPK=1 \
  -e VLLM_DCP_SHARD_DRAFT=1 \
  -e VLLM_DCP_GLOBAL_TOPK_PREFILL_ONLY=0 \
  -e VLLM_DCP_TOPK_FORCE_DEEPGEMM=0 \
  voipmonitor/vllm:glm52-dark-devotion-pr31-dcpglobaltopk-mtp-cu132-20260621 \
  /bin/bash -lc 'set -euo pipefail
    unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
    exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve \
      /root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522 \
      --served-model-name GLM-5.2-NVFP4 \
      --trust-remote-code --host 0.0.0.0 --port 5545 \
      --tensor-parallel-size 8 --pipeline-parallel-size 1 \
      --max-model-len 256000 \
      --decode-context-parallel-size 1 \
      --enable-chunked-prefill --enable-prefix-caching \
      --load-format fastsafetensors --async-scheduling \
      -cc.pass_config.fuse_allreduce_rms=True \
      --gpu-memory-utilization 0.88 \
      --max-num-batched-tokens 8192 \
      --max-num-seqs 4 \
      --max-cudagraph-capture-size 24 \
      --attention-backend B12X_MLA_SPARSE \
      --moe-backend b12x \
      --kv-cache-dtype fp8 \
      --tool-call-parser glm47 \
      --enable-auto-tool-choice \
      --reasoning-parser glm45 \
      --hf-overrides '\''{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'\'' \
      --quantization modelopt_fp4 \
      --speculative-config '\''{"model":"/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522","method":"mtp","num_speculative_tokens":3,"moe_backend":"b12x","draft_sample_method":"probabilistic"}'\'''
```

Local validation launch:

| Field | Value |
|---|---|
| Container | `glm52-df8ad3b-mtptopkscores-dcp1-mtp3-8to15` |
| Port | `5545` |
| GPUs | `8,9,10,11,12,13,14,15` |
| TP / DCP / MTP | `8 / 1 / 3` |
| `max_num_seqs` | `4` |
| `max_cudagraph_capture_size` | `24` |
| KV cache | `403,328` tokens |

## Smoke Tests

Health:

```bash
curl -s http://127.0.0.1:5543/v1/models | python3 -m json.tool | head
```

Short decode:

```bash
python3 /mnt/test.py --port 5543 -L
```

Decode benchmark:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 5543 \
  --concurrency 1 \
  --contexts 0k \
  --skip-prefill
```

Runtime logs:

```bash
docker logs -f glm52-v12-dcp4-mtp3
```
