# GLM-5.2 v13 Eldritch NVFP4 / B12X

This page documents the current GLM-5.2 NVFP4 serving stack for RTX 6000 Pro
Blackwell. The default target is Luke's NVFP4 checkpoint, B12X MoE forced to
A16, FP8 KV cache, vLLM V2 model runner, and optional MTP3.

## Image

```text
voipmonitor/vllm:eldritch-final-vfcc6141-b12x284a2ea-cu132-20260626
voipmonitor/vllm@sha256:dd41066fc2bd00fbc9446a78a386a3fe3700d42a4553ddf7a5bcb304ba200f86
```

| Component | Revision |
|---|---|
| vLLM repo | `https://github.com/local-inference-lab/vllm.git` |
| vLLM branch | `codex/eldritch-final-20260626` |
| vLLM commit | `fcc614141e5e9ab18cb304c476f7feed2a9552e3` |
| B12X branch | `codex/eldritch-fullstack-20260625` |
| B12X commit | `284a2eae83754ee1abd31c37b9ca66b68e20b8a8` |
| FlashInfer | `25dd814e03791e370f96c3148242f0dc8de504ac` |
| DeepGEMM | `2073ddb2814892014c33ef4cd1c7d4c148baf1fe` |
| CUDA / cuBLAS | CUDA `13.2.1`, cuBLAS `13.4.1.2-1` |
| cuDNN / NCCL | cuDNN `9.22.0.52-1`, local NCCL `2.30.4` |
| PyTorch | `2.12.0+cu132` |
| Docker build helper | `/root/vllm/blackwell-llm-docker/build-eldritch-final-cu132.sh` |

The image is a clean Docker build, not a runtime overlay.
See [`eldritch-final-docker.md`](./eldritch-final-docker.md) for the exact
reproducible build recipe and component pins.

The final `fcc6141` build includes the DCP shard-safe warmup prompt fix needed
for DCP4 no-MTP and DCP8 MTP3 graph capture. Earlier `v0ec1381` images can
hang during warmup when the synthetic prompt is shorter than the DCP shard
count.

## Model

```text
lukealonso/GLM-5.2-NVFP4
/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522
```

GLM-5.2 still needs the index-cache sparsity pattern override:

```text
FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS
```

## Backend Choice

For `DCP=1`, the fastest attention path is usually
`FLASHINFER_MLA_SPARSE_SM120`. It is the preferred single-DCP path and should
be launched with `-cc.pass_config.fuse_allreduce_rms=True`; without that pass,
the same DCP1/SM120 recipe can drop from about `81-82 tok/s` to about
`75 tok/s` on the coding smoke test.

For `DCP>1`, use `B12X_MLA_SPARSE`. The FlashInfer SM120 sparse MLA path does
not currently provide the DCP decode/LSE behavior we need, so DCP serving should
use B12X attention, B12X sparse indexer, global top-k, and sharded draft KV.

## Runtime Defaults

| Setting | Value |
|---|---|
| TP | `8` |
| DCP | `1`, `2`, `4`, or `8` |
| MoE backend | `b12x` |
| Quantization | `modelopt_fp4` |
| KV cache | `fp8` |
| A16 | `B12X_MOE_FORCE_A16=1` |
| Decode kernel | `B12X_W4A16_TC_DECODE=1` |
| DCP top-k | `VLLM_DCP_GLOBAL_TOPK=1` |
| DCP draft KV | `VLLM_DCP_SHARD_DRAFT=1` |
| MTP default | `3`, probabilistic draft sampling |
| Max batched tokens | `8192` |

Do not set `NCCL_GRAPH_FILE` to an empty string. If no XML topology file is
used, unset it before starting vLLM.

## Docker Compose

This compose file supports both DCP1/SM120 and DCP/B12X. Set `ATTN_BACKEND` to
`FLASHINFER_MLA_SPARSE_SM120` only for `DCP_SIZE=1`; otherwise use
`B12X_MLA_SPARSE`.

```yaml
services:
  glm52:
    image: ${IMAGE:-voipmonitor/vllm:eldritch-final-vfcc6141-b12x284a2ea-cu132-20260626}
    container_name: ${NAME:-glm52-v13}
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
      - ${CACHE_ROOT:-/root/.cache/vllm-glm52-v13}:/cache
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
      VLLM_USE_AOT_COMPILE: "1"
      VLLM_USE_BREAKABLE_CUDAGRAPH: "0"
      VLLM_USE_MEGA_AOT_ARTIFACT: "1"
      VLLM_USE_FLASHINFER_SAMPLER: "1"
      VLLM_USE_B12X_FP8_GEMM: "1"
      VLLM_USE_B12X_MOE: "1"
      VLLM_USE_B12X_SPARSE_INDEXER: "1"
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_ENABLE_PCIE_ALLREDUCE: ${VLLM_ENABLE_PCIE_ALLREDUCE:-1}
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
      VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: 64KB
      B12X_DENSE_SPLITK_TURBO: "1"
      B12X_W4A16_TC_DECODE: "1"
      B12X_MOE_FORCE_A16: "1"
      VLLM_DCP_GLOBAL_TOPK: "1"
      VLLM_DCP_SHARD_DRAFT: "1"
      VLLM_CACHE_DIR: /cache/jit/vllm
      TRITON_CACHE_DIR: /cache/jit/triton
      TORCH_EXTENSIONS_DIR: /cache/jit/torch_extensions
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/torchinductor
      FLASHINFER_WORKSPACE_BASE: /cache/jit/flashinfer
      XDG_CACHE_HOME: /cache/jit
      MODEL: ${MODEL:-/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522}
      SERVED_MODEL_NAME: ${SERVED_MODEL_NAME:-GLM-5.2-NVFP4}
      PORT: ${PORT:-8000}
      TP_SIZE: ${TP_SIZE:-8}
      DCP_SIZE: ${DCP_SIZE:-1}
      MTP_TOKENS: ${MTP_TOKENS:-3}
      ATTN_BACKEND: ${ATTN_BACKEND:-FLASHINFER_MLA_SPARSE_SM120}
      GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION:-0.955}
      MAX_MODEL_LEN: ${MAX_MODEL_LEN:-262144}
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
        SPEC_ARGS=()
        if [ "$${MTP_TOKENS}" != "0" ]; then
          SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$${MTP_TOKENS},\"moe_backend\":\"b12x\",\"draft_sample_method\":\"probabilistic\"}")
        fi
        exec vllm serve "$${MODEL}" \
          --served-model-name "$${SERVED_MODEL_NAME}" \
          --host 0.0.0.0 \
          --port "$${PORT}" \
          --trust-remote-code \
          --tensor-parallel-size "$${TP_SIZE}" \
          --decode-context-parallel-size "$${DCP_SIZE}" \
          --dcp-comm-backend ag_rs \
          --dcp-kv-cache-interleave-size 1 \
          --quantization modelopt_fp4 \
          --kv-cache-dtype fp8 \
          --attention-backend "$${ATTN_BACKEND}" \
          --moe-backend b12x \
          --load-format fastsafetensors \
          -cc.pass_config.fuse_allreduce_rms=True \
          --gpu-memory-utilization "$${GPU_MEMORY_UTILIZATION}" \
          --max-model-len "$${MAX_MODEL_LEN}" \
          --max-num-seqs "$${MAX_NUM_SEQS}" \
          --max-num-batched-tokens "$${MAX_NUM_BATCHED_TOKENS}" \
          --max-cudagraph-capture-size "$${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
          --async-scheduling \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --enable-auto-tool-choice \
          --tool-call-parser glm47 \
          --reasoning-parser glm45 \
          --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
          --hf-overrides "{\"use_index_cache\":true,\"index_topk_pattern\":\"$${GLM52_INDEX_TOPK_PATTERN}\"}" \
          "$${SPEC_ARGS[@]}"
```

## Single Docker Run: DCP1 Fast Path

DCP1 with SM120 attention, B12X MoE, A16 decode, and MTP3:

```bash
docker rm -f glm52-v13 2>/dev/null || true

docker run -d --name glm52-v13 \
  --gpus all --ipc=host --shm-size=32g --network=host --init \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 --ulimit stack=67108864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/.cache/vllm-glm52-v13:/cache \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e NCCL_IB_DISABLE=1 -e NCCL_P2P_LEVEL=SYS -e NCCL_PROTO=LL,LL128,Simple \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_FP8_GEMM=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e B12X_W4A16_TC_DECODE=1 \
  -e B12X_MOE_FORCE_A16=1 \
  voipmonitor/vllm:eldritch-final-vfcc6141-b12x284a2ea-cu132-20260626 \
  /bin/bash -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS; exec vllm serve /root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522 --served-model-name GLM-5.2-NVFP4 --host 0.0.0.0 --port 8000 --trust-remote-code --tensor-parallel-size 8 --decode-context-parallel-size 1 --quantization modelopt_fp4 --kv-cache-dtype fp8 --attention-backend FLASHINFER_MLA_SPARSE_SM120 --moe-backend b12x --load-format fastsafetensors -cc.pass_config.fuse_allreduce_rms=True --gpu-memory-utilization 0.955 --max-model-len 262144 --max-num-seqs 32 --max-num-batched-tokens 8192 --max-cudagraph-capture-size 128 --async-scheduling --enable-chunked-prefill --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 --default-chat-template-kwargs "{\"reasoning_effort\":\"high\"}" --hf-overrides "{\"use_index_cache\":true,\"index_topk_pattern\":\"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS\"}" --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"moe_backend\":\"b12x\",\"draft_sample_method\":\"probabilistic\"}"'
```

For a no-MTP baseline, remove the final `--speculative-config ...` argument.
That baseline is useful because the DCP1/SM120 fast path should be around
`81-82 tok/s` on `/mnt/test.py -L`; with MTP3 and normal acceptance it should
be materially higher.

DCP4 with B12X attention and MTP3: change
`--decode-context-parallel-size 4`, `--attention-backend B12X_MLA_SPARSE`, add
`VLLM_USE_B12X_SPARSE_INDEXER=1`, `VLLM_DCP_GLOBAL_TOPK=1`,
`VLLM_DCP_SHARD_DRAFT=1`, `VLLM_ENABLE_PCIE_ALLREDUCE=1`,
`VLLM_PCIE_ALLREDUCE_BACKEND=b12x`, and add:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"b12x","draft_sample_method":"probabilistic"}'
```

## Validation

Measured on one 8-GPU group with `max_num_seqs=1` / small graph caps for fast
smoke startup. Full production launches should use `max_num_seqs=32` and a
larger graph cap.

| Mode | Attention | MTP | KV cache tokens | Decode cc1 ctx0 tok/s | Prefill 8k tok/s | Prefill 64k tok/s |
|---|---|---:|---:|---:|---:|---:|
| TP8 DCP1 | `FLASHINFER_MLA_SPARSE_SM120` + fused RMS/all-reduce | off | 682,624 | 81.4-82.0 | 2,663 | 4,851 |
| TP8 DCP1 | `FLASHINFER_MLA_SPARSE_SM120` + fused RMS/all-reduce | 3 | not recorded | 128.0 | not rerun | not rerun |
| TP8 DCP4 | `B12X_MLA_SPARSE` | off | 2,704,128 | 62.3 | 1,980 | 3,188 |
| TP8 DCP4 | `B12X_MLA_SPARSE` | 3 | 2,579,456 | 70.6 | not rerun | not rerun |
| TP8 DCP8 | `B12X_MLA_SPARSE` | 3 | 5,143,552 | 83-88 on 30k-context smoke | not rerun | not rerun |

Clean `vfcc6141` smoke status:

- TP8/DCP4/MTP-off, `max_num_seqs=1`, graph cap `4`, long-context `-c 30000`:
  coherent Sieve output across 20+ iterations, `0` CJK characters, about
  `62.3 tok/s` generation-only.
- TP8/DCP8/MTP3, `max_num_seqs=1`, graph cap `4`, long-context `-c 30000`:
  coherent Sieve output across 30+ iterations, `0` CJK characters, about
  `83-88 tok/s` generation-only. Acceptance logs were typically around
  `0.93/0.80/0.65`.

Warm DCP1 MTP3 acceptance examples were around:

```text
0.91 / 0.75 / 0.58
0.94 / 0.81 / 0.66
```

Warm MTP3 acceptance examples for DCP4 were:

```text
0.958 / 0.796 / 0.660
0.900 / 0.751 / 0.570
0.895 / 0.733 / 0.599
```

Useful checks:

```bash
python3 /mnt/test.py --port 8000 -L
python3 /root/llm-inference-bench/llm_decode_bench.py --port 8000 --concurrency 1 --contexts 0k --duration 30 --skip-prefill
python3 /root/llm-inference-bench/llm_decode_bench.py --port 8000 --prefill-only --standalone-prefill --prefill-contexts 8k,64k --prefill-duration 10
```
