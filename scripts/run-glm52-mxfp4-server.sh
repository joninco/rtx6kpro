#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v3f65c52-b12x80eb49b-fi5a73a36-cu132-20260703}
MODEL=${MODEL:-/root/models/GLM-5.2-FP8-MXFP4experts}
DRAFT_MODEL=${DRAFT_MODEL:-/root/.cache/huggingface/hub/models--RedHatAI--GLM-5.2-speculator.dspark/snapshots/7985f0391a3d4f309729eb6f79ea086c812f81fb}

NAME=${NAME:-glm52-mxfp4}
PORT=${PORT:-8000}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
TP=${TP:-8}
DCP=${DCP:-1}
MODE=${MODE:-baseline} # baseline | dspark
DSPARK_TOKENS=${DSPARK_TOKENS:-5}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_BATCHED=${MAX_BATCHED:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
GPU_MEM=${GPU_MEM:-0.90}
GRAPH=${GRAPH:-auto}
ATTN_BACKEND=${ATTN_BACKEND:-B12X_MLA_SPARSE}
SERVED_MODEL=${SERVED_MODEL:-GLM-5.2-FP8-MXFP4-Experts}
SAMPLE=${SAMPLE:-probabilistic}
CACHE=${CACHE:-/root/.cache/vllm-glm52-mxfp4/$NAME}
CONTAINER_TMP=${CONTAINER_TMP:-$CACHE/tmp}

TOPK_PATTERN=${GLM52_INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}

case "$MODE" in
  baseline)
    SPEC_ARGS=()
    if [[ "$GRAPH" == "auto" ]]; then GRAPH="$MAX_NUM_SEQS"; fi
    ;;
  dspark)
    test -f "$DRAFT_MODEL/config.json"
    SPEC_JSON=$(printf '{"model":"%s","method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"%s"}' "$DRAFT_MODEL" "$DSPARK_TOKENS" "$SAMPLE")
    SPEC_ARGS=(--speculative-config "$SPEC_JSON")
    if [[ "$GRAPH" == "auto" ]]; then GRAPH=$(( MAX_NUM_SEQS * (DSPARK_TOKENS + 1) )); fi
    ;;
  *)
    echo "Unknown MODE=$MODE" >&2
    exit 2
    ;;
esac

test -f "$MODEL/config.json"
mkdir -p \
  "$CACHE/vllm" \
  "$CACHE/tilelang/tmp" \
  "$CACHE/tvm" \
  "$CACHE/triton" \
  "$CACHE/torchinductor" \
  "$CACHE/torch_extensions" \
  "$CACHE/flashinfer" \
  "$CONTAINER_TMP"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$NAME" \
  --gpus all \
  --runtime nvidia \
  --ipc host \
  --shm-size 32g \
  --network host \
  --init \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:rw \
  -v /root/models:/root/models:ro \
  -v "$CACHE:/cache:rw" \
  -v "$CONTAINER_TMP:/container-tmp:rw" \
  -e CUDA_VISIBLE_DEVICES="$GPUS" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_FP8_GEMM=1 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB \
  -e B12X_DENSE_SPLITK_TURBO=1 \
  -e B12X_W4A16_TC_DECODE=1 \
  -e B12X_MOE_FORCE_A16=0 \
  -e B12X_MOE_FORCE_A8=1 \
  -e TMPDIR=/container-tmp \
  -e XDG_CACHE_HOME=/cache \
  -e VLLM_CACHE_DIR=/cache/vllm \
  -e TILELANG_CACHE_DIR=/cache/tilelang \
  -e TILELANG_TMP_DIR=/cache/tilelang/tmp \
  -e TVM_CACHE_DIR=/cache/tvm \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
  ${EXTRA_DOCKER_ARGS:-} \
  "$IMAGE" \
  /bin/bash -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS; exec vllm serve "$@"' \
  -- "$MODEL" \
  --served-model-name "$SERVED_MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  --tensor-parallel-size "$TP" \
  --decode-context-parallel-size "$DCP" \
  --kv-cache-dtype fp8 \
  --attention-backend "$ATTN_BACKEND" \
  --moe-backend b12x \
  --load-format fastsafetensors \
  -cc.pass_config.fuse_allreduce_rms=True \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED" \
  --max-cudagraph-capture-size "$GRAPH" \
  --async-scheduling \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
  --hf-overrides "{\"use_index_cache\":true,\"index_topk_pattern\":\"$TOPK_PATTERN\"}" \
  "${SPEC_ARGS[@]}"

echo "$NAME $SERVED_MODEL MODE=$MODE DSpark=$DSPARK_TOKENS TP=$TP DCP=$DCP GPUS=$GPUS PORT=$PORT GRAPH=$GRAPH MAX_NUM_SEQS=$MAX_NUM_SEQS ATTN=$ATTN_BACKEND"
