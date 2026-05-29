# Kimi-K2.6 v7 on 8x RTX PRO 6000 Blackwell

Status: measured 2026-05-29. This page is the Kimi-K2.6 v7 runbook and speed
sweep for the CUDA 13.2 GLM/Kimi vLLM stack with B12X PR8 small-M overlay.

The runtime recipe below intentionally mirrors the Kimi v5 standard+greedy
runbook: vLLM V2 model runner, TRITON_MLA, fp8 target KV, fp8 Eagle3 draft KV,
standard+greedy verifier-backed MTP, FlashInfer autotune, and PCIe custom
allreduce disabled.

## Image

```bash
voipmonitor/vllm:cu132-vllm2f5db31f9bcd-b12xfbb76ca3a914
```

Image digest:

```text
sha256:1eab72d9c83a9f0a82420c8be2bab5b0266c502fb7a2400a3941382c59b34e66
```

Image metadata:

| Component | Revision |
|---|---|
| CUDA base | `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` |
| CUDA | `13.2.1` |
| CUDA compat | `595.71.05-1ubuntu1` |
| cuBLAS | `13.4.1.2-1` |
| cuDNN overlay | `9.22.0.52-1` |
| PyTorch | `2.12.0+cu132` |
| NCCL | `local-inference-lab/nccl-canonical`, branch `canonical/cu132-nccl2304-amd-noxml`, version `2.30.4` |
| FlashInfer | `flashinfer-ai/flashinfer`, branch `main`, commit `8eb61546e82169759801c7895537f3c09ec423f9` |
| vLLM repo | `https://github.com/voipmonitor/vllm.git` |
| vLLM branch | `codex/glm51-v6-awq-mxfp8-clean-rebase-20260528` |
| vLLM commit | `2f5db31f9bcddf8d0cdd4d52f012759f50f37875` |
| B12X branch | `codex/glm51-v6-awq-mxfp8-pr8-smallm-20260528` |
| B12X commit | `fbb76ca3a91491c8f26a2edf729540414323e55b` |
| B12X overlay | `pr8-smallm-direct` |
| Docker recipe repo | `https://github.com/local-inference-lab/blackwell-llm-docker.git` |
| Docker recipe commit | `9f7974843753784715ca46746254722ec28df16b` |

Build command used for this image family:

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 9f7974843753784715ca46746254722ec28df16b

IMAGE=voipmonitor/vllm:cu132-vllm2f5db31f9bcd-b12xfbb76ca3a914 \
SYSTEM_BASE_IMAGE=voipmonitor/vllm:glm-kimi-cu132-system-base-20260528 \
BUILD_BASE_IMAGE_TAG=voipmonitor/vllm:glm-kimi-cu132-build-base-20260528 \
BUILD_BASE_IMAGE=1 \
MAX_JOBS=128 VLLM_MAX_JOBS=128 NVCC_THREADS=1 VLLM_NVCC_THREADS=1 \
VLLM_REPO=https://github.com/voipmonitor/vllm.git \
VLLM_REF=codex/glm51-v6-awq-mxfp8-clean-rebase-20260528 \
VLLM_COMMIT=2f5db31f9bcddf8d0cdd4d52f012759f50f37875 \
LAUNCHER_REPO=https://github.com/voipmonitor/vllm.git \
LAUNCHER_REF=codex/glm51-v6-awq-mxfp8-clean-rebase-20260528 \
LAUNCHER_COMMIT=2f5db31f9bcddf8d0cdd4d52f012759f50f37875 \
FLASHINFER_REF=main \
FLASHINFER_COMMIT=8eb61546e82169759801c7895537f3c09ec423f9 \
B12X_REF=codex/glm51-v6-awq-mxfp8-pr8-smallm-20260528 \
B12X_COMMIT=fbb76ca3a91491c8f26a2edf729540414323e55b \
VLLM_BUILD_VERSION=0.20.2+glm51v6.cu132.20260528 \
./build-glm-kimi-cu132.sh
```

Verify after build:

```bash
docker run --rm "$IMAGE" /usr/local/bin/verify-glm-kimi-cu132
docker run --rm "$IMAGE" python -c "import importlib.metadata as md; print(md.version('b12x')); print(md.version('vllm'))"
docker image inspect "$IMAGE" --format '{{json .Config.Labels}}' | python3 -m json.tool
```

## What Changed Versus v5

| Area | v5 | v7 |
|---|---|---|
| image tag | `voipmonitor/vllm:kimi-v5-cu132-89da7631` | neutral CUDA/source tag `voipmonitor/vllm:cu132-vllm2f5db31f9bcd-b12xfbb76ca3a914` |
| vLLM branch | `codex/glm-kimi-upstream-rebase-20260519` | `codex/glm51-v6-awq-mxfp8-clean-rebase-20260528` |
| vLLM commit | `89da7631ebb844d39dcd5abe5265bc20983be69f` | `2f5db31f9bcddf8d0cdd4d52f012759f50f37875` |
| B12X branch | `codex/glm51-kimi-b12x-a16-cpuhangfix-cutedsl45-20260512` | `codex/glm51-v6-awq-mxfp8-pr8-smallm-20260528` |
| B12X commit | `c929144c7689668b07ca65af10ceadf1c745165d` | `fbb76ca3a91491c8f26a2edf729540414323e55b` |
| CUDA libraries | cuBLAS/cuDNN from v5 CUDA 13.2 base | cuBLAS `13.4.1.2-1`, cuDNN `9.22.0.52-1` overlay |
| FlashInfer | `9035311e975a6aeb2d229f5162e999dfb7c9a733` | `8eb61546e82169759801c7895537f3c09ec423f9` |
| B12X optimization | older B12X | PR8 small-M direct overlay present |
| Kimi runtime policy | V2, TRITON_MLA, fp8 KV, AR off, standard+greedy MTP | same policy |

## MTP Policy

The measured v7 speculative path is verifier-backed and lossless in vLLM. This
page uses the same fastest Kimi v5 MTP profile:

```json
{
  "model": "festr2/kimi-k2.6-eagle3-mla-fp8",
  "method": "eagle3",
  "num_speculative_tokens": 3,
  "draft_attention_backend": "TRITON_MLA",
  "draft_kv_cache_dtype": "fp8",
  "rejection_sample_method": "standard",
  "draft_sample_method": "greedy"
}
```

For `MTP=0`, no speculative config is passed. For `MTP=1`, pass the JSON above
directly to `--speculative-config`.

## Allreduce Policy

Current v7 default: disable PCIe custom allreduce, same as v5.

```bash
VLLM_ENABLE_PCIE_ALLREDUCE=0
VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0
VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER=0
```

Expected log check:

```text
Custom allreduce is disabled for >2 PCIe-only GPUs
Using ['PYNCCL'] all-reduce backends
```

## Runtime Profiles

Use these values unless doing a strict A/B:

| Profile | `DCP` | `MTP` | `GPU_MEM` | Notes |
|---|---:|---:|---:|---|
| DCP1 + MTP | 1 | 1 | 0.90 | v5-compatible standard+greedy speculative profile |
| DCP1 no-MTP | 1 | 0 | 0.94 | target-only baseline, larger KV |
| DCP2 + MTP | 2 | 1 | 0.90 | v5-compatible standard+greedy speculative profile |
| DCP2 no-MTP | 2 | 0 | 0.94 | target-only baseline, larger KV |
| DCP4 + MTP | 4 | 1 | 0.90 | v5-compatible standard+greedy speculative profile |
| DCP4 no-MTP | 4 | 0 | 0.94 | target-only baseline, larger KV |
| DCP8 + MTP | 8 | 1 | 0.89 | v7 safe profile; `0.90` starts but OOMs in `fused_marlin_moe` |
| DCP8 no-MTP | 8 | 0 | 0.94 | target-only baseline, larger KV |

Important KV sizing policy:

- Do not pass a manual `--kv-budget` to the benchmark.
- Let `llm_decode_bench.py` read vLLM KV capacity from `/metrics`.
- Keep `--max-model-len 262144`, `--max-num-seqs 128`, and
  `--max-num-batched-tokens 8192`.
- `DCP8/MTP=1` at `GPU_MEM=0.90` produced `2,741,376` KV tokens but OOMed during
  first inference. The safe rerun uses `GPU_MEM=0.89`, producing `2,338,048`
  KV tokens and completing the measured cells.

## Docker Run

Create the launcher:

```bash
cat >/tmp/run-kimi-k26-v7 <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:cu132-vllm2f5db31f9bcd-b12xfbb76ca3a914}"
NAME="${NAME:-kimi-k26-v7}"
PORT="${PORT:-8402}"
DCP="${DCP:-8}"
MTP="${MTP:-1}"
GPU_MEM="${GPU_MEM:-0.89}"
CACHE_ROOT="${CACHE_ROOT:-${HOME}/.cache/vllm-kimi-k26-v7}"
MODEL_PATH="${MODEL_PATH:-${HOME}/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/b5aabbfb20227ed42becbf5541dbffd213942c58}"

SPEC_CONFIG='{"model":"festr2/kimi-k2.6-eagle3-mla-fp8","method":"eagle3","num_speculative_tokens":3,"draft_attention_backend":"TRITON_MLA","draft_kv_cache_dtype":"fp8","rejection_sample_method":"standard","draft_sample_method":"greedy"}'

mkdir -p \
  "${CACHE_ROOT}/cutlass_dsl" \
  "${CACHE_ROOT}/jit" \
  "${CACHE_ROOT}/triton" \
  "${CACHE_ROOT}/torchinductor" \
  "${CACHE_ROOT}/vllm"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

mtp_disable=0
if [[ "${MTP}" == "0" ]]; then
  mtp_disable=1
fi

exec docker run -d --gpus '"device=0,1,2,3,4,5,6,7"' \
  --ipc=host --network host --privileged \
  --name "${NAME}" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${CACHE_ROOT}/cutlass_dsl:/root/.cache/cutlass_dsl" \
  -v "${CACHE_ROOT}/jit:/cache/jit" \
  -v "${CACHE_ROOT}/triton:/root/.cache/triton" \
  -v "${CACHE_ROOT}/torchinductor:/root/.cache/torchinductor" \
  -v "${CACHE_ROOT}/vllm:/root/.cache/vllm" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e USE_NCCL_XML=0 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0 \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER=0 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e PORT="${PORT}" \
  -e MODEL="${MODEL_PATH}" \
  -e SERVED_MODEL_NAME=Kimi-K2.6 \
  -e TP_SIZE=8 \
  -e DCP_SIZE="${DCP}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEM}" \
  -e MAX_MODEL_LEN=262144 \
  -e MAX_NUM_BATCHED_TOKENS=8192 \
  -e MAX_NUM_SEQS=128 \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=512 \
  -e ATTENTION_BACKEND=TRITON_MLA \
  -e KV_CACHE_DTYPE=fp8 \
  -e LOAD_FORMAT=fastsafetensors \
  -e ENABLE_PREFIX_CACHING=1 \
  -e ENABLE_CHUNKED_PREFILL=1 \
  -e ENABLE_ASYNC_SCHEDULING=1 \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc "$(cat <<RUN_EOF
set -euo pipefail
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
spec_args=()
if [[ "${mtp_disable}" != "1" ]]; then
  spec_args+=(--speculative-config '${SPEC_CONFIG}')
fi
exec /opt/venv/bin/vllm serve '${MODEL_PATH}' \
  --served-model-name Kimi-K2.6 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port '${PORT}' \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --decode-context-parallel-size '${DCP}' \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --async-scheduling \
  --gpu-memory-utilization '${GPU_MEM}' \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode weights \
  --attention-backend TRITON_MLA \
  --kv-cache-dtype fp8 \
  --enable-flashinfer-autotune \
  --max-cudagraph-capture-size 512 \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --enable-auto-tool-choice \
  "\${spec_args[@]}"
RUN_EOF
)"
EOF
chmod +x /tmp/run-kimi-k26-v7
```

Examples:

```bash
# DCP8, standard+greedy MTP on, AR off. Use 0.89 for v7 DCP8/MTP headroom.
PORT=8402 DCP=8 MTP=1 GPU_MEM=0.89 /tmp/run-kimi-k26-v7

# DCP4 target-only baseline.
PORT=8402 DCP=4 MTP=0 GPU_MEM=0.94 /tmp/run-kimi-k26-v7

# DCP1 MTP on, v5-compatible memory profile.
PORT=8402 DCP=1 MTP=1 GPU_MEM=0.90 /tmp/run-kimi-k26-v7
```

Readiness:

```bash
curl -fsS http://127.0.0.1:8402/v1/models | jq .
docker logs kimi-k26-v7 2>&1 | grep -E 'Application startup complete|GPU KV cache size|speculative_config|Custom allreduce'
```

Expected checks:

```text
vLLM version includes 0.20.2+glm51v6.cu132.20260528
decode_context_parallel_size=<DCP>
speculative_config includes festr2/kimi-k2.6-eagle3-mla-fp8 when MTP=1
Custom allreduce is disabled and PYNCCL is selected
structured_outputs_config includes reasoning_parser='kimi_k2'
tool_call_parser is kimi_k2 and auto tool choice is enabled
Application startup complete.
```

Reasoning and tool-call smoke test:

```bash
curl -fsS http://127.0.0.1:8402/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Kimi-K2.6",
    "messages": [
      {
        "role": "user",
        "content": "Use the weather tool to get the current weather in Prague. Do not answer in text; call the tool."
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a city.",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"},
              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 256
  }' | jq '.choices[0].finish_reason, .choices[0].message.reasoning, .choices[0].message.tool_calls'
```

Expected result: `finish_reason` is `tool_calls`, `message.reasoning` is
populated, `message.content` is `null`, and the tool call arguments are clean
JSON, for example `{"city":"Prague"}`.

## Docker Compose

The same profile can be launched with Docker Compose. Runtime knobs are still
environment variables so the same compose file can run DCP1/2/4/8 and MTP on/off.

```bash
cat >/tmp/kimi-k26-v7.compose.yaml <<'EOF'
services:
  kimi-k26-v7:
    image: ${IMAGE:-voipmonitor/vllm:cu132-vllm2f5db31f9bcd-b12xfbb76ca3a914}
    container_name: ${NAME:-kimi-k26-v7}
    network_mode: host
    ipc: host
    privileged: true
    gpus: all
    entrypoint: ["/bin/bash", "-lc"]
    command: |
      set -euo pipefail
      unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
      spec_args=()
      if [[ "$${KIMI_DISABLE_MTP:-0}" != "1" ]]; then
        spec_args+=(--speculative-config "$${KIMI_SPEC_CONFIG}")
      fi
      exec /opt/venv/bin/vllm serve "$${MODEL}" \
        --served-model-name Kimi-K2.6 \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port "$${PORT}" \
        --tensor-parallel-size 8 \
        --pipeline-parallel-size 1 \
        --decode-context-parallel-size "$${DCP_SIZE}" \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --load-format fastsafetensors \
        --async-scheduling \
        --gpu-memory-utilization "$${GPU_MEMORY_UTILIZATION}" \
        --max-model-len 262144 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 128 \
        --mm-processor-cache-gb 0 \
        --mm-encoder-tp-mode weights \
        --attention-backend TRITON_MLA \
        --kv-cache-dtype fp8 \
        --enable-flashinfer-autotune \
        --max-cudagraph-capture-size 512 \
        --reasoning-parser kimi_k2 \
        --tool-call-parser kimi_k2 \
        --enable-auto-tool-choice \
        "$${spec_args[@]}"
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v7}/cutlass_dsl:/root/.cache/cutlass_dsl
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v7}/jit:/cache/jit
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v7}/triton:/root/.cache/triton
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v7}/torchinductor:/root/.cache/torchinductor
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v7}/vllm:/root/.cache/vllm
    environment:
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUDA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7
      NVIDIA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7
      VLLM_USE_V2_MODEL_RUNNER: "1"
      NCCL_IB_DISABLE: "1"
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      USE_NCCL_XML: "0"
      VLLM_ENABLE_PCIE_ALLREDUCE: "0"
      VLLM_RTX6K_FUSED_ALLREDUCE_ADD: "0"
      VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER: "0"
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: "1"
      VLLM_DISABLED_KERNELS: MarlinFP8ScaledMMLinearKernel
      PORT: ${PORT:-8402}
      MODEL: ${MODEL_PATH:-/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/b5aabbfb20227ed42becbf5541dbffd213942c58}
      SERVED_MODEL_NAME: Kimi-K2.6
      TP_SIZE: "8"
      DCP_SIZE: ${DCP:-8}
      GPU_MEMORY_UTILIZATION: ${GPU_MEM:-0.89}
      MAX_MODEL_LEN: "262144"
      MAX_NUM_BATCHED_TOKENS: "8192"
      MAX_NUM_SEQS: "128"
      MAX_CUDAGRAPH_CAPTURE_SIZE: "512"
      ATTENTION_BACKEND: TRITON_MLA
      KV_CACHE_DTYPE: fp8
      LOAD_FORMAT: fastsafetensors
      ENABLE_PREFIX_CACHING: "1"
      ENABLE_CHUNKED_PREFILL: "1"
      ENABLE_ASYNC_SCHEDULING: "1"
      KIMI_DISABLE_MTP: ${KIMI_DISABLE_MTP:-0}
      KIMI_SPEC_CONFIG: '{"model":"festr2/kimi-k2.6-eagle3-mla-fp8","method":"eagle3","num_speculative_tokens":3,"draft_attention_backend":"TRITON_MLA","draft_kv_cache_dtype":"fp8","rejection_sample_method":"standard","draft_sample_method":"greedy"}'
EOF
```

Run:

```bash
# DCP8 + MTP on. Use 0.89 for v7 DCP8/MTP headroom.
DCP=8 KIMI_DISABLE_MTP=0 GPU_MEM=0.89 docker compose -f /tmp/kimi-k26-v7.compose.yaml up -d

# DCP1 no-MTP
DCP=1 KIMI_DISABLE_MTP=1 GPU_MEM=0.94 docker compose -f /tmp/kimi-k26-v7.compose.yaml up -d
```

Stop:

```bash
docker compose -f /tmp/kimi-k26-v7.compose.yaml down
```

## Benchmark Method

Use `/root/llm-inference-bench/llm_decode_bench.py` v0.4.24 or newer. For vLLM,
let the benchmark infer KV budget from `/metrics`; always pass the correct
`--dcp-size`.

Current limited v7 sweep:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 8402 \
  --model Kimi-K2.6 \
  --concurrency 1,32 \
  --contexts 0,128k \
  --duration 30 \
  --skip-prefill \
  --dcp-size <DCP> \
  --display-mode plain \
  --no-hw-monitor \
  --output /root/bench-results/kimi-v7-standard-greedy-matrix-cu132-vllm2f5db31-b12xfbb76ca-20260529/dcp<DCP>/mtp<MTP>/result.json
```

Do not use the older v7 draft command that passed `--kv-budget` manually. The
measured rows below were collected without manual KV budget; the benchmark read
KV capacity from vLLM metrics.

Full matrix to run later:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 8402 \
  --model Kimi-K2.6 \
  --concurrency 1,2,4,8,16,32,64,128 \
  --contexts 0,16k,32k,64k,128k \
  --duration 30 \
  --skip-prefill \
  --dcp-size <DCP> \
  --display-mode plain \
  --no-hw-monitor \
  --output /root/bench-results/kimi-v7-full-matrix-cu132-vllm2f5db31-b12xfbb76ca-20260529/dcp<DCP>/mtp<MTP>/result.json
```

## Standard+Greedy Limited Sweep Results

Result directories:

```bash
/root/bench-results/kimi-v7-standard-greedy-matrix-cu132-vllm2f5db31-b12xfbb76ca-20260529
/root/bench-results/kimi-v7-standard-greedy-matrix-cu132-vllm2f5db31-b12xfbb76ca-20260529-dcp8mtp1-gpumem089
```

These numbers were collected on helper host `10.229.14.14` on 2026-05-29 with
30s sustained decode cells, AR off, V2 model runner, `ctx0` and `ctx128k`,
`cc1` and `cc32`, and standard+greedy MTP for `MTP=1`. The helper host is a
one-CPU system with different switches, so small transport differences versus
the main host are possible.

Run command used:

```bash
IMAGE=voipmonitor/vllm:cu132-vllm2f5db31f9bcd-b12xfbb76ca3a914 \
DURATION=30 \
DCP_LIST="1 2 4 8" \
MTP_LIST="0 1" \
CACHE_ROOT=/root/.cache/vllm-kimi-k26-v7 \
/root/run_kimi_v7_standard_greedy_matrix_20260529.sh
```

`acc` is the average speculative acceptance rate reported by server metrics for
that measured cell. For `MTP=0`, acceptance is `0.000` by definition. `N/A`
means the cell was skipped because it did not fit in KV cache. For `128k/c32`,
the request would require about 4.26M KV tokens, above the measured capacity of
all profiles in this limited sweep.

### DCP 1

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 491,344 | 99.3 | 0.000 | 987.9 | 0.000 | 51.0 | 0.000 | N/A | N/A | AR off, no MTP |
| 1 | 0.90 | 342,672 | 92.3 | 0.398 | 1106.0 | 0.344 | 43.6 | 0.452 | N/A | N/A | AR off, standard+greedy MTP |

### DCP 2

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 982,688 | 86.8 | 0.000 | 807.4 | 0.000 | 60.2 | 0.000 | N/A | N/A | AR off, no MTP |
| 1 | 0.90 | 685,344 | 67.0 | 0.357 | 672.8 | 0.395 | 42.2 | 0.400 | N/A | N/A | AR off, standard+greedy MTP |

### DCP 4

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 1,965,376 | 83.8 | 0.000 | 736.1 | 0.000 | 60.9 | 0.000 | N/A | N/A | AR off, no MTP |
| 1 | 0.90 | 1,370,688 | 66.4 | 0.318 | 594.4 | 0.373 | 36.0 | 0.367 | N/A | N/A | AR off, standard+greedy MTP |

### DCP 8

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 3,930,752 | 82.1 | 0.000 | 438.5 | 0.000 | 64.6 | 0.000 | N/A | N/A | AR off, no MTP |
| 1 | 0.89 | 2,338,048 | 100.1 | 0.404 | 416.1 | 0.412 | 55.3 | 0.377 | N/A | N/A | AR off, standard+greedy MTP; `GPU_MEM=0.89` safe rerun, `0.90` OOM in `fused_marlin_moe` |

## Comparison Against v5 Standard+Greedy

Primary v5 result directory:

```bash
/root/bench-results/kimi-v5-standard-greedy-matrix-89da7631-20260520
```

Short-context `cc1` comparison:

| DCP | MTP | v5 tok/s | v7 tok/s | delta |
|---:|---:|---:|---:|---:|
| 1 | 0 | 89.7 | 99.3 | +10.7% |
| 1 | 1 | 139.4 | 92.3 | -33.8% |
| 2 | 0 | 79.2 | 86.8 | +9.6% |
| 2 | 1 | 122.4 | 67.0 | -45.3% |
| 4 | 0 | 76.1 | 83.8 | +10.1% |
| 4 | 1 | 105.8 | 66.4 | -37.2% |
| 8 | 0 | 74.8 | 82.1 | +9.8% |
| 8 | 1 | 108.6 | 100.1 | -7.8% |

Short-context `cc32` comparison:

| DCP | MTP | v5 tok/s | v7 tok/s | delta |
|---:|---:|---:|---:|---:|
| 1 | 0 | 999.4 | 987.9 | -1.2% |
| 1 | 1 | 1166.5 | 1106.0 | -5.2% |
| 2 | 0 | 885.4 | 807.4 | -8.8% |
| 2 | 1 | 1050.2 | 672.8 | -35.9% |
| 4 | 0 | 835.4 | 736.1 | -11.9% |
| 4 | 1 | 966.3 | 594.4 | -38.5% |
| 8 | 0 | 688.4 | 438.5 | -36.3% |
| 8 | 1 | 697.8 | 416.1 | -40.4% |

Main observation: no-MTP `cc1` improved versus v5, but MTP throughput regressed
for DCP1/2/4 and `cc32` regressed increasingly with higher DCP. Acceptance is
not zero; the regression is not explained by missing KV capacity or completely
broken draft acceptance.

## Notes And Risks

- v7 measurements use `llm_decode_bench.py` v0.4.24. Earlier draft v7 results
  used an older benchmark and manual `--kv-budget`; do not mix those values with
  this table.
- KV cache values in this page are server-reported via `/metrics`, not guessed.
- `DCP8/MTP=1` at `GPU_MEM=0.90` starts with `2,741,376` KV tokens but OOMs in
  `fused_marlin_moe` on first inference. The publishable v7 row uses
  `GPU_MEM=0.89`.
- The currently observed MTP regression needs separate investigation against
  v5. The MTP config itself matches v5 standard+greedy.
- Keep `NCCL_GRAPH_FILE`, `NCCL_GRAPH_DUMP_FILE`, and
  `VLLM_B12X_MLA_EXTEND_MAX_CHUNKS` unset, not set to empty strings.
