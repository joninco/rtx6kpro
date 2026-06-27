# Eldritch Enlightenment vLLM Docker

This page documents the reproducible Docker build used by the GLM-5.2 v13,
DS4 Flash v6, Kimi 2.7, and MiMo validation work.

## Image

```text
voipmonitor/vllm:eldritch-enlightenment-v67e95e7-b12x284a2ea-cu132-20260627
voipmonitor/vllm@sha256:cdc9ee372d97754d624d46e195fafe13cfbd405c9be72a0b455f54f278278777
```

The image is a clean Docker build. It does not require runtime bind-mount
overlays for vLLM or B12X sources.

## Build Source

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 85f3e12
IMAGE=voipmonitor/vllm:eldritch-enlightenment-v67e95e7-b12x284a2ea-cu132-20260627 \
BUILD_BASE_IMAGE=1 \
VLLM_REF=codex/eldritch-enlightenment-release-20260627 \
VLLM_COMMIT=67e95e77da1a45f5d28cedd8958e50284939e03e \
LAUNCHER_REF=codex/eldritch-enlightenment-release-20260627 \
LAUNCHER_COMMIT=67e95e77da1a45f5d28cedd8958e50284939e03e \
VLLM_BUILD_VERSION=0.11.2.dev279+eldritch.enlightenment.67e95e7.b12x284a2ea.fi25dd814.cu132.20260627 \
./build-eldritch-final-cu132.sh
```

The build helper is:

```text
build-eldritch-final-cu132.sh
```

It rebuilds the system/build base images instead of inheriting the older base
with an empty `NCCL_GRAPH_FILE=` environment entry.

## Component Pins

| Component | Revision |
|---|---|
| vLLM repo | `https://github.com/local-inference-lab/vllm.git` |
| vLLM branch | `codex/eldritch-enlightenment-release-20260627` |
| vLLM commit | `67e95e77da1a45f5d28cedd8958e50284939e03e` |
| B12X repo | `https://github.com/voipmonitor/b12x.git` |
| B12X branch | `codex/eldritch-fullstack-20260625` |
| B12X commit | `284a2eae83754ee1abd31c37b9ca66b68e20b8a8` |
| FlashInfer | `25dd814e03791e370f96c3148242f0dc8de504ac` |
| DeepGEMM | `2073ddb2814892014c33ef4cd1c7d4c148baf1fe` |
| CUTLASS | `d80a4e53b52b42550659a8696dab32705265e324` |
| CUDA / cuBLAS | CUDA `13.2.1`, cuBLAS `13.4.1.2-1` |
| cuDNN / NCCL | cuDNN `9.22.0.52-1`, local NCCL `2.30.4` |
| PyTorch | `2.12.0+cu132` |
| FlashInfer cubin wheel | disabled in this build (`FLASHINFER_BUILD_CUBIN=0`) |

## vLLM Patch Stack

The release branch is based on `codex/eldritch-fullstack-20260625` and adds the
validated follow-up fixes:

| Commit | Purpose |
|---|---|
| `1e8d565` | Avoid sparse-indexer host sync on the fast path. |
| `5d35206` | Pass K/V strides and scales to Triton MLA, needed for Kimi-style page layouts. |
| `0ec1381` | Keep DFlash target `lm_head` sharing, needed by MiMo/Kimi DFlash runs. |
| `fcc6141` | Make speculative warmup prompts DCP shard-safe; fixes DCP4 no-MTP and DCP8 MTP3 graph warmup. |
| `dc5bb1` | Include DCP communicator in CUDA graph capture context; fixes Kimi DCP4+DFlash illegal memory access under graph capture. |
| `9c9d23e` | Make DFlash DCP KV metadata prefix-safe. |
| `a81d072` | Preserve DFlash prefix cache under DCP by aligning effective target/draft prefix boundaries. |
| `bfaa36b` | Keep Eagle3 draft DCP opt-in so Kimi DCP4+Eagle3 does not inherit an invalid target DCP layout. |
| `905d6a5` | Shard native MTP draft under DCP by default; fixes GLM/DS native MTP `topk_scores_buffer` startup failures without requiring env overrides. |
| `67e95e7` | Integrate the GLM sparse-indexer fused prefill kernel cherry-picked from upstream vLLM PR #46862. |

The underlying fullstack branch includes the DS4 Flash SM120/CUTLASS runtime
fixes, GLM DCP global top-k and sharded draft KV, structural tool-call fixes,
Kimi parser/spec-decode fixes, MiMo DFlash hooks, and selected upstream vLLM
performance/bugfix PRs.

## Verification

Inside the final image:

```text
torch 2.12.0+cu132
flashinfer 0.6.13+cu132
deep_gemm 2.5.0
vllm 0.11.2.dev279+eldritch.enlightenment.67e95e7.b12x284a2ea.fi25dd814.cu132.20260627
b12x 0.23.0
Rust tool parser present
```

The runtime image config must not contain an empty `NCCL_GRAPH_FILE` or
`NCCL_GRAPH_DUMP_FILE`. Launch scripts should still defensively unset both:

```bash
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
```

Validated GLM-5.2 clean-image smokes:

| Mode | Result |
|---|---|
| TP8/DCP4/MTP-off | Coherent 30k-context Sieve output, `0` CJK, about `62.3 tok/s`. |
| TP8/DCP8/MTP3 | Coherent 30k-context Sieve output, `0` CJK, about `83-88 tok/s`, acceptance around `0.93/0.80/0.65`. |
| Kimi TP8/DCP4/DFlash7 | Coherent short-context Sieve output, `0` CJK, `112.8 tok/s` cc1 decode. |
| Kimi TP8/DCP4/Eagle3 | Coherent short and 30k-context Sieve output, `0` CJK, `115.3 tok/s` cc1 decode. |
