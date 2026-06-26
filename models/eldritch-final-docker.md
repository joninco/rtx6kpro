# Eldritch Final vLLM Docker

This page documents the reproducible Docker build used by the GLM-5.2 v13,
DS4 Flash v6, Kimi 2.7, and MiMo validation work.

## Image

```text
voipmonitor/vllm:eldritch-final-vfcc6141-b12x284a2ea-cu132-20260626
voipmonitor/vllm@sha256:dd41066fc2bd00fbc9446a78a386a3fe3700d42a4553ddf7a5bcb304ba200f86
```

The image is a clean Docker build. It does not require runtime bind-mount
overlays for vLLM or B12X sources.

## Build Source

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 85f3e12
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
| vLLM branch | `codex/eldritch-final-20260626` |
| vLLM commit | `fcc614141e5e9ab18cb304c476f7feed2a9552e3` |
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

The final branch is based on `codex/eldritch-fullstack-20260625` and adds four
small follow-up fixes:

| Commit | Purpose |
|---|---|
| `1e8d565` | Avoid sparse-indexer host sync on the fast path. |
| `5d35206` | Pass K/V strides and scales to Triton MLA, needed for Kimi-style page layouts. |
| `0ec1381` | Keep DFlash target `lm_head` sharing, needed by MiMo/Kimi DFlash runs. |
| `fcc6141` | Make speculative warmup prompts DCP shard-safe; fixes DCP4 no-MTP and DCP8 MTP3 graph warmup. |

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
vllm 0.11.2.dev279+eldritch.final.fcc6141.b12x284a2ea.fi25dd814.cu132.20260626
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

