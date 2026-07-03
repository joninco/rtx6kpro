# DeepSeek-V4-Flash and DSpark v9

This page documents the DS4 standard-checkpoint and DSpark v9 validation on the
Eldritch Enlightenment line. The DSpark rows compare the new vLLM+B12X branches
against the v8 DSpark rows. The standard-checkpoint rows add the requested MTP
off, MTP2, and MTP3 comparison.

The tested checkpoints are:

```text
deepseek-ai/DeepSeek-V4-Flash
deepseek-ai/DeepSeek-V4-Flash-DSpark
```

Standard MTP rows use the base checkpoint with `method=mtp` and either `2` or
`3` draft tokens. `standard-mtp0` disables speculative decoding completely.
DSpark uses `method=dspark` with its native block size of `5` draft tokens.

## What Changed From v8

- vLLM starts from the exact v8 image source commit
  `2226f261e9a6befef7a344997fb3b6769baa3bf7` and then merges
  `dev/eldritch-enlightenment` at `0c68fd7637a54efeb6f83df7aeb508811557aeb0`.
- B12X starts from the exact v8 source commit
  `15cd38ce3f10ee5cb7db1179cbc7c88fd15e37b7` and then merges B12X master at
  `80eb49b7683b32a3a1197c03d69142dd9f835cc7`.
- The image defaults now enable OpenAI usage/request metadata:
  `--enable-prompt-tokens-details`, `--enable-force-include-usage`, and
  `--enable-request-id-headers`.
- The image and helper set `VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1`. With this
  attention-aware memory profile, DSpark TP2 needed `gpu_memory_utilization=0.93`
  to keep `max_model_len=262144`; the first `0.92` run was aborted before
  valid results.
- B12X is now split into explicit `A16`, `A8`, and `A8 + DeepGEMM linear`
  rows. The v8 `b12x` row is effectively the `A16` comparison point.
- The launcher now exposes explicit `standard-mtp3` in addition to the existing
  standard MTP2 and no-MTP modes.

## Docker Image

```text
voipmonitor/vllm:eldritch-enlightenment-ds4dspark-v9-ve72ad00-b12x57422ad-cu132-20260703
voipmonitor/vllm@sha256:937db99d072fe089be6d9d88c356e5e2540243389bc7669ecf57a102f637c15e
```

Local image config digest:

```text
sha256:c0a99bff8bb0f5ee02b42a05f354fada9d0d44592a56da5d013171534e5a303a
```

Runtime version:

```text
0.11.2.dev279+eldritch.enlightenment.ds4dspark.v9.e72ad00.b12x57422ad.fi25dd814.cu132.20260703
```

Component pins from image labels:

| Component | Commit / branch |
|---|---|
| vLLM | `codex/eldritch-ds4dspark-v9-20260703` @ `e72ad0057b5f382abb40bec1f9d731f6a22600d9` |
| vLLM upstream merged | `dev/eldritch-enlightenment` @ `0c68fd7637a54efeb6f83df7aeb508811557aeb0` |
| B12X | `codex/ds4dspark-v9-b12x-20260703` @ `57422ad9042773165fa0d4e71ae6442fc26b48c9` |
| B12X upstream merged | `master` @ `80eb49b7683b32a3a1197c03d69142dd9f835cc7` |
| FlashInfer | `25dd814e03791e370f96c3148242f0dc8de504ac` |
| DeepGEMM | `2073ddb2814892014c33ef4cd1c7d4c148baf1fe` (`nv_dev`) |
| NCCL | `2.30.4`, `local-inference-lab/nccl-canonical`, `canonical/cu132-nccl2304-amd-noxml` |
| CUDA / PyTorch | CUDA `13.2.1`, PyTorch `2.12.0+cu132` |

Installed package versions:

```text
vllm 0.11.2.dev279+eldritch.enlightenment.ds4dspark.v9.e72ad00.b12x57422ad.fi25dd814.cu132.20260703
b12x 0.23.0
flashinfer-python 0.6.13+cu132
deep_gemm 2.5.0+2073ddb
torch 2.12.0+cu132
```

## Build Command

The image was built from the existing DSpark TP4 CUDA 13.2 Docker helper:

```bash
cd /root/vllm/blackwell-llm-docker

IMAGE=voipmonitor/vllm:eldritch-enlightenment-ds4dspark-v9-ve72ad00-b12x57422ad-cu132-20260703 \
BUILD_BASE_IMAGE=0 \
PIN_SOURCE_COMMITS=1 \
VLLM_REPO=https://github.com/local-inference-lab/vllm.git \
VLLM_REF=codex/eldritch-ds4dspark-v9-20260703 \
VLLM_COMMIT=e72ad0057b5f382abb40bec1f9d731f6a22600d9 \
LAUNCHER_REPO=https://github.com/local-inference-lab/vllm.git \
LAUNCHER_REF=codex/eldritch-ds4dspark-v9-20260703 \
LAUNCHER_COMMIT=e72ad0057b5f382abb40bec1f9d731f6a22600d9 \
B12X_REPO=https://github.com/voipmonitor/b12x.git \
B12X_REF=codex/ds4dspark-v9-b12x-20260703 \
B12X_COMMIT=57422ad9042773165fa0d4e71ae6442fc26b48c9 \
VLLM_BUILD_VERSION=0.11.2.dev279+eldritch.enlightenment.ds4dspark.v9.e72ad00.b12x57422ad.fi25dd814.cu132.20260703 \
./build-eldritch-enlightenment-dspark-tp4-cu132.sh
```

Build log:

```text
/root/vllm/blackwell-llm-docker/logs/build-eldritch-enlightenment-ds4dspark-v9-ve72ad00-b12x57422ad-20260703T161231Z.log
```

## Models

Standard checkpoint:

```text
/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136
```

DSpark checkpoint:

```text
/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/snapshots/913f0657a874f76844e2e91cbe706dbcaceeb6d7
```

DSpark checkpoint settings:

```text
dspark_block_size=5
dspark_target_layer_ids=[40,41,42]
n_mtp_layers=3
dspark_noise_token_id=128799
dspark_markov_rank=256
```

## Runtime Matrix

| Mode | Checkpoint | Speculative config | Graph cap |
|---|---|---|---:|
| `standard-mtp0` | `DeepSeek-V4-Flash` | none | 256 |
| `standard-mtp2` | `DeepSeek-V4-Flash` | `{"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic"}` plus `moe_backend=b12x` on B12X rows | 512 |
| `standard-mtp3` | `DeepSeek-V4-Flash` | `{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}` plus `moe_backend=b12x` on B12X rows | 512 |
| `dspark` | `DeepSeek-V4-Flash-DSpark` | `{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}` | 512 |

| Backend | Attention | MoE / linear |
|---|---|---|
| `b12x-a16` | `B12X_MLA_SPARSE` | `--moe-backend=b12x --linear-backend=b12x`, `VLLM_USE_B12X_FP8_GEMM=1`, `B12X_MOE_FORCE_A8=0`, `B12X_MOE_FORCE_A16=1` |
| `b12x-a8` | `B12X_MLA_SPARSE` | `--moe-backend=b12x --linear-backend=b12x`, `VLLM_USE_B12X_FP8_GEMM=1`, `B12X_MOE_FORCE_A8=1`, `B12X_MOE_FORCE_A16=0` |
| `b12x-a8-dglin` | `B12X_MLA_SPARSE` | `--moe-backend=b12x`, no `--linear-backend=b12x`, `VLLM_USE_B12X_FP8_GEMM=0`, `B12X_MOE_FORCE_A8=1`, `B12X_MOE_FORCE_A16=0` |
| `lucifer-default` | `FLASHINFER_MLA_SPARSE_DSV4` | default DS4 MoE path, B12X PCIe one-shot all-reduce for small decode tensors |
| `lucifer-cutlass` | `FLASHINFER_MLA_SPARSE_DSV4` | `--kernel-config.moe_backend=flashinfer_cutlass`, B12X PCIe one-shot all-reduce for small decode tensors |

Common B12X env:

```text
VLLM_USE_B12X_WO_PROJECTION=1
VLLM_USE_B12X_MHC=1
VLLM_USE_B12X_MOE=1
VLLM_USE_B12X_SPARSE_INDEXER=1
VLLM_ENABLE_PCIE_ALLREDUCE=1
VLLM_PCIE_ALLREDUCE_BACKEND=b12x
VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB
B12X_MLA_SM120_UNIFIED=1
B12X_MHC_MAX_TOKENS=16384
B12X_DENSE_SPLITK_TURBO=1
B12X_W4A16_TC_DECODE=1
VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1
```

OpenAI server defaults are enabled in the image and explicitly passed by the
helper:

```text
--enable-prompt-tokens-details
--enable-force-include-usage
--enable-request-id-headers
```

The B12X force modes were verified in the server logs:

```text
B12X MoE force-A16 enabled: using quant_mode=w4a16.
B12X MoE force-A8 enabled: using quant_mode=w4a8_mx for E8M0 FP4 weights.
```

## Launch Helper

The recommended v9 launch path is:

```bash
cd /root/rtx6kpro

IMAGE=voipmonitor/vllm:eldritch-enlightenment-ds4dspark-v9-ve72ad00-b12x57422ad-cu132-20260703 \
NAME=ds4-v9 \
PORT=8000 \
GPUS=0,1,2,3 \
TP=4 \
BACKEND=b12x-a8 \
MODE=dspark \
MAX_NUM_SEQS=64 \
scripts/run-ds4-v9-server.sh
```

Examples:

```bash
# Full B12X A16, closest to the v8 B12X row.
TP=4 GPUS=0,1,2,3 BACKEND=b12x-a16 MODE=dspark scripts/run-ds4-v9-server.sh

# Full B12X A8.
TP=4 GPUS=0,1,2,3 BACKEND=b12x-a8 MODE=dspark scripts/run-ds4-v9-server.sh

# B12X attention+MoE A8 with DeepGEMM FP8 linear.
TP=4 GPUS=0,1,2,3 BACKEND=b12x-a8-dglin MODE=dspark scripts/run-ds4-v9-server.sh

# Lucifer CUTLASS reference path.
TP=4 GPUS=0,1,2,3 BACKEND=lucifer-cutlass MODE=dspark scripts/run-ds4-v9-server.sh

# Standard checkpoint with MTP disabled.
TP=4 GPUS=0,1,2,3 BACKEND=lucifer-cutlass MODE=standard-mtp0 scripts/run-ds4-v9-server.sh

# Standard checkpoint with MTP2 / MTP3.
TP=4 GPUS=0,1,2,3 BACKEND=lucifer-cutlass MODE=standard-mtp2 scripts/run-ds4-v9-server.sh
TP=4 GPUS=0,1,2,3 BACKEND=lucifer-cutlass MODE=standard-mtp3 scripts/run-ds4-v9-server.sh
```

The helper uses `gpu_memory_utilization=0.93` for DSpark unless `GPU_MEM` is
overridden. This is required with `VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1` at
`max_model_len=262144`.

## Full Sweep Commands

The v9 benchmark was run with TP-sized waves on the 16-GPU host. DSpark has ten
cases total. The standard checkpoint has thirty cases total: five backends,
three MTP modes, and TP2/TP4.

DSpark sweep:

```bash
cd /root/rtx6kpro

OUT=/root/bench-results/ds4-v9-ve72ad00-gpu093-20260703 \
IMAGE=voipmonitor/vllm:eldritch-enlightenment-ds4dspark-v9-ve72ad00-b12x57422ad-cu132-20260703 \
TPS=2,4 \
BACKENDS=b12x-a16,b12x-a8,b12x-a8-dglin,lucifer-default,lucifer-cutlass \
MODES=dspark \
MAX_NUM_SEQS=64 \
DECODE_CONCURRENCY=1,16,32,64 \
DECODE_CONTEXTS=0 \
DECODE_DURATION=30 \
PREFILL_CONTEXTS=8k,64k,128k \
PREFILL_DURATION=10 \
PORT_BASE=7100 \
PROGRESS_FILE=/root/vllm/prubezne_vysledky \
scripts/run-ds4-v9-sweep.sh
```

Standard checkpoint MTP sweep:

```bash
cd /root/rtx6kpro

OUT=/root/bench-results/ds4-v9-standard-mtp-ve72ad00-20260703 \
IMAGE=voipmonitor/vllm:eldritch-enlightenment-ds4dspark-v9-ve72ad00-b12x57422ad-cu132-20260703 \
TPS=2,4 \
BACKENDS=b12x-a16,b12x-a8,b12x-a8-dglin,lucifer-default,lucifer-cutlass \
MODES=standard-mtp0,standard-mtp2,standard-mtp3 \
MAX_NUM_SEQS=64 \
DECODE_CONCURRENCY=1,16,32,64 \
DECODE_CONTEXTS=0 \
DECODE_DURATION=30 \
PREFILL_CONTEXTS=8k,64k,128k \
PREFILL_DURATION=10 \
PORT_BASE=8100 \
PROGRESS_FILE=/root/vllm/prubezne_vysledky \
scripts/run-ds4-v9-sweep.sh
```

Progress log:

```text
/root/vllm/prubezne_vysledky
```

## Decode Throughput

Sustained decode is aggregate tok/s from `llm_decode_bench.py`, `ctx=0`, 30
seconds per cell. `coding peak` is the median generation-only tok/s over five
Sieve-of-Eratosthenes cc1 runs; every valid row had `0` CJK runs.

### DSpark Checkpoint

| TP | Backend | Mode | cc1 tok/s | cc16 tok/s | cc32 tok/s | cc64 tok/s | coding peak median | CJK runs |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 2 | b12x-a16 | dspark | 201.7 | 1020.3 | 813.4 | 2042.1 | 290.0 | 0 |
| 2 | b12x-a8 | dspark | 202.5 | 994.6 | 1045.3 | 2139.7 | 264.7 | 0 |
| 2 | b12x-a8-dglin | dspark | 194.7 | 1004.5 | 952.7 | 2132.4 | 263.4 | 0 |
| 2 | lucifer-default | dspark | 192.3 | 1030.8 | 1534.1 | 2267.0 | 268.8 | 0 |
| 2 | lucifer-cutlass | dspark | 227.7 | 1147.5 | 1740.3 | 2497.9 | 307.8 | 0 |
| 4 | b12x-a16 | dspark | 263.8 | 1411.6 | 2074.9 | 2522.2 | 362.7 | 0 |
| 4 | b12x-a8 | dspark | 253.0 | 1352.5 | 2098.2 | 2278.8 | 343.1 | 0 |
| 4 | b12x-a8-dglin | dspark | 247.9 | 1342.0 | 2053.7 | 1134.5 | 336.5 | 0 |
| 4 | lucifer-default | dspark | 255.9 | 1490.6 | 2240.5 | 3117.6 | 335.8 | 0 |
| 4 | lucifer-cutlass | dspark | 289.1 | 1680.5 | 2463.1 | 3326.0 | 374.5 | 0 |

### Standard Checkpoint

| TP | Backend | Mode | cc1 tok/s | cc16 tok/s | cc32 tok/s | cc64 tok/s | coding peak median | CJK runs |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 2 | b12x-a16 | standard-mtp0 | 134.2 | 825.9 | 1240.3 | 1862.4 | 133.9 | 0 |
| 2 | b12x-a16 | standard-mtp2 | 218.2 | 1113.9 | 1655.8 | 2501.8 | 228.7 | 0 |
| 2 | b12x-a16 | standard-mtp3 | 203.2 | 1036.9 | 1548.6 | 2267.8 | 220.0 | 0 |
| 2 | b12x-a8 | standard-mtp0 | 133.5 | 751.7 | 1155.3 | 1812.9 | 133.4 | 0 |
| 2 | b12x-a8 | standard-mtp2 | 217.6 | 1063.8 | 1613.8 | 2545.1 | 229.7 | 0 |
| 2 | b12x-a8 | standard-mtp3 | 205.3 | 970.9 | 1516.2 | 2359.3 | 219.4 | 0 |
| 2 | b12x-a8-dglin | standard-mtp0 | 133.0 | 747.4 | 1159.4 | 1811.5 | 132.7 | 0 |
| 2 | b12x-a8-dglin | standard-mtp2 | 213.8 | 1069.8 | 1611.1 | 2538.8 | 225.5 | 0 |
| 2 | b12x-a8-dglin | standard-mtp3 | 208.7 | 1007.0 | 1532.7 | 2383.4 | 224.5 | 0 |
| 2 | lucifer-default | standard-mtp0 | 118.9 | 772.8 | 1147.3 | 1759.9 | 119.9 | 0 |
| 2 | lucifer-default | standard-mtp2 | 192.1 | 1061.6 | 1672.2 | 2578.0 | 211.1 | 0 |
| 2 | lucifer-default | standard-mtp3 | 186.1 | 992.3 | 1567.8 | 2407.7 | 204.1 | 0 |
| 2 | lucifer-cutlass | standard-mtp0 | 127.5 | 866.7 | 1281.1 | 1975.6 | 128.8 | 0 |
| 2 | lucifer-cutlass | standard-mtp2 | 215.0 | 1183.7 | 1844.4 | 2790.4 | 228.2 | 0 |
| 2 | lucifer-cutlass | standard-mtp3 | 203.8 | 1124.8 | 1755.2 | 2647.9 | 227.0 | 0 |
| 4 | b12x-a16 | standard-mtp0 | 162.2 | 1160.3 | 1782.5 | 2645.2 | 161.7 | 0 |
| 4 | b12x-a16 | standard-mtp2 | 284.2 | 1659.2 | 2400.2 | 3498.0 | 301.2 | 0 |
| 4 | b12x-a16 | standard-mtp3 | 259.6 | 1508.0 | 2232.4 | 3112.4 | 280.9 | 0 |
| 4 | b12x-a8 | standard-mtp0 | 161.7 | 1029.8 | 1619.2 | 2478.5 | 161.2 | 0 |
| 4 | b12x-a8 | standard-mtp2 | 270.7 | 1522.1 | 2308.1 | 3543.5 | 295.5 | 0 |
| 4 | b12x-a8 | standard-mtp3 | 260.5 | 1433.9 | 2182.5 | 3291.2 | 297.6 | 0 |
| 4 | b12x-a8-dglin | standard-mtp0 | 160.7 | 1029.5 | 1618.3 | 2478.0 | 161.2 | 0 |
| 4 | b12x-a8-dglin | standard-mtp2 | 276.3 | 1520.1 | 2313.9 | 3553.9 | 294.5 | 0 |
| 4 | b12x-a8-dglin | standard-mtp3 | 263.6 | 1440.4 | 2205.6 | 3312.0 | 291.3 | 0 |
| 4 | lucifer-default | standard-mtp0 | 146.4 | 1113.9 | 1684.2 | 2566.2 | 147.7 | 0 |
| 4 | lucifer-default | standard-mtp2 | 263.0 | 1553.7 | 2427.0 | 3732.7 | 273.3 | 0 |
| 4 | lucifer-default | standard-mtp3 | 253.1 | 1485.9 | 2370.0 | 3535.7 | 279.8 | 0 |
| 4 | lucifer-cutlass | standard-mtp0 | 152.7 | 1227.7 | 1914.4 | 2901.7 | 154.6 | 0 |
| 4 | lucifer-cutlass | standard-mtp2 | 279.4 | 1807.7 | 2792.0 | 4142.3 | 296.3 | 0 |
| 4 | lucifer-cutlass | standard-mtp3 | 267.2 | 1660.6 | 2577.2 | 3799.0 | 294.7 | 0 |

## Prefill Throughput

Client-side prompt tokens / TTFT, `standalone-prefill`, prefix cache enabled but
non-repeating prompts.

### DSpark Checkpoint

| TP | Backend | Mode | 8k tok/s | 64k tok/s | 128k tok/s | Note |
|---:|---|---|---:|---:|---:|---|
| 2 | b12x-a16 | dspark | 11197 | 11185 | 10534 |  |
| 2 | b12x-a8 | dspark | 12883 | 12741 | 11967 |  |
| 2 | b12x-a8-dglin | dspark | 12896 | 12845 | 12029 | DeepGEMM linear |
| 2 | lucifer-default | dspark | 12541 | 12482 | 11460 |  |
| 2 | lucifer-cutlass | dspark | 12443 | 12226 | 11277 |  |
| 4 | b12x-a16 | dspark | 13604 | 13339 | 12562 |  |
| 4 | b12x-a8 | dspark | 14827 | 14552 | 13512 |  |
| 4 | b12x-a8-dglin | dspark | 14750 | 14591 | 13467 | DeepGEMM linear |
| 4 | lucifer-default | dspark | 14777 | 14516 | 13266 |  |
| 4 | lucifer-cutlass | dspark | 14811 | 14399 | 13203 |  |

### Standard Checkpoint

| TP | Backend | Mode | 8k tok/s | 64k tok/s | 128k tok/s | Note |
|---:|---|---|---:|---:|---:|---|
| 2 | b12x-a16 | standard-mtp0 | 11943 | 11491 | 10711 |  |
| 2 | b12x-a16 | standard-mtp2 | 11668 | 11266 | 10476 |  |
| 2 | b12x-a16 | standard-mtp3 | 11589 | 11254 | 10501 |  |
| 2 | b12x-a8 | standard-mtp0 | 13534 | 13084 | 12069 |  |
| 2 | b12x-a8 | standard-mtp2 | 13197 | 12716 | 11781 |  |
| 2 | b12x-a8 | standard-mtp3 | 12941 | 12518 | 11572 |  |
| 2 | b12x-a8-dglin | standard-mtp0 | 13702 | 13025 | 12114 | DeepGEMM linear |
| 2 | b12x-a8-dglin | standard-mtp2 | 13357 | 12902 | 11932 | DeepGEMM linear |
| 2 | b12x-a8-dglin | standard-mtp3 | 13220 | 12756 | 11788 | DeepGEMM linear |
| 2 | lucifer-default | standard-mtp0 | 13521 | 12844 | 11782 |  |
| 2 | lucifer-default | standard-mtp2 | 13086 | 12524 | 11467 |  |
| 2 | lucifer-default | standard-mtp3 | 12883 | 12341 | 11279 |  |
| 2 | lucifer-cutlass | standard-mtp0 | 13026 | 12416 | 11372 |  |
| 2 | lucifer-cutlass | standard-mtp2 | 12632 | 12068 | 11089 |  |
| 2 | lucifer-cutlass | standard-mtp3 | 12674 | 12127 | 11141 |  |
| 4 | b12x-a16 | standard-mtp0 | 14361 | 13729 | 12802 |  |
| 4 | b12x-a16 | standard-mtp2 | 13849 | 13371 | 12421 |  |
| 4 | b12x-a16 | standard-mtp3 | 13688 | 13270 | 12286 |  |
| 4 | b12x-a8 | standard-mtp0 | 15693 | 14986 | 13783 |  |
| 4 | b12x-a8 | standard-mtp2 | 15084 | 14543 | 13398 |  |
| 4 | b12x-a8 | standard-mtp3 | 15131 | 14593 | 13395 |  |
| 4 | b12x-a8-dglin | standard-mtp0 | 15444 | 14885 | 13715 | DeepGEMM linear |
| 4 | b12x-a8-dglin | standard-mtp2 | 15149 | 14608 | 13440 | DeepGEMM linear |
| 4 | b12x-a8-dglin | standard-mtp3 | 15139 | 14618 | 13444 | DeepGEMM linear |
| 4 | lucifer-default | standard-mtp0 | 15501 | 14790 | 13540 |  |
| 4 | lucifer-default | standard-mtp2 | 14889 | 14255 | 12992 |  |
| 4 | lucifer-default | standard-mtp3 | 14794 | 14327 | 13125 |  |
| 4 | lucifer-cutlass | standard-mtp0 | 15337 | 14591 | 13354 |  |
| 4 | lucifer-cutlass | standard-mtp2 | 14938 | 14257 | 13005 |  |
| 4 | lucifer-cutlass | standard-mtp3 | 14629 | 14028 | 12788 |  |

## Quick Read

- The v8 B12X DSpark comparison row is `A16`. In v9, B12X A16 greatly improves
  high-concurrency DSpark decode and prefill versus v8 while keeping cc1/coding
  roughly in the same range.
- DSpark full B12X A8 improves B12X prefill further. At TP4 it reaches `14827`,
  `14552`, and `13512 tok/s` at 8k/64k/128k, essentially matching or beating
  the Lucifer prefill rows in this sweep.
- On the standard checkpoint, MTP2 is the best decode setting in every measured
  cc64 row. The strongest standard row is `tp4-lucifer-cutlass-standard-mtp2`
  at `4142.3 tok/s` cc64 and `296.3 tok/s` coding median.
- Standard MTP3 does not improve cc64 versus MTP2 in this sweep. It can be close
  on coding median, but sustained decode is lower across all backends at TP2
  and TP4.
- The historical hybrid (`b12x-a8-dglin`) does not beat full B12X A8 overall on
  DSpark. On the standard checkpoint it is close to full B12X A8: TP4 MTP2
  cc64 is `3553.9 tok/s` versus `3543.5 tok/s`, with near-identical prefill.
- DSpark Lucifer CUTLASS remains the strongest DSpark decode row: TP2 cc64
  `2497.9 tok/s`, TP4 cc64 `3326.0 tok/s`, and TP4 coding peak `374.5 tok/s`.
- `VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1` changed the startup memory accounting.
  The valid DSpark v9 sweep is the `gpu093` run; ignore the earlier partial
  `/root/bench-results/ds4-v9-ve72ad00-20260703` artefacts with `NA` rows.

## Artifacts

```text
/root/bench-results/ds4-v9-ve72ad00-gpu093-20260703/
/root/bench-results/ds4-v9-standard-mtp-ve72ad00-20260703/
/root/vllm/prubezne_vysledky
/root/rtx6kpro/scripts/run-ds4-v9-server.sh
/root/rtx6kpro/scripts/run-ds4-v9-sweep.sh
```

Source worktrees used for the image:

```text
/root/vllm/worktrees/vllm-ds4dspark-v9-20260703
/root/vllm/worktrees/b12x-ds4dspark-v9-20260703
```

## Caveats

- Standard rows use the base `DeepSeek-V4-Flash` checkpoint. DSpark rows use
  the DSpark checkpoint, not the standard checkpoint with an extra flag.
- `standard-mtp0` disables speculative decoding completely. `standard-mtp2` and
  `standard-mtp3` use the base checkpoint MTP heads with `2` and `3` draft
  tokens.
- DSpark's native tested draft count is `5`; do not treat it as equivalent to
  standard MTP2 or MTP3.
- The helper scripts assume the model snapshot already exists under
  `/root/.cache/huggingface/hub`. Override `STANDARD_MODEL` or `DSPARK_MODEL`
  if your path differs.
