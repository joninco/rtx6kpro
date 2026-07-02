# DSpark v8 Optimization and Acceptance Study

Companion to [`ds4dspark-v8.md`](./ds4dspark-v8.md). It answers two questions
from the v8 sweep:

1. Where is the remaining DSpark headroom, and which knobs are worth testing?
2. Is the measured DSpark acceptance the best possible?

No new hardware runs were needed for this page; it consolidates every
acceptance measurement we have, adds extraction tooling for the artifacts the
v8 sweep already saved, and defines the run plan for the open experiments.

## Short Answers

1. The biggest wins left are workload-side, not kernel-side: pick DSpark vs
   MTP2 by concurrency (crossover table below), sweep `DSPARK_TOKENS=3/4/5`
   at cc16-64, and A/B `draft_sample_method` greedy vs probabilistic. The
   two implementation-level items are the conservative prefix-cache draft
   rehydration and the B12X DSpark high-concurrency stall. For cc1
   specifically, the most promising untested lever is the 64 KB one-shot
   all-reduce cap, which likely excludes DSpark's 6-token decode steps from
   the fast path while both standard modes fit under it (E5).
2. No - and the published numbers cannot be read as an acceptance ceiling.
   The v7 matrix acceptance (`34.7%`, `2.73 tok/step`) is a synthetic-decode
   floor, not a property of the deployment: real coding workloads measure
   `60-78%` draft-token acceptance (`3.98-4.9 tok/step` out of a hard ceiling
   of `6.0`). The v8 page recorded no acceptance at all, but the sweep saved
   `final-metrics.prom` for every cell, so the v8 numbers can be extracted
   today with `scripts/dspark-acceptance-report.py` at zero GPU cost.

## What Bounds DSpark Acceptance

DSpark drafts a block of `num_speculative_tokens=5` tokens per step; the
verify pass accepts a prefix of the block plus one bonus token, so:

```text
tokens/step        = 1 + accepted_tokens / drafts        (max 6.0)
draft-token accept = accepted_tokens / draft_tokens       (max 100%)
```

Checkpoint-fixed (not tunable at runtime):

```text
dspark_block_size=5            # draft head trained for 5; >5 unsupported
dspark_target_layer_ids=[40,41,42]
n_mtp_layers=3
dspark_markov_rank=256
dspark_noise_token_id=128799
```

Runtime-tunable (all already plumbed through `run-ds4-v8-server.sh`):

| Knob | Helper env | v8 value | Untested alternatives |
|---|---|---|---|
| Draft length | `DSPARK_TOKENS` | `5` | `3`, `4` at high concurrency |
| Draft sampling | `SAMPLE` | `probabilistic` | `greedy` |
| Backend | `BACKEND` | all three | - |
| Client sampling | request params | bench defaults | temperature 0 raises acceptance |

## Every Acceptance Measurement We Have

| Source | Workload | Draft-token accept | tokens/step | % of 6.0 ceiling |
|---|---|---:|---:|---:|
| v7 matrix, B12X | `llm_decode_bench` ctx=0 synthetic | 35.45% | 2.77 | 46% |
| v7 matrix, Lucifer default | same | 34.70% | 2.73 | 46% |
| v7 matrix, Lucifer CUTLASS | same | 34.62% | 2.73 | 45% |
| v7 prefix-cache smoke | repeated 37k prompt | 50.1% | 3.51 | 58% |
| ormandj, 2026-06-29 | agentic Rust coding, 171k ctx | ~60% | 3.98 | 66% |
| acronjob, 2026-06-28 | coding tasks, early fork, 300W | ~78% | ~4.9 | 82% |

Reference points: the same real-world run measured standard MTP2 at 75%
acceptance but only 2.49 tok/step - DSpark's per-pass yield is ~1.6x MTP2
even at lower per-token acceptance, which is why DSpark wins cc1 coding.

The v8 sweep did not publish acceptance, but its own throughput ratios show
the same workload dependence. DSpark-vs-`standard-mtp0` speedup on identical
hardware (TP4, Lucifer CUTLASS): `1.88x` on the synthetic decode bench versus
`2.40x` on the coding-peak runs. Acceptance on synthetic decode traffic is a
floor; treat it that way.

## Extracting the v8 Acceptance (Zero GPU Cost)

Every v8 sweep cell saved a Prometheus snapshot. Run:

```bash
python3 scripts/dspark-acceptance-report.py --sweep /root/bench-results/ds4-v8-v2226f26-20260630
python3 scripts/dspark-acceptance-report.py --sweep /root/bench-results/ds4-v8-v2226f26-b12xar-lucifer-20260630
```

The report prints accepted/draft, draft-token acceptance, and tokens/step per
case. When `vllm:spec_decode_num_accepted_tokens_per_pos` counters are
present it also prints per-position acceptance (the house methodology already
used for DFlash, e.g. `0.89/0.75/0.57/0.49/...` in
[`kimi-k27-code_v2.md`](./kimi-k27-code_v2.md)) and an expected-tokens/step
model for every truncated draft length `k`, which is the direct input for the
`DSPARK_TOKENS` decision below.

Caveat for the existing v8 artifacts: `final-metrics.prom` was captured after
both decode and prefill stages, so counters mix the two. The sweep script now
also snapshots `baseline-metrics.prom` (post-startup) and
`decode-metrics.prom` (post-decode); future runs get a clean decode-stage
delta automatically, and the report prefers it when present.

## Where DSpark Wins and Loses (v8 data)

DSpark vs `standard-mtp2`, same backend/TP, from the v8 tables:

| TP | Backend | cc1 | coding peak | cc16 | cc32 | cc64 |
|---:|---|---:|---:|---:|---:|---:|
| 2 | lucifer-cutlass | +3% | +35% | -7% | -8% | -13% |
| 4 | lucifer-cutlass | +3% | +24% | -6% | -12% | -19% |
| 4 | b12x | -4% | +19% | -22% | -26% | -42% |

- DSpark is the right default for cc1 and coding-heavy traffic.
- On the synthetic bench, MTP2 takes over from cc16 up. But that bench also
  understates DSpark acceptance by ~2x versus coding traffic, so the real
  crossover for coding workloads is at higher concurrency than this table
  suggests - measuring it is experiment E1 below.
- B12X DSpark is decode-flat from cc32 to cc64 (TP4: `1354.2 -> 1357.0
  tok/s`), a known v7-era limitation, not an acceptance problem. Do not
  deploy B12X DSpark above cc1 until that path is reworked.

## Run Plan

### E1: draft-length sweep at concurrency (the main open experiment)

Per-position acceptance decays steeply (DFlash reference vectors drop below
0.35 by position 5-6; MTP-style measurements show ~0.31 at position 5).
Positions with low marginal acceptance still cost verify-batch tokens on
every step, which is exactly where the cc32/cc64 regression comes from. The
helper already accepts `DSPARK_TOKENS`, and the sweep launcher inherits it:

```bash
for K in 3 4 5; do
  OUT=/root/bench-results/ds4-v8-dspark-k$K \
  DSPARK_TOKENS=$K \
  TPS=4 MODES=dspark BACKENDS=lucifer-cutlass \
  DECODE_CONCURRENCY=1,16,32,64 \
  scripts/run-ds4-v8-sweep.sh
  python3 scripts/dspark-acceptance-report.py --sweep /root/bench-results/ds4-v8-dspark-k$K
done
```

Decision rule: adopt the largest `k` whose marginal position clears the
verify cost at the target concurrency; expect `k=5` to hold for cc1 and a
smaller `k` to win at cc32+ if per-position rates on this workload look like
the DFlash reference decay.

### E2: greedy vs probabilistic draft sampling

v7/v8 pinned `draft_sample_method=probabilistic`. The branch default is
`greedy` ([`ds4-flash-v2.md`](./ds4-flash-v2.md)), and on Kimi K2.6 the
probabilistic drafter hot path cost ~13% cc1 throughput
([`kimi-k26-v4.md`](./kimi-k26-v4.md)). Untested for DSpark:

```bash
TP=4 GPUS=0,1,2,3 BACKEND=lucifer-cutlass MODE=dspark SAMPLE=greedy \
NAME=ds4-v8-dspark-greedy scripts/run-ds4-v8-server.sh
```

Quality gate required: a `standard` rejection sampler combined with a greedy
drafter has previously collapsed sampling to effective temperature 0 on
GLM/Kimi MTP stacks (2026-05-20). Verify output diversity at temperature
`>0` before adopting, and re-check the CJK-run counter stays at `0`.

### E3: coding-workload acceptance attribution

The matrix acceptance should be published from a workload that resembles
deployment. Cheapest version: snapshot `/metrics` immediately before and
after the five coding-peak runs and diff with the report script:

```bash
curl -fsS localhost:8000/metrics > before.prom
# ... coding-peak runs ...
curl -fsS localhost:8000/metrics > after.prom
python3 scripts/dspark-acceptance-report.py --baseline before.prom after.prom
```

### E4: implementation-level items (tracked, not runnable from this repo)

- Prefix-cache draft rehydration is conservative: after a prefix-cache hit,
  drafts are deferred until DSpark's rolling KV window is rebuilt (v7 page).
  Agentic clients re-send a growing cached prefix every turn, so each turn
  starts with a no-draft window. A zero-wait rehydration would raise
  effective tokens/step for exactly the workload DSpark is best at.
- B12X DSpark cc32+ stall (see table above).
- vLLM v0.24.0 shipped dynamic speculative decoding (2026-06-30); on a
  future rebase it could replace the manual DSpark-vs-MTP2 mode choice by
  scaling speculation with load.

### E5: single-stream (cc1) throughput recipe

cc1 is DSpark's home turf, and its levers are different from the cc16+ case:
per-step latency dominates, and the verify batch (6 tokens) is so small that
extra draft positions are nearly free. Keep `DSPARK_TOKENS=5` at cc1; the
tuning targets are per-step all-reduce latency and drafter overhead.

Baseline profile (best measured cc1 in v8): TP4 Lucifer CUTLASS DSpark -
`287.9 tok/s` synthetic, `370.2 tok/s` coding peak. TP4 is +29% over TP2 for
this mode; use 4 GPUs if they are free.

Ranked cc1 experiments:

1. **One-shot all-reduce size cap (`ONESHOT_MAX`).** The v8 helper pins
   `VLLM_PCIE_ALLREDUCE_MAX_SIZE=64KB` for Lucifer backends. Decode
   all-reduce payload per layer is `tokens_in_step x hidden_size x 2B`
   (bf16). With the DeepSeek V3-lineage hidden size of 7168 (verify against
   the checkpoint `config.json`), a cc1 step carries:

   | Mode | Tokens/step | Payload | vs 64 KiB cap |
   |---|---:|---:|---|
   | `standard-mtp0` | 1 | 14 KiB | one-shot |
   | `standard-mtp2` | 3 | 42 KiB | one-shot |
   | `dspark` | 6 | 84 KiB | **falls back to PyNCCL** |

   If confirmed, DSpark is the only mode whose cc1 decode all-reduces are
   excluded from the one-shot path - on every layer of every step. The raw
   crossover measurements say one-shot wins well past 84 KiB: up to 512 KB
   on the switch topology
   ([`asus-esc8000a-e13p-broadcom-switches.md`](../hardware/asus-esc8000a-e13p-broadcom-switches.md)),
   120 KB auto-tuned on 4 GPUs
   ([`pcie-oneshot-allreduce.md`](../optimization/pcie-oneshot-allreduce.md)).
   The helper now takes the cap as an env:

   ```bash
   # A/B: default 64KB vs 128KB, TP4 Lucifer CUTLASS DSpark
   TP=4 GPUS=0,1,2,3 BACKEND=lucifer-cutlass MODE=dspark \
   ONESHOT_MAX=128KB NAME=ds4-v8-dspark-ar128 scripts/run-ds4-v8-server.sh
   ```

   Verify in the server log which path the decode all-reduce selects at
   each size, and confirm `hidden_size` first. Gates: the 64 KB cap may be
   part of the language-hallucination mitigation (2026-06-30: all-reduce
   payload reduction cut corruption latency 6-7ms to 1.5ms), so the CJK-run
   counter must stay `0`; also re-check 64k/128k prefill for regressions
   since larger one-shot payloads affect mixed batches.
2. **`SAMPLE=greedy` (E2).** Drafter hot-path cost is pure per-step latency,
   which is why the Kimi analog showed its ~13% penalty specifically at c1.
   Same command and quality gate as E2.
3. **Host prerequisites.** One-shot AR only helps if P2P BAR1 mapping is
   active - on direct-attach (NODE) topology the nvidia `ForceP2P` override
   is mandatory or the kernel silently degrades ~15x
   ([`pcie-oneshot-allreduce.md`](../optimization/pcie-oneshot-allreduce.md)).
   Check GPU power limits too: the 241 tok/s community coding number was
   measured at 300 W; the v8 numbers assume full power.
4. **Client-side sampling.** Acceptance (and thus cc1 throughput) rises as
   client temperature drops; temperature 0 is the cc1-throughput-optimal
   request shape for agentic coding.

For agentic cc1 loops specifically, the prefix-cache rehydration item in E4
is also on the critical path: every turn starts with a no-draft window after
the prefix-cache hit, which is pure lost cc1 throughput.

Not worth chasing at cc1: `DSPARK_TOKENS<5` (only helps when verify-batch
cost matters, i.e. cc16+), TP8 (cross-socket system-scope barriers penalize
the one-shot path and GLM TP8+MTP regressed with it), and B12X (cc1 coding
peak 363.7 is close, but it loses everywhere else in the DSpark matrix).

## Verdict on "Best Possible" Acceptance

- Hard ceiling is `6.0 tokens/step` (`dspark_block_size=5` is trained-in;
  values above 5 are unsupported for this checkpoint).
- On coding traffic the stack already reaches `~3.98-4.9 tokens/step`
  (66-82% of ceiling). Remaining acceptance headroom there is modest and
  sits in E2 (draft sampling) and E4 (prefix-cache rehydration windows).
- The widely-quoted `34.7%` figure is not evidence of a poorly tuned
  drafter; it is what unconstrained synthetic decode traffic yields. Do not
  spend tuning effort chasing it.
- What is genuinely not yet optimal is net throughput at cc16+, and the
  lever there is draft length (E1) and mode choice, not acceptance itself.
- Verification is incomplete until the v8 per-cell counters are extracted
  (zero-cost) and E1/E3 are run; this page and the report script exist to
  close that gap.

## Artifacts

```text
/root/bench-results/ds4-v8-v2226f26-20260630/*/final-metrics.prom
/root/bench-results/ds4-v8-v2226f26-b12xar-lucifer-20260630/*/final-metrics.prom
/root/rtx6kpro/scripts/dspark-acceptance-report.py
/root/rtx6kpro/scripts/run-ds4-v8-sweep.sh   # now snapshots baseline/decode metrics
```

Sources: [`ds4dspark-v7.md`](./ds4dspark-v7.md) (acceptance counters,
prefix-cache semantics), [`ds4dspark-v8.md`](./ds4dspark-v8.md) (throughput
matrix), daily summaries 2026-06-28/29/30 (real-world acceptance reports,
TP4 fix, hallucination mitigation, vLLM v0.24 dynamic spec decode),
[`optimization/speculative-decoding.md`](../optimization/speculative-decoding.md)
(per-position decay reference).
