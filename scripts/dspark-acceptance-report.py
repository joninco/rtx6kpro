#!/usr/bin/env python3
"""Report speculative-decoding acceptance from vLLM Prometheus snapshots.

Built for the DS4 DSpark/MTP sweeps (scripts/run-ds4-v8-sweep.sh), but works
with any vLLM v1 server that exports the spec_decode counters:

    vllm:spec_decode_num_drafts_total
    vllm:spec_decode_num_draft_tokens_total
    vllm:spec_decode_num_accepted_tokens_total
    vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} ...

Usage:

    # Single snapshot:
    dspark-acceptance-report.py case-dir/final-metrics.prom

    # Delta between two snapshots (isolate one bench stage):
    dspark-acceptance-report.py --baseline case-dir/baseline-metrics.prom \
        case-dir/decode-metrics.prom

    # Whole sweep output dir (one row per case). Prefers the
    # decode-metrics minus baseline-metrics delta when both exist,
    # otherwise falls back to final-metrics.prom:
    dspark-acceptance-report.py --sweep /root/bench-results/ds4-v8-...

When per-position counters are present, the report also prints the expected
tokens/step for every truncated draft length k, so the marginal value of
draft positions can be judged directly (is num_speculative_tokens=5 worth
it, or does k=3/4 win at high concurrency?). The truncation model assumes
per-position acceptance rates are unchanged by truncation, which holds for
prefix-accept rejection sampling to first order.
"""

import argparse
import re
import sys
from pathlib import Path

# `vllm:` prefix optional; `_total` suffix optional (exposition formats vary).
_METRIC_RE = re.compile(
    r"^(?:vllm:)?spec_decode_num_"
    r"(?P<kind>drafts|draft_tokens|accepted_tokens|accepted_tokens_per_pos)"
    r"(?:_total)?"
    r"(?:\{(?P<labels>[^}]*)\})?"
    r"\s+(?P<value>[0-9.eE+-]+)\s*$"
)
_POSITION_RE = re.compile(r'position="(\d+)"')


def parse_metrics(path):
    """Return {'drafts': x, 'draft_tokens': x, 'accepted_tokens': x,
    'per_pos': {pos: x}} summed across all other label sets."""
    counts = {"drafts": 0.0, "draft_tokens": 0.0, "accepted_tokens": 0.0}
    per_pos = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        value = float(m.group("value"))
        kind = m.group("kind")
        if kind == "accepted_tokens_per_pos":
            pm = _POSITION_RE.search(m.group("labels") or "")
            if pm:
                pos = int(pm.group(1))
                per_pos[pos] = per_pos.get(pos, 0.0) + value
        else:
            counts[kind] += value
    counts["per_pos"] = per_pos
    return counts


def subtract(after, before):
    out = {k: after[k] - before.get(k, 0.0) for k in ("drafts", "draft_tokens", "accepted_tokens")}
    out["per_pos"] = {
        pos: v - before["per_pos"].get(pos, 0.0) for pos, v in after["per_pos"].items()
    }
    return out


def summarize(counts):
    drafts = counts["drafts"]
    draft_tokens = counts["draft_tokens"]
    accepted = counts["accepted_tokens"]
    if drafts <= 0:
        return None
    summary = {
        "drafts": int(drafts),
        "draft_tokens": int(draft_tokens),
        "accepted_tokens": int(accepted),
        "accepted_per_draft": accepted / drafts,
        "draft_token_acceptance": accepted / draft_tokens if draft_tokens else 0.0,
        # +1 for the bonus token the verify pass always emits.
        "tokens_per_step": 1.0 + accepted / drafts,
    }
    if counts["per_pos"]:
        max_pos = max(counts["per_pos"])
        rates = [counts["per_pos"].get(i, 0.0) / drafts for i in range(max_pos + 1)]
        summary["per_pos_rates"] = rates
    return summary


def print_case(label, summary):
    if summary is None:
        print(f"{label}: no spec_decode drafts recorded")
        return
    print(f"{label}:")
    print(
        f"  drafts={summary['drafts']}"
        f" draft_tokens={summary['draft_tokens']}"
        f" accepted_tokens={summary['accepted_tokens']}"
    )
    print(
        f"  accepted/draft={summary['accepted_per_draft']:.2f}"
        f"  draft-token acceptance={summary['draft_token_acceptance'] * 100:.2f}%"
        f"  tokens/step={summary['tokens_per_step']:.2f}"
    )
    rates = summary.get("per_pos_rates")
    if rates:
        print("  per-position acceptance: " + "/".join(f"{r:.2f}" for r in rates))
        print("  truncated draft-length model (assumes rates hold under truncation):")
        print("    k  E[accepted]  tokens/step  marginal")
        cumulative = 0.0
        for k, rate in enumerate(rates, start=1):
            cumulative += rate
            print(f"    {k}  {cumulative:11.2f}  {1.0 + cumulative:11.2f}  {rate:+.2f}")


def report_files(metrics_path, baseline_path, label):
    counts = parse_metrics(metrics_path)
    if baseline_path:
        counts = subtract(counts, parse_metrics(baseline_path))
    print_case(label, summarize(counts))


def report_sweep(sweep_dir):
    sweep = Path(sweep_dir)
    case_dirs = sorted(p.parent for p in sweep.glob("*/final-metrics.prom"))
    if not case_dirs:
        sys.exit(f"No */final-metrics.prom found under {sweep}")
    for case in case_dirs:
        decode = case / "decode-metrics.prom"
        baseline = case / "baseline-metrics.prom"
        if decode.is_file():
            report_files(
                decode,
                baseline if baseline.is_file() else None,
                f"{case.name} (decode stage)",
            )
        else:
            report_files(case / "final-metrics.prom", None, f"{case.name} (final)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="metrics .prom file, or sweep dir with --sweep")
    parser.add_argument("--baseline", help="earlier snapshot to subtract")
    parser.add_argument("--sweep", action="store_true", help="treat path as a sweep output dir")
    args = parser.parse_args()
    if args.sweep:
        if args.baseline:
            parser.error("--baseline is not valid with --sweep")
        report_sweep(args.path)
    else:
        report_files(args.path, args.baseline, args.path)


if __name__ == "__main__":
    main()
