"""Aggregate ItemResults into the comparison report the whole harness exists to produce."""

import csv
from dataclasses import dataclass
from pathlib import Path

from model_comparison_harness.backtest import ItemResult


@dataclass
class Summary:
    model_name: str
    n_items: int
    n_errors: int
    n_scored: int
    decision_agreement_rate: float | None
    mean_abs_score_diff: float | None
    avg_latency_s: float | None
    false_dismiss_rate: float | None  # kept by Claude, would be dismissed by candidate -- the costly error
    false_keep_rate: float | None  # dismissed by Claude, would be kept by candidate


def summarize(results: list[ItemResult], model_name: str, threshold: float) -> Summary:
    n_items = len(results)
    scored = [r for r in results if r.candidate_score is not None]
    n_errors = n_items - len(scored)

    if not scored:
        return Summary(model_name, n_items, n_errors, 0, None, None, None, None, None)

    agreements = [r.agrees(threshold) for r in scored]
    agreement_rate = sum(agreements) / len(agreements)

    diffs = [abs(r.candidate_score - r.original_score) for r in scored]
    mean_abs_diff = sum(diffs) / len(diffs)

    avg_latency = sum(r.latency_s for r in scored) / len(scored)

    kept = [r for r in scored if r.original_decision]
    false_dismiss = [r for r in kept if r.candidate_decision(threshold) is False]
    false_dismiss_rate = len(false_dismiss) / len(kept) if kept else None

    dismissed = [r for r in scored if not r.original_decision]
    false_keep = [r for r in dismissed if r.candidate_decision(threshold) is True]
    false_keep_rate = len(false_keep) / len(dismissed) if dismissed else None

    return Summary(
        model_name=model_name,
        n_items=n_items,
        n_errors=n_errors,
        n_scored=len(scored),
        decision_agreement_rate=agreement_rate,
        mean_abs_score_diff=mean_abs_diff,
        avg_latency_s=avg_latency,
        false_dismiss_rate=false_dismiss_rate,
        false_keep_rate=false_keep_rate,
    )


def render_markdown(summary: Summary) -> str:
    def pct(v: float | None) -> str:
        return f"{v:.0%}" if v is not None else "n/a"

    def fnum(v: float | None, digits: int = 2) -> str:
        return f"{v:.{digits}f}" if v is not None else "n/a"

    lines = [
        f"# Model comparison: {summary.model_name}",
        "",
        f"- Items sampled: {summary.n_items} ({summary.n_scored} scored, {summary.n_errors} errors/unparseable)",
        f"- Decision agreement with Claude's original keep/dismiss: **{pct(summary.decision_agreement_rate)}**",
        f"- Mean absolute score difference: {fnum(summary.mean_abs_score_diff)}",
        f"- Avg latency per item: {fnum(summary.avg_latency_s)}s",
        "",
        "## The two error types that actually matter",
        "",
        f"- **False dismiss** (Claude kept it, candidate would drop it — a real item silently lost): {pct(summary.false_dismiss_rate)}",
        f"- **False keep** (Claude dismissed it, candidate would surface it — more noise to review): {pct(summary.false_keep_rate)}",
        "",
        "False dismiss is the expensive error: it's not a false alarm, it's an item that never gets a second chance.",
    ]
    return "\n".join(lines) + "\n"


def write_items_csv(results: list[ItemResult], path: Path, threshold: float) -> None:
    """Per-item detail alongside the aggregate report.

    Written 2026-09-14 after a real gap: a completed gemma4 run only had its
    aggregate summary saved, and "what did gemma4 and Claude actually disagree
    on" couldn't be answered without a full 200-item, ~2-hour re-run. The
    aggregate numbers alone are not the whole record -- this is.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "item_id",
                "status",
                "claude_score",
                "candidate_score",
                "agrees",
                "abs_diff",
                "latency_s",
                "error",
                "title",
                "url",
            ]
        )
        for r in results:
            agrees = r.agrees(threshold)
            abs_diff = abs(r.candidate_score - r.original_score) if r.candidate_score is not None else ""
            writer.writerow(
                [
                    r.item_id,
                    r.status,
                    r.original_score,
                    r.candidate_score if r.candidate_score is not None else "",
                    "" if agrees is None else agrees,
                    abs_diff,
                    f"{r.latency_s:.2f}",
                    r.error or "",
                    r.title,
                    r.url,
                ]
            )
