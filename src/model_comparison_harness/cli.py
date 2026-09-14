"""model-comparison-harness -- is a local model good enough to replace Claude here?

Backtests a candidate model against content-discovery-agent's real scoring history
instead of guessing. See seeds/one-harness-two-proving-grounds-when-local-models-are-
actually-good-enough.md in Contexta for the full thesis; this is proving ground #1.
"""

import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Annotated

import typer
from discovery.config import DEFAULT_THRESHOLD, INTEREST_EXCLUSIONS, INTEREST_PROFILE, STORE_PATH
from local_first_common.cli import resolve_provider
from local_first_common.providers import PROVIDERS
from local_first_common.tracking import register_tool, timed_run

from model_comparison_harness.backtest import ItemResult, run_backtest, sample_items
from model_comparison_harness.report import render_markdown, summarize, write_items_csv

_TOOL_NAME = "model-comparison-harness"
_TOOL = register_tool(_TOOL_NAME)
# NOT derived from __file__: that resolves inside the installed uv tool's
# site-packages once `uv tool install` copies the package there, which silently
# buried real results where git (and the repo's own results/ dir) would never
# see them. Fixed 2026-09-13 after exactly that happened to a real run.
_RESULTS_DIR = Path(
    os.environ.get("MODEL_COMPARISON_HARNESS_RESULTS_DIR", "~/projects/local-first/model-comparison-harness/results")
).expanduser()

app = typer.Typer(add_completion=False)


def _default_output_path(provider: str, model: str | None, limit: int) -> Path:
    """Every run is recorded by default -- results/<date>-<provider>-<model>-n<limit>.md
    in the repo, not just printed to stdout, so past runs aren't lost the moment the
    terminal scrolls."""
    model_slug = (model or "default").replace("/", "_").replace(":", "-")
    return _RESULTS_DIR / f"{date.today().isoformat()}-{provider}-{model_slug}-n{limit}.md"


def _print_progress(i: int, total: int, result: ItemResult, cutoff: float, verbose: bool, run_start: float) -> None:
    """Fires after every item -- a 200-item run against a slow/verbose model can take
    over an hour with the previous silent version (confirmed live: gemma4 vs llama3.2:3b,
    same 200 items, one finished in 5 minutes, the other took 68+ with zero visibility
    into how far along it was). Explicitly flushed: stdout is fully buffered, not
    line-buffered, once redirected to a file or pipe -- the common case for a run this
    long -- so an unflushed print wouldn't actually show up until the process exits.
    """
    elapsed = time.monotonic() - run_start
    eta = (elapsed / i) * (total - i)
    progress = f"[{i}/{total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
    if verbose:
        mark = "?" if result.error else ("OK" if result.agrees(cutoff) else "DISAGREE")
        line = f"{progress}  [{mark}] {result.status:>9} claude={result.original_score:.2f} candidate={result.candidate_score} :: {result.title[:60]}"
    else:
        line = progress
    print(line, file=sys.stdout, flush=True)


@app.command()
def backtest(
    provider: Annotated[
        str, typer.Option("--provider", "-p", help="Candidate provider to test against Claude's stored scores")
    ] = "ollama",
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model name (e.g. qwen2.5:7b)")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Items to sample (split evenly kept/dismissed)")] = 200,
    seed: Annotated[int, typer.Option("--seed", help="Sampling seed, for a reproducible sample")] = 42,
    store: Annotated[str | None, typer.Option("--store", help="Path to content-discovery-agent's store.db")] = None,
    threshold: Annotated[
        float | None, typer.Option("--threshold", help="Keep/dismiss cutoff (defaults to the live config's)")
    ] = None,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Write the markdown report to a file")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Print per-item results as they run")] = False,
) -> None:
    """Score a sample of content-discovery-agent's real history with `provider`/`model`,
    compare against the Claude scores already stored, and report where they agree."""
    if provider in ("anthropic",):
        typer.echo(
            "Error: comparing Anthropic against itself proves nothing -- pick a candidate "
            "provider (ollama, groq, deepseek, gemini).",
            err=True,
        )
        raise typer.Exit(1)

    db_path = store or STORE_PATH
    cutoff = threshold if threshold is not None else DEFAULT_THRESHOLD

    try:
        llm_provider = resolve_provider(PROVIDERS, provider, model, fallback=False)
    except Exception as e:  # noqa: BLE001 - top-level CLI boundary: report cleanly and exit, don't show a raw traceback
        typer.echo(f"Error initializing provider '{provider}': {e}", err=True)
        raise typer.Exit(1) from e

    items = sample_items(db_path, limit, seed)
    if not items:
        typer.echo(f"No scored items with a description found in {db_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Sampled {len(items)} items from {db_path}. Scoring with {provider}/{getattr(llm_provider, 'model', model)}...")
    sys.stdout.flush()

    run_start = time.monotonic()

    def _on_result(i: int, total: int, result: ItemResult) -> None:
        _print_progress(i, total, result, cutoff, verbose, run_start)

    with timed_run(_TOOL_NAME, getattr(llm_provider, "model", None)) as run:
        results = run_backtest(items, llm_provider, INTEREST_PROFILE, INTEREST_EXCLUSIONS, on_result=_on_result)
        run.item_count = len(results)

    model_label = f"{provider}/{getattr(llm_provider, 'model', model) or 'default'}"
    summary = summarize(results, model_label, cutoff)
    report = render_markdown(summary)

    typer.echo("\n" + report)

    output_path = Path(output) if output else _default_output_path(provider, model, limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    typer.echo(f"Written: {output_path}")

    items_path = output_path.with_suffix(".csv")
    write_items_csv(results, items_path, cutoff)
    typer.echo(f"Written: {items_path}")


if __name__ == "__main__":
    app()
