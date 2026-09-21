# model-comparison-harness

Backtest a candidate model against content-discovery-agent's real Claude-scored history to answer: is local good enough here?

Samples items content-discovery-agent already scored with Claude (split evenly kept/dismissed), re-scores the same items with a candidate provider/model, and reports where the two agree -- so you can swap in a cheaper or local model with evidence instead of a guess.

## Installation
```bash
uv sync
```

## Usage
```bash
uv run model-comparison-harness --provider ollama --model qwen2.5:7b --limit 200
```

Reads content-discovery-agent's `store.db` (path auto-detected, or pass `--store`), samples `--limit` items with `--seed` for reproducibility, and scores them against the same keep/dismiss `--threshold` the live config uses. Pass `--output report.md` to write the comparison as markdown, `--verbose` to see per-item agreement as it runs.

Report output from real runs tends to land in `results/` for reference across comparisons -- generated data, not source, so it's untracked rather than committed.
