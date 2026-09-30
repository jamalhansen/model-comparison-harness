# model-comparison-harness

Backtest a candidate model against ground truth you already have, to answer: is a cheaper or local model good enough here? Two proving grounds:

- `backtest` -- content-discovery-agent's scoring history, graded against your real keep/dismiss decisions.
- `tags` -- obsidian-vault-auto-tagger, graded against tags you already put on your notes.

Samples items content-discovery-agent already scored with Claude (split evenly kept/dismissed), re-scores the same items with a candidate provider/model, and reports where the two agree -- so you can swap in a cheaper or local model with evidence instead of a guess.

## Installation
```bash
uv sync
```

## Usage
```bash
uv run model-comparison-harness backtest --provider ollama --model qwen2.5:7b --limit 200
uv run model-comparison-harness tags --provider claude-code --model haiku --limit 50
```

`--provider claude-code` runs Claude on your subscription (see local-first-common's README) -- useful both as a candidate ("can this tool move off the metered API?") and as a cheap reference point for local models.

### `tags`

Holds out `--limit` notes that already carry `--min-tags` or more tags (default vault `~/vaults/BrainSync`; `archive/`, `templates/`, daily-note dirs and the `daily` tag are skipped), strips their tags, and re-tags them with the tagger's own prompt and the vault's full tag vocabulary. Reports micro precision/recall/F1 against your tags, how many notes got at least one match, and vocabulary adherence (suggested tags that already exist -- the tagger's rule #1, and the one that keeps a vault from fragmenting). Caveat: `vault-auto-tagger --apply` doesn't mark what it wrote, so some ground-truth tags may be earlier tagger suggestions you accepted.

### `backtest`

Reads content-discovery-agent's `store.db` (path auto-detected, or pass `--store`), samples `--limit` items with `--seed` for reproducibility, and scores them against the same keep/dismiss `--threshold` the live config uses. Pass `--output report.md` to write the comparison as markdown, `--verbose` to see per-item agreement as it runs.

Report output from real runs tends to land in `results/` for reference across comparisons -- generated data, not source, so it's untracked rather than committed.
