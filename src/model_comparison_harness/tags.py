"""Proving ground #2: obsidian-vault-auto-tagger against tags Jamal already applied.

Hold out a sample of already-tagged notes, strip their tags, and ask a candidate
model to tag them with the tagger's own prompt and the vault's full tag
vocabulary -- the same context the live tool sees. The hidden tags are the
ground truth. Unlike proving ground #1 this needs no stored model history: the
labels are the tags themselves.

Caveat: `vault-auto-tagger --apply` writes tags with no marker, so some ground
truth may be earlier tagger output that Jamal accepted. Still a human-reviewed
label, but not a purely hand-written one.
"""

import csv
import os
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import frontmatter
from local_first_common.models import ContentMetadata
from local_first_common.providers.base import BaseProvider
from obsidian_vault_auto_tagger.core import get_all_vault_tags
from obsidian_vault_auto_tagger.prompts import build_system_prompt, build_user_prompt
from obsidian_vault_auto_tagger.schema import VaultTagReport

DEFAULT_EXCLUDE_DIRS = ("archive", "templates", "Templates", "daily", "Daily", "journal")
DEFAULT_IGNORE_TAGS = ("daily",)


def normalize_tag(tag: str) -> str:
    return tag.strip().lstrip("#").strip().lower()


@dataclass
class Note:
    path: str
    content: str
    tags: list[str]


@dataclass
class TagResult:
    path: str
    truth: list[str]
    predicted: list[str] | None
    in_vocab: int
    latency_s: float
    error: str | None = None

    @property
    def hits(self) -> int:
        return len(set(self.predicted or []) & set(self.truth))


def load_tagged_notes(
    vault: Path,
    min_tags: int = 2,
    exclude_dirs: Iterable[str] = DEFAULT_EXCLUDE_DIRS,
    ignore_tags: Iterable[str] = DEFAULT_IGNORE_TAGS,
) -> list[Note]:
    """Notes carrying at least `min_tags` meaningful tags, sorted for reproducible sampling."""
    excluded = set(exclude_dirs)
    ignored = {normalize_tag(t) for t in ignore_tags}
    notes: list[Note] = []
    for root, dirs, files in os.walk(vault):
        dirs[:] = sorted(d for d in dirs if d not in excluded and not d.startswith("."))
        for name in sorted(files):
            if not name.endswith(".md"):
                continue
            path = Path(root) / name
            try:
                post = frontmatter.load(path)
                tags = ContentMetadata.from_metadata(post.metadata).tags
            except Exception:  # noqa: BLE001, S112 - a malformed note is skipped, same as the tagger's own scan
                continue
            clean = sorted({normalize_tag(t) for t in tags} - ignored - {""})
            if len(clean) >= min_tags and post.content.strip():
                notes.append(Note(str(path.relative_to(vault)), post.content, clean))
    return notes


def sample_notes(notes: list[Note], limit: int, seed: int) -> list[Note]:
    return random.Random(seed).sample(notes, k=min(limit, len(notes)))


def vault_vocabulary(vault: Path) -> list[str]:
    """The tagger's own vocabulary scan, so the candidate sees exactly what the live tool sees."""
    return sorted({normalize_tag(t) for t in get_all_vault_tags(vault)} - {""})


def run_tag_backtest(
    notes: list[Note],
    provider: BaseProvider,
    vocabulary: list[str],
    on_result: Callable[[int, int, TagResult], None] | None = None,
) -> list[TagResult]:
    system = build_system_prompt(vocabulary)
    vocab = set(vocabulary)
    results: list[TagResult] = []
    for i, note in enumerate(notes, start=1):
        # Same 2000-char truncation and one-note-per-call shape as the live tool's batches.
        user = build_user_prompt([{"path": note.path, "content": note.content[:2000], "tags": []}])
        provider.source_location = note.path
        provider.item_count = 1
        start = time.monotonic()
        try:
            report = provider.complete(system, user, response_model=VaultTagReport)
            suggestions = report.suggestions
            match = next((s for s in suggestions if s.file_path == note.path), suggestions[0] if suggestions else None)
            predicted = sorted({normalize_tag(t) for t in match.suggested_tags} - {""}) if match else []
            result = TagResult(note.path, note.tags, predicted, len(set(predicted) & vocab), time.monotonic() - start)
        except Exception as e:  # noqa: BLE001 - one bad response is a data point, not a reason to abort a long run
            result = TagResult(note.path, note.tags, None, 0, time.monotonic() - start, error=str(e)[:200])
        results.append(result)
        if on_result:
            on_result(i, len(notes), result)
    return results


@dataclass
class TagSummary:
    model_name: str
    n_notes: int
    n_errors: int
    precision: float | None
    recall: float | None
    f1: float | None
    notes_with_any_hit: float | None
    vocab_adherence: float | None
    avg_suggested: float | None
    avg_latency_s: float | None


def summarize_tags(results: list[TagResult], model_name: str) -> TagSummary:
    scored = [r for r in results if r.predicted is not None]
    n_errors = len(results) - len(scored)
    if not scored:
        return TagSummary(model_name, len(results), n_errors, None, None, None, None, None, None, None)
    hits = sum(r.hits for r in scored)
    n_pred = sum(len(r.predicted) for r in scored)
    n_truth = sum(len(r.truth) for r in scored)
    precision = hits / n_pred if n_pred else None
    recall = hits / n_truth if n_truth else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return TagSummary(
        model_name=model_name,
        n_notes=len(results),
        n_errors=n_errors,
        precision=precision,
        recall=recall,
        f1=f1,
        notes_with_any_hit=sum(1 for r in scored if r.hits) / len(scored),
        vocab_adherence=sum(r.in_vocab for r in scored) / n_pred if n_pred else None,
        avg_suggested=n_pred / len(scored),
        avg_latency_s=sum(r.latency_s for r in scored) / len(scored),
    )


def render_tags_markdown(s: TagSummary) -> str:
    def pct(v: float | None) -> str:
        return f"{v:.0%}" if v is not None else "n/a"

    def num(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else "n/a"

    return (
        "\n".join(
            [
                f"# Tag backtest: {s.model_name}",
                "",
                f"- Notes sampled: {s.n_notes} ({s.n_notes - s.n_errors} scored, {s.n_errors} errors/unparseable)",
                f"- **F1 vs your own tags: {pct(s.f1)}** (precision {pct(s.precision)}, recall {pct(s.recall)})",
                f"- Notes where at least one suggested tag matched yours: {pct(s.notes_with_any_hit)}",
                f"- Suggested tags already in the vault vocabulary: {pct(s.vocab_adherence)} (the tagger's rule #1)",
                f"- Avg tags suggested per note: {num(s.avg_suggested)}",
                f"- Avg latency per note: {num(s.avg_latency_s)}s",
                "",
                "Precision is the noise cost (tags you'd have to delete); recall is what the model misses.",
                "Vocabulary adherence matters as much as either: a model that invents near-duplicate",
                "tags fragments the vault even when each individual tag looks reasonable.",
            ]
        )
        + "\n"
    )


def write_tags_csv(results: list[TagResult], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "your_tags", "suggested_tags", "hits", "in_vocab", "latency_s", "error"])
        for r in results:
            writer.writerow(
                [
                    r.path,
                    " ".join(r.truth),
                    "" if r.predicted is None else " ".join(r.predicted),
                    r.hits,
                    r.in_vocab,
                    f"{r.latency_s:.2f}",
                    r.error or "",
                ]
            )
