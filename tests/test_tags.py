from pathlib import Path

import pytest

from model_comparison_harness.tags import (
    TagResult,
    load_tagged_notes,
    normalize_tag,
    render_tags_markdown,
    run_tag_backtest,
    sample_notes,
    summarize_tags,
    write_tags_csv,
)


def _note(vault: Path, rel: str, tags: list[str], body: str = "Some body text.") -> None:
    path = vault / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    tag_yaml = "\n".join(f'  - "{t}"' for t in tags)
    path.write_text(f"---\ntags:\n{tag_yaml}\n---\n{body}\n", encoding="utf-8")


@pytest.fixture
def vault(tmp_path):
    _note(tmp_path, "blog/duckdb.md", ["duckdb", "sql", "#Python"])
    _note(tmp_path, "blog/one-tag.md", ["sql"])
    _note(tmp_path, "archive/old.md", ["duckdb", "sql"])
    _note(tmp_path, "daily/2026-09-01.md", ["daily", "journal"])
    _note(tmp_path, "notes/empty.md", ["a", "b"], body="")
    return tmp_path


class FakeProvider:
    """Returns canned VaultTagReport-shaped results keyed by note path."""

    def __init__(self, by_path: dict[str, list[str]], fail_on: set[str] = frozenset()):
        self.by_path = by_path
        self.fail_on = fail_on

    def complete(self, system, user, response_model=None):
        path = next(p for p in list(self.by_path) + list(self.fail_on) if f"FILE: {p}\n" in user)
        if path in self.fail_on:
            raise RuntimeError("unparseable")
        return response_model.model_validate(
            {
                "suggestions": [
                    {"file_path": path, "existing_tags": [], "suggested_tags": self.by_path[path], "reasoning": "r"}
                ]
            }
        )


def test_normalize_tag():
    assert normalize_tag("  #Python ") == "python"


class TestLoadTaggedNotes:
    def test_filters_excluded_dirs_min_tags_empty_bodies_and_ignored_tags(self, vault):
        notes = load_tagged_notes(vault)
        assert [n.path for n in notes] == ["blog/duckdb.md"]
        assert notes[0].tags == ["duckdb", "python", "sql"]

    def test_ignored_tag_can_drop_a_note_below_min_tags(self, tmp_path):
        _note(tmp_path, "x.md", ["daily", "sql"])
        assert load_tagged_notes(tmp_path) == []


def test_sample_is_reproducible(vault):
    notes = load_tagged_notes(vault, min_tags=1)
    assert sample_notes(notes, 1, seed=7) == sample_notes(notes, 1, seed=7)


class TestRunTagBacktest:
    def test_scores_hits_vocab_and_errors_without_aborting(self, vault):
        _note(vault, "blog/second.md", ["vectors", "sql"])
        notes = load_tagged_notes(vault)
        provider = FakeProvider({"blog/duckdb.md": ["DuckDB", "analytics"]}, fail_on={"blog/second.md"})
        seen = []
        results = run_tag_backtest(notes, provider, ["duckdb", "sql"], on_result=lambda i, n, r: seen.append(i))
        by_path = {r.path: r for r in results}
        ok = by_path["blog/duckdb.md"]
        assert ok.predicted == ["analytics", "duckdb"]
        assert ok.hits == 1
        assert ok.in_vocab == 1
        assert by_path["blog/second.md"].predicted is None
        assert "unparseable" in by_path["blog/second.md"].error
        assert seen == [1, 2]

    def test_candidate_never_sees_the_held_out_tags(self, vault):
        notes = load_tagged_notes(vault)
        captured = {}

        class Spy(FakeProvider):
            def complete(self, system, user, response_model=None):
                captured["user"] = user
                return super().complete(system, user, response_model)

        run_tag_backtest(notes, Spy({"blog/duckdb.md": ["x"]}), ["duckdb"])
        assert "CURRENT TAGS: []" in captured["user"]


class TestSummarize:
    def test_micro_precision_recall_f1(self):
        results = [
            TagResult("a", ["x", "y"], ["x", "z"], in_vocab=1, latency_s=1.0),
            TagResult("b", ["p", "q"], ["r"], in_vocab=1, latency_s=3.0),
            TagResult("c", ["m", "n"], None, in_vocab=0, latency_s=9.0, error="boom"),
        ]
        s = summarize_tags(results, "m")
        assert s.n_errors == 1
        assert s.precision == pytest.approx(1 / 3)
        assert s.recall == pytest.approx(1 / 4)
        assert s.f1 == pytest.approx(2 * (1 / 3) * (1 / 4) / ((1 / 3) + (1 / 4)))
        assert s.notes_with_any_hit == pytest.approx(0.5)
        assert s.vocab_adherence == pytest.approx(2 / 3)
        assert s.avg_latency_s == pytest.approx(2.0)

    def test_zero_hits_gives_zero_f1(self):
        s = summarize_tags([TagResult("a", ["x"], ["y"], 0, 1.0)], "m")
        assert s.f1 == 0.0

    def test_all_errors(self):
        s = summarize_tags([TagResult("a", ["x"], None, 0, 1.0, error="e")], "m")
        assert s.f1 is None
        assert "n/a" in render_tags_markdown(s)


def test_csv_round_trip(tmp_path):
    path = tmp_path / "out.csv"
    write_tags_csv([TagResult("a.md", ["x", "y"], ["x"], 1, 0.5)], path)
    lines = path.read_text().splitlines()
    assert lines[0].startswith("path,your_tags,suggested_tags")
    assert lines[1] == "a.md,x y,x,1,1,0.50,"
