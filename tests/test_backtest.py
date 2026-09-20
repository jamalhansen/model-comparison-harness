import sqlite3

import pytest

from model_comparison_harness.backtest import ItemResult, run_backtest, sample_items


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "store.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE items (
            id INTEGER PRIMARY KEY, url TEXT, title TEXT, description TEXT,
            status TEXT, score REAL
        )"""
    )
    rows = [
        (1, "https://a.example", "A kept item", "desc a", "kept", 0.9),
        (2, "https://b.example", "B kept item", "desc b", "kept", 0.85),
        (3, "https://c.example", "C dismissed item", "desc c", "dismissed", 0.2),
        (4, "https://d.example", "D dismissed item", "desc d", "dismissed", 0.1),
        (5, "https://e.example", "E no description", "", "dismissed", 0.0),
    ]
    conn.executemany("INSERT INTO items VALUES (?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return str(path)


class TestSampleItems:
    def test_excludes_empty_descriptions(self, db_path):
        items = sample_items(db_path, limit=10, seed=1)
        assert all(row["description"] for row in items)

    def test_stratifies_across_statuses(self, db_path):
        items = sample_items(db_path, limit=4, seed=1)
        statuses = [row["status"] for row in items]
        assert statuses.count("kept") == 2
        assert statuses.count("dismissed") == 2

    def test_reproducible_with_same_seed(self, db_path):
        first = [row["id"] for row in sample_items(db_path, limit=4, seed=7)]
        second = [row["id"] for row in sample_items(db_path, limit=4, seed=7)]
        assert first == second

    def test_caps_at_available_items_per_status(self, db_path):
        items = sample_items(db_path, limit=100, seed=1)
        # only 2 kept, 2 dismissed-with-description exist
        assert len(items) == 4


class _FakeScoredItem:
    def __init__(self, score):
        self.score = score
        self.tags = []
        self.summary = "fake"
        self.language = "en"


class _FakeProvider:
    """Not a real BaseProvider -- ContentDiscoveryScorer.score() only calls .complete()."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def complete(self, system_prompt, user_message):
        self.calls += 1
        return self._responses.pop(0)


class TestRunBacktest:
    def test_pairs_candidate_score_with_original(self, db_path):
        items = sample_items(db_path, limit=2, seed=1)
        provider = _FakeProvider(['{"score": 0.7, "tags": [], "summary": "x", "language": "en"}'] * len(items))
        results = run_backtest(items, provider, "testing")
        assert len(results) == len(items)
        for r in results:
            assert isinstance(r, ItemResult)
            assert r.candidate_score == 0.7
            assert r.error is None

    def test_on_result_callback_fires_once_per_item_with_1_indexed_progress(self, db_path):
        items = sample_items(db_path, limit=2, seed=1)
        provider = _FakeProvider(['{"score": 0.5, "tags": [], "summary": "x", "language": "en"}'] * len(items))
        calls = []
        run_backtest(items, provider, "testing", on_result=lambda i, total, r: calls.append((i, total, r)))
        assert [c[0] for c in calls] == [1, 2]
        assert all(c[1] == 2 for c in calls)
        assert all(isinstance(c[2], ItemResult) for c in calls)

    def test_no_callback_required(self, db_path):
        # on_result is optional -- existing callers that don't pass it must still work.
        items = sample_items(db_path, limit=2, seed=1)
        provider = _FakeProvider(['{"score": 0.5, "tags": [], "summary": "x", "language": "en"}'] * len(items))
        results = run_backtest(items, provider, "testing")
        assert len(results) == len(items)

    def test_provider_exception_surfaces_as_no_result(self, db_path):
        # BaseScorer.score() catches provider exceptions internally and returns
        # None rather than raising -- this can't distinguish "provider down"
        # from "bad response" from the outside, and the error message says so.
        items = sample_items(db_path, limit=2, seed=1)

        class _Boom:
            def complete(self, *a, **k):
                raise RuntimeError("provider down")

        results = run_backtest(items, _Boom(), "testing")
        assert len(results) == len(items)
        assert all(r.candidate_score is None for r in results)
        assert all("no result" in r.error for r in results)

    def test_unparseable_response_recorded_as_error(self, db_path):
        items = sample_items(db_path, limit=2, seed=1)
        provider = _FakeProvider(["not json at all"] * len(items))
        results = run_backtest(items, provider, "testing")
        assert all(r.candidate_score is None for r in results)
        assert all("no result" in r.error for r in results)


class TestItemResult:
    def test_agrees_true_when_decisions_match(self):
        r = ItemResult(1, "u", "t", "kept", 0.9, 0.85, 0.1)
        assert r.agrees(threshold=0.81) is True

    def test_agrees_false_when_decisions_differ(self):
        r = ItemResult(1, "u", "t", "kept", 0.9, 0.5, 0.1)
        assert r.agrees(threshold=0.81) is False

    def test_agrees_none_when_candidate_errored(self):
        r = ItemResult(1, "u", "t", "kept", 0.9, None, 0.1, error="boom")
        assert r.agrees(threshold=0.81) is None
