import sqlite3
from typing import Any, ClassVar

import pytest
from local_first_common.providers.base import BaseProvider

from model_comparison_harness.backtest import ItemResult, run_backtest, sample_items


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "store.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE items (
            id INTEGER PRIMARY KEY, url TEXT, title TEXT, description TEXT,
            status TEXT, score REAL, fetched_at TEXT
        )"""
    )
    rows = [
        (1, "https://a.example", "A kept item", "desc a", "kept", 0.9, "2026-09-27"),
        (2, "https://b.example", "B kept item", "desc b", "kept", 0.85, "2026-08-01"),
        (3, "https://c.example", "C dismissed item", "desc c", "dismissed", 0.2, "2026-09-28"),
        (4, "https://d.example", "D dismissed item", "desc d", "dismissed", 0.1, "2026-08-02"),
        (5, "https://e.example", "E no description", "", "dismissed", 0.0, "2026-09-28"),
    ]
    conn.executemany("INSERT INTO items VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
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

    def test_since_keeps_only_items_scored_under_the_current_profile(self, db_path):
        items = sample_items(db_path, limit=10, seed=1, since="2026-09-26")
        assert sorted(row["id"] for row in items) == [1, 3]

    def test_since_with_no_recent_stratum_fails_loudly(self, db_path):
        with pytest.raises(ValueError, match="since 2027-01-01"):
            sample_items(db_path, limit=4, seed=1, since="2027-01-01")

    def test_empty_stratum_fails_loudly(self, db_path):
        # A status rename in content-discovery-agent must not silently collapse
        # the stratified sample into a single-status backtest.
        conn = sqlite3.connect(db_path)
        conn.execute("UPDATE items SET status = 'retained' WHERE status = 'kept'")
        conn.commit()
        conn.close()
        with pytest.raises(ValueError, match="status 'kept'"):
            sample_items(db_path, limit=4, seed=1)


class _FakeScoredItem:
    def __init__(self, score):
        self.score = score
        self.tags = []
        self.summary = "fake"
        self.language = "en"


class _StubProvider(BaseProvider):
    """A BaseProvider whose subclasses override complete() directly."""

    provider_name = "stub"
    default_model = "stub"
    known_models: ClassVar[list[str]] = ["stub"]
    models_url = ""

    def _complete(self, system, user, response_model=None, images=None):
        raise NotImplementedError

    async def _acomplete(self, system, user, response_model=None, images=None):
        raise NotImplementedError


class _FakeProvider(_StubProvider):
    """ContentDiscoveryScorer.score() only calls .complete(); this answers from a list."""

    def __init__(self, responses):
        super().__init__()
        self._responses = list(responses)
        self.calls = 0

    def complete(self, system, user, response_model=None, images=None, max_retries=1, rate_limit_retries=3) -> Any:
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

    def test_source_location_and_item_count_set_before_each_call(self, db_path):
        """Regression 2026-09-20: an LLM call is logged once, inside the
        gateway -- source_location/item_count now travel to the gateway per
        item via provider.source_location/.item_count instead of the
        removed outer timed_run() wrapping the whole batch (which reported
        one row for N calls, not N)."""
        items = sample_items(db_path, limit=2, seed=1)
        provider = _FakeProvider(['{"score": 0.7, "tags": [], "summary": "x", "language": "en"}'] * len(items))
        run_backtest(items, provider, "testing")
        assert provider.source_location == items[-1]["title"]
        assert provider.item_count == 1

    def test_provider_exception_surfaces_as_no_result(self, db_path):
        # BaseScorer.score() catches provider exceptions internally and returns
        # None rather than raising -- this can't distinguish "provider down"
        # from "bad response" from the outside, and the error message says so.
        items = sample_items(db_path, limit=2, seed=1)

        class _Boom(_StubProvider):
            def complete(
                self, system, user, response_model=None, images=None, max_retries=1, rate_limit_retries=3
            ) -> Any:
                raise RuntimeError("provider down")

        results = run_backtest(items, _Boom(), "testing")
        assert len(results) == len(items)
        assert all(r.candidate_score is None for r in results)
        assert all(r.error is not None and "no result" in r.error for r in results)

    def test_unparseable_response_recorded_as_error(self, db_path):
        items = sample_items(db_path, limit=2, seed=1)
        provider = _FakeProvider(["not json at all"] * len(items))
        results = run_backtest(items, provider, "testing")
        assert all(r.candidate_score is None for r in results)
        assert all(r.error is not None and "no result" in r.error for r in results)


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
