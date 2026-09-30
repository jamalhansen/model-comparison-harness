"""Replay content-discovery-agent's real scoring history against a candidate model.

The question this answers: content-discovery-agent has scored 6,000+ real items with
Claude (`scoring_provider = "anthropic"` in its live config), and every one of those
scores is sitting in its store as ground truth nobody's checked against anything else.
This re-runs the same scorer, same prompt, same interest profile against a candidate
model and measures where it agrees and disagrees with Claude, instead of guessing.
"""

import random
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass

from discovery.scorer import ContentDiscoveryScorer, build_user_message
from local_first_common.providers.base import BaseProvider


@dataclass
class ItemResult:
    """One item's original (Claude) score compared against a candidate model's score."""

    item_id: int
    url: str
    title: str
    status: str  # the human's real decision: 'kept' or 'dismissed'
    original_score: float
    candidate_score: float | None
    latency_s: float
    error: str | None = None

    @property
    def original_decision(self) -> bool:
        return self.status == "kept"

    def candidate_decision(self, threshold: float) -> bool | None:
        if self.candidate_score is None:
            return None
        return self.candidate_score >= threshold

    def agrees(self, threshold: float) -> bool | None:
        decision = self.candidate_decision(threshold)
        if decision is None:
            return None
        return decision == self.original_decision


def sample_items(db_path: str, limit: int, seed: int, since: str | None = None) -> list[sqlite3.Row]:
    """Stratified sample: half kept, half dismissed (as close to that as the data allows).

    `since` (ISO date) keeps only items fetched on/after it. Stored scores and keep/dismiss
    decisions are only ground truth for the interest profile they were made under: after
    the 2026-09-26 profile rewrite, a run over older items graded the new profile against
    old decisions and reported a meaningless 92% false-dismiss rate.

    A pure random sample would be ~94% dismissed (the real traffic mix) and barely
    exercise whether a candidate model can actually recognize a good item -- the
    question that matters more here than reproducing the traffic mix exactly.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        per_status = max(1, limit // 2)
        rng = random.Random(seed)
        rows: list[sqlite3.Row] = []
        where, params = "status = ? AND description != ''", []
        if since:
            where += " AND fetched_at >= ?"
            params = [since]
        for status in ("kept", "dismissed"):
            ids = [
                r[0]
                for r in conn.execute(f"SELECT id FROM items WHERE {where}", (status, *params)).fetchall()
            ]
            # An empty stratum must fail loudly: this raw query couples to
            # content-discovery-agent's status vocabulary, and a rename there
            # would otherwise silently collapse the stratified sample into a
            # single-status backtest with meaningless agreement numbers.
            if not ids:
                raise ValueError(
                    f"No scorable items with status '{status}'{f' since {since}' if since else ''} in {db_path} -- either the "
                    f"store has no {status} items yet, or content-discovery-agent's status "
                    f"vocabulary changed out from under this query."
                )
            chosen = rng.sample(ids, k=min(per_status, len(ids)))
            placeholders = ",".join("?" * len(chosen))
            if chosen:
                rows.extend(
                    conn.execute(
                        f"SELECT id, url, title, description, status, score FROM items "
                        f"WHERE id IN ({placeholders})",
                        chosen,
                    ).fetchall()
                )
        return rows
    finally:
        conn.close()


def run_backtest(
    items: list[sqlite3.Row],
    provider: BaseProvider,
    interest_profile: str,
    exclusions: str = "",
    on_result: Callable[[int, int, "ItemResult"], None] | None = None,
) -> list[ItemResult]:
    """Score every sampled item with `provider` and pair it against its stored score.

    `on_result(i, total, result)` fires after each item completes, 1-indexed --
    a 200-item run against a verbose model can take over an hour with zero
    visibility otherwise (confirmed live 2026-09-13: gemma4 took 68+ minutes on
    a run llama3.2:3b finished in 5, with no way to tell how far along it was).
    """
    # ContentDiscoveryScorer.score() (via BaseScorer) already catches provider
    # exceptions internally and returns None rather than raising -- it logs a
    # warning but doesn't expose which failure mode it was. That means a dead
    # network and a malformed JSON response are indistinguishable from here;
    # the error message below says so honestly rather than guessing.
    scorer = ContentDiscoveryScorer()
    results: list[ItemResult] = []
    total = len(items)
    for i, item in enumerate(items, start=1):
        user_message = build_user_message(item["title"], item["description"], interest_profile, exclusions)
        provider.source_location = item["title"]
        provider.item_count = 1
        start = time.monotonic()
        scored = scorer.score(provider, user_message)
        latency = time.monotonic() - start
        result = ItemResult(
            item_id=item["id"],
            url=item["url"],
            title=item["title"],
            status=item["status"],
            original_score=item["score"],
            candidate_score=scored.score if scored else None,
            latency_s=latency,
            error=None if scored else "no result (provider error or unparseable response, see log)",
        )
        results.append(result)
        if on_result:
            on_result(i, total, result)
    return results
