import csv

from model_comparison_harness.backtest import ItemResult
from model_comparison_harness.report import render_markdown, summarize, write_items_csv

THRESHOLD = 0.81


def _result(status, original, candidate, error=None):
    return ItemResult(1, "https://x.example", "title", status, original, candidate, 0.5, error)


class TestSummarize:
    def test_perfect_agreement(self):
        results = [_result("kept", 0.9, 0.9), _result("dismissed", 0.1, 0.1)]
        s = summarize(results, "test-model", THRESHOLD)
        assert s.decision_agreement_rate == 1.0
        assert s.false_dismiss_rate == 0.0
        assert s.false_keep_rate == 0.0

    def test_false_dismiss_counted(self):
        # Claude kept it (0.9), candidate would drop it (0.5 < 0.81 threshold)
        results = [_result("kept", 0.9, 0.5)]
        s = summarize(results, "test-model", THRESHOLD)
        assert s.false_dismiss_rate == 1.0
        assert s.false_keep_rate is None  # no dismissed items in this sample

    def test_false_keep_counted(self):
        # Claude dismissed it (0.1), candidate would keep it (0.9 >= 0.81 threshold)
        results = [_result("dismissed", 0.1, 0.9)]
        s = summarize(results, "test-model", THRESHOLD)
        assert s.false_keep_rate == 1.0
        assert s.false_dismiss_rate is None

    def test_errors_excluded_from_agreement_but_counted(self):
        results = [_result("kept", 0.9, 0.9), _result("kept", 0.9, None, error="boom")]
        s = summarize(results, "test-model", THRESHOLD)
        assert s.n_items == 2
        assert s.n_errors == 1
        assert s.n_scored == 1
        assert s.decision_agreement_rate == 1.0

    def test_all_errors_returns_none_metrics(self):
        results = [_result("kept", 0.9, None, error="boom")]
        s = summarize(results, "test-model", THRESHOLD)
        assert s.decision_agreement_rate is None
        assert s.mean_abs_score_diff is None

    def test_mean_abs_score_diff(self):
        results = [_result("kept", 0.9, 0.7), _result("dismissed", 0.1, 0.3)]
        s = summarize(results, "test-model", THRESHOLD)
        assert abs(s.mean_abs_score_diff - 0.2) < 1e-9


class TestRenderMarkdown:
    def test_includes_model_name_and_rates(self):
        results = [_result("kept", 0.9, 0.9)]
        s = summarize(results, "ollama/qwen2.5:7b", THRESHOLD)
        md = render_markdown(s)
        assert "ollama/qwen2.5:7b" in md
        assert "100%" in md

    def test_handles_none_metrics_gracefully(self):
        results = [_result("kept", 0.9, None, error="boom")]
        s = summarize(results, "test-model", THRESHOLD)
        md = render_markdown(s)
        assert "n/a" in md


class TestWriteItemsCsv:
    def test_writes_one_row_per_item_with_expected_columns(self, tmp_path):
        results = [
            ItemResult(1, "https://a.example", "title a", "kept", 0.9, 0.85, 1.5),
            ItemResult(2, "https://b.example", "title b", "dismissed", 0.1, 0.9, 2.0),
        ]
        path = tmp_path / "items.csv"
        write_items_csv(results, path, THRESHOLD)

        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        assert rows[0]["item_id"] == "1"
        assert rows[0]["claude_score"] == "0.9"
        assert rows[0]["candidate_score"] == "0.85"
        assert rows[0]["title"] == "title a"
        assert rows[0]["url"] == "https://a.example"

    def test_disagreement_row_is_identifiable(self, tmp_path):
        # Claude kept it (0.9), candidate would dismiss it (0.5 < 0.81) -- a real disagreement
        results = [ItemResult(1, "https://a.example", "title a", "kept", 0.9, 0.5, 1.0)]
        path = tmp_path / "items.csv"
        write_items_csv(results, path, THRESHOLD)

        with open(path, newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))

        assert row["agrees"] == "False"
        assert abs(float(row["abs_diff"]) - 0.4) < 1e-9

    def test_error_row_has_empty_score_and_agreement(self, tmp_path):
        results = [ItemResult(1, "https://a.example", "title a", "kept", 0.9, None, 1.0, error="no result")]
        path = tmp_path / "items.csv"
        write_items_csv(results, path, THRESHOLD)

        with open(path, newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))

        assert row["candidate_score"] == ""
        assert row["agrees"] == ""
        assert row["error"] == "no result"
