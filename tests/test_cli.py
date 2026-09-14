from model_comparison_harness.backtest import ItemResult
from model_comparison_harness.cli import _default_output_path, _print_progress


def _result(status="kept", original=0.9, candidate=0.85, error=None):
    return ItemResult(1, "https://x.example", "a title", status, original, candidate, 0.1, error)


def test_default_output_path_includes_provider_model_limit_and_date():
    path = _default_output_path("ollama", "qwen2.5:7b", 200)
    assert path.name.endswith("-ollama-qwen2.5-7b-n200.md")
    assert path.parent.name == "results"


def test_default_output_path_handles_missing_model():
    path = _default_output_path("ollama", None, 50)
    assert "default" in path.name
    assert path.name.endswith("-n50.md")


class TestPrintProgress:
    def test_compact_form_shows_position_and_eta(self, capsys):
        _print_progress(1, 200, _result(), cutoff=0.81, verbose=False, run_start=0.0)
        out = capsys.readouterr().out
        assert "[1/200]" in out
        assert "remaining" in out

    def test_verbose_form_includes_scores_and_title(self, capsys):
        _print_progress(1, 200, _result(), cutoff=0.81, verbose=True, run_start=0.0)
        out = capsys.readouterr().out
        assert "[1/200]" in out
        assert "claude=0.90" in out
        assert "a title" in out

    def test_verbose_form_flags_disagreement(self, capsys):
        r = _result(status="kept", original=0.9, candidate=0.2)  # kept by Claude, would be dismissed
        _print_progress(1, 200, r, cutoff=0.81, verbose=True, run_start=0.0)
        out = capsys.readouterr().out
        assert "DISAGREE" in out

    def test_verbose_form_flags_error_rather_than_agreement(self, capsys):
        r = _result(error="no result (provider error or unparseable response, see log)")
        _print_progress(1, 200, r, cutoff=0.81, verbose=True, run_start=0.0)
        out = capsys.readouterr().out
        assert "[?]" in out


def test_results_dir_env_var_override(monkeypatch, tmp_path):
    # Must be settable independent of where the package happens to be installed --
    # __file__-derived paths broke silently once this shipped as an installed uv tool.
    monkeypatch.setenv("MODEL_COMPARISON_HARNESS_RESULTS_DIR", str(tmp_path / "custom-results"))
    import importlib

    import model_comparison_harness.cli as cli_module

    importlib.reload(cli_module)
    try:
        assert cli_module._RESULTS_DIR == tmp_path / "custom-results"
    finally:
        monkeypatch.delenv("MODEL_COMPARISON_HARNESS_RESULTS_DIR", raising=False)
        importlib.reload(cli_module)
