from model_comparison_harness.cli import _default_output_path


def test_default_output_path_includes_provider_model_limit_and_date():
    path = _default_output_path("ollama", "qwen2.5:7b", 200)
    assert path.name.endswith("-ollama-qwen2.5-7b-n200.md")
    assert path.parent.name == "results"


def test_default_output_path_handles_missing_model():
    path = _default_output_path("ollama", None, 50)
    assert "default" in path.name
    assert path.name.endswith("-n50.md")


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
