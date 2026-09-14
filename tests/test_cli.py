from model_comparison_harness.cli import _default_output_path


def test_default_output_path_includes_provider_model_limit_and_date():
    path = _default_output_path("ollama", "qwen2.5:7b", 200)
    assert path.name.endswith("-ollama-qwen2.5-7b-n200.md")
    assert path.parent.name == "results"


def test_default_output_path_handles_missing_model():
    path = _default_output_path("ollama", None, 50)
    assert "default" in path.name
    assert path.name.endswith("-n50.md")
