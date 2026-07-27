from unittest.mock import patch

import pytest

from src.core.config import Config


@pytest.fixture
def fresh_config():
    """提供一个未初始化的 Config 实例用于测试。"""
    with (
        patch.object(Config, "_instance", None),
        patch.object(Config, "_initialized", False),
    ):
        yield Config


def test_config_resolves_env_vars_in_yaml(fresh_config, tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-test-from-env")
    monkeypatch.setenv("TEST_BASE_URL", "https://test.example.com/v1")

    yaml_content = """
model:
  name: Qwen/Qwen3-32B
  api_key: ${TEST_API_KEY}
  base_url: ${TEST_BASE_URL}
"""
    with patch("src.core.config.Path") as mock_path:
        root_path = tmp_path
        src_path = tmp_path / "src" / "core"
        src_path.mkdir(parents=True)

        yaml_file = src_path / "config.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        def path_factory(*args, **kwargs):
            if str(args[0]).endswith(".env"):
                return root_path / ".env"
            return src_path / "config.yaml"

        mock_path.side_effect = path_factory
        cfg = fresh_config()

    assert cfg.get("model.api_key") == "sk-test-from-env"
    assert cfg.get("model.base_url") == "https://test.example.com/v1"


def test_config_keeps_unresolved_placeholder_when_env_missing(
    fresh_config, tmp_path, monkeypatch
):
    monkeypatch.delenv("MISSING_KEY", raising=False)

    yaml_content = """
model:
  api_key: ${MISSING_KEY}
"""
    with patch("src.core.config.Path") as mock_path:
        src_path = tmp_path / "src" / "core"
        src_path.mkdir(parents=True)
        yaml_file = src_path / "config.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        def path_factory(*args, **kwargs):
            if str(args[0]).endswith(".env"):
                return tmp_path / ".env"
            return yaml_file

        mock_path.side_effect = path_factory
        cfg = fresh_config()

    assert cfg.get("model.api_key") == "${MISSING_KEY}"


def test_config_env_overrides_yaml(fresh_config, tmp_path, monkeypatch):
    monkeypatch.setenv("OVERRIDE_MODEL", "override-model")

    yaml_content = """
model:
  name: original-model
"""
    with patch("src.core.config.Path") as mock_path:
        src_path = tmp_path / "src" / "core"
        src_path.mkdir(parents=True)
        yaml_file = src_path / "config.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        def path_factory(*args, **kwargs):
            if str(args[0]).endswith(".env"):
                return tmp_path / ".env"
            return yaml_file

        mock_path.side_effect = path_factory
        cfg = fresh_config()

    assert cfg.get("model.name") == "original-model"
    assert cfg.get("OVERRIDE_MODEL") == "override-model"
