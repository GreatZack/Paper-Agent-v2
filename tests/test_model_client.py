from unittest.mock import patch

import pytest

from src.core.model_client import (
    ModelClient,
    create_local_model_client,
    create_model_client,
)
from src.core.openai_client import OpenAICompatClient


@pytest.fixture
def valid_model_config():
    return {
        "model": {
            "name": "Qwen/Qwen3-32B",
            "api_key": "sk-test-key",
            "base_url": "https://api.test.com/v1",
            "capabilities": {
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "structured_output": True,
                "family": "Qwen",
            },
        },
    }


def test_model_client_create_client_with_valid_config(valid_model_config):
    with patch("src.core.model_client.config", valid_model_config):
        client = ModelClient.create_client(
            model="Qwen/Qwen3-32B",
            api_key="sk-test-key",
            base_url="https://api.test.com/v1",
        )

    assert isinstance(client, OpenAICompatClient)
    assert client._model == "Qwen/Qwen3-32B"


def test_model_client_create_client_missing_api_key(valid_model_config):
    with pytest.raises(ValueError, match="未配置 model.api_key"):
        ModelClient.create_client(
            model="Qwen/Qwen3-32B",
            api_key="",
            base_url="https://api.test.com/v1",
        )


def test_model_client_create_client_missing_base_url(valid_model_config):
    with pytest.raises(ValueError, match="未配置 model.base_url"):
        ModelClient.create_client(
            model="Qwen/Qwen3-32B",
            api_key="sk-test-key",
            base_url="",
        )


def test_create_model_client_with_valid_config(valid_model_config):
    with patch("src.core.model_client.config", valid_model_config):
        client = create_model_client()

    assert isinstance(client, OpenAICompatClient)
    assert client._base_url == "https://api.test.com/v1"


def test_create_model_client_missing_name(valid_model_config):
    bad_config = {
        "model": {
            "api_key": "sk-test-key",
            "base_url": "https://api.test.com/v1",
        }
    }
    with (
        pytest.raises(ValueError, match="未配置 model.name"),
        patch("src.core.model_client.config", bad_config),
    ):
        create_model_client()


def test_create_model_client_missing_config():
    empty_config = {}
    with (
        pytest.raises(ValueError, match="未配置 model.name"),
        patch("src.core.model_client.config", empty_config),
    ):
        create_model_client()


def test_create_model_client_rejects_unresolved_environment_placeholder():
    unresolved_config = {
        "model": {
            "name": "${MODEL_NAME}",
            "api_key": "${MODEL_API_KEY}",
            "base_url": "${MODEL_BASE_URL}",
        }
    }
    with (
        pytest.raises(ValueError, match="未配置 model.name"),
        patch("src.core.model_client.config", unresolved_config),
    ):
        create_model_client()


def test_create_local_model_client_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="本地模型调用功能尚未实现"):
        create_local_model_client(model_path="/path/to/local/model")
