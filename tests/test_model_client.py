from unittest.mock import patch

import pytest
from autogen_ext.models.openai import OpenAIChatCompletionClient

from src.core.model_client import (
    ModelClient,
    create_local_model_client,
    create_model_client,
    create_search_model_client,
)


@pytest.fixture
def valid_model_config():
    return {
        "search-model": {
            "model-provider": "siliconflow",
            "model": "Qwen/Qwen3-32B",
            "api_key": "sk-test-key",
            "base_url": "https://api.test.com/v1",
        },
        "siliconflow": {
            "api_key": "sk-test-key",
            "base_url": "https://api.test.com/v1",
        },
    }


def test_model_client_create_client_with_valid_config(valid_model_config):
    with patch("src.core.model_client.config", valid_model_config):
        client = ModelClient.create_client(
            provider="siliconflow",
            model="Qwen/Qwen3-32B",
        )

    assert isinstance(client, OpenAIChatCompletionClient)
    assert client.model_info is not None


def test_model_client_create_client_missing_api_key(valid_model_config):
    config_without_key = {
        "siliconflow": {
            "api_key": None,
            "base_url": "https://api.test.com/v1",
        }
    }
    with pytest.raises(ValueError, match="未配置 siliconflow 的 api_key"):
        with patch("src.core.model_client.config", config_without_key):
            ModelClient.create_client(provider="siliconflow", model="Qwen/Qwen3-32B")


def test_model_client_create_client_missing_base_url(valid_model_config):
    config_without_url = {
        "siliconflow": {
            "api_key": "sk-test-key",
            "base_url": None,
        }
    }
    with pytest.raises(ValueError, match="未配置 siliconflow 的 base_url"):
        with patch("src.core.model_client.config", config_without_url):
            ModelClient.create_client(provider="siliconflow", model="Qwen/Qwen3-32B")


def test_create_model_client_with_valid_config(valid_model_config):
    with patch("src.core.model_client.config", valid_model_config):
        client = create_model_client("search-model")

    assert isinstance(client, OpenAIChatCompletionClient)


def test_create_model_client_missing_provider(valid_model_config):
    bad_config = {
        "search-model": {
            "model": "Qwen/Qwen3-32B",
        }
    }
    with pytest.raises(ValueError, match="未配置 search-model 的 model-provider 或 model"):
        with patch("src.core.model_client.config", bad_config):
            create_model_client("search-model")


def test_create_model_client_missing_model(valid_model_config):
    bad_config = {
        "search-model": {
            "model-provider": "siliconflow",
        }
    }
    with pytest.raises(ValueError, match="未配置 search-model 的 model-provider 或 model"):
        with patch("src.core.model_client.config", bad_config):
            create_model_client("search-model")


def test_create_search_model_client_with_valid_config(valid_model_config):
    with patch("src.core.model_client.config", valid_model_config):
        client = create_search_model_client()

    assert isinstance(client, OpenAIChatCompletionClient)


def test_create_search_model_client_missing_config():
    empty_config = {}
    with pytest.raises(ValueError, match="未配置 search-model 的 model-provider 或 model"):
        with patch("src.core.model_client.config", empty_config):
            create_search_model_client()


def test_create_local_model_client_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="本地模型调用功能尚未实现"):
        create_local_model_client(model_path="/path/to/local/model")
