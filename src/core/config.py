import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class Config:
    """轻量级配置管理类，支持 .env 和 YAML 配置加载。"""

    _instance: Optional["Config"] = None
    _initialized: bool = False

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._config: Dict[str, Any] = {}
        self._load_env()
        self._load_yaml_config()
        self._initialized = True

    def _load_env(self) -> None:
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        for key, value in os.environ.items():
            self._config[key] = value

    def _load_yaml_config(self) -> None:
        yaml_path = Path(__file__).parent / "config.yaml"
        if not yaml_path.exists():
            return
        try:
            with open(yaml_path, encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
            if yaml_config:
                resolved_config = self._resolve_env_vars(yaml_config)
                self._merge_config(self._config, resolved_config)
        except yaml.YAMLError as e:
            logger.error(f"解析 config.yaml 失败: {e}")

    def _resolve_env_vars(self, value: Any) -> Any:
        """递归解析字符串中的 ${VAR_NAME} 环境变量引用。"""
        if isinstance(value, str):

            def replacer(match: re.Match) -> str:
                var_name = match.group(1)
                return os.environ.get(var_name, match.group(0))

            return _ENV_VAR_PATTERN.sub(replacer, value)
        if isinstance(value, dict):
            return {k: self._resolve_env_vars(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_env_vars(item) for item in value]
        return value

    def _merge_config(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_config(target[key], value)
            else:
                target[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        if "." in key:
            keys = key.split(".")
            value: Any = self._config
            for k in keys:
                if not isinstance(value, dict) or k not in value:
                    return default
                value = value[k]
            return value
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if "." in key:
            keys = key.split(".")
            config = self._config
            for k in keys[:-1]:
                if k not in config or not isinstance(config[k], dict):
                    config[k] = {}
                config = config[k]
            config[keys[-1]] = value
        else:
            self._config[key] = value

    def get_bool(self, key: str, default: bool = False) -> bool:
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1", "y", "t")
        return bool(value)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


config = Config()
