"""Smoke-Tests für LocalProxy (plain def + assert, kein pytest-asyncio)."""

import importlib
import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_config():
    config_path = REPO_ROOT / "data" / "config.json"
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def test_config_json_loads():
    """data/config.json existiert und enthält 'model_categories' und 'tokens'."""
    config_path = REPO_ROOT / "data" / "config.json"
    assert config_path.is_file(), f"config.json nicht gefunden: {config_path}"
    config = _load_config()
    assert "model_categories" in config, "'model_categories' fehlt in config.json"
    assert "tokens" in config, "'tokens' fehlt in config.json"


def test_coworker_config_shape():
    """tokens.coworker enthält die erwarteten Keys (enabled, enable_fork_join, max_parallel)."""
    config = _load_config()
    coworker = config["tokens"]["coworker"]
    assert "enabled" in coworker, "'enabled' fehlt in tokens.coworker"
    assert "enable_fork_join" in coworker, "'enable_fork_join' fehlt in tokens.coworker"
    assert "max_parallel" in coworker, "'max_parallel' fehlt in tokens.coworker"


def test_proxy_module_imports():
    """proxy-Modul ist importierbar und exponiert 'app'."""
    proxy = importlib.import_module("proxy")
    assert hasattr(proxy, "app"), "proxy.app fehlt"
