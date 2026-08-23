"""Smoke tests (part C): cooldowns JSON, webui import, config category."""

import json
import importlib
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_cooldowns_json_valid():
    path = REPO_ROOT / "data" / "cooldowns.json"
    if not path.exists():
        pytest.skip("data/cooldowns.json does not exist")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), (
        f"cooldowns.json must be a JSON object (dict), got {type(data).__name__}"
    )


def test_webui_module_imports():
    webui = importlib.import_module("webui")
    assert hasattr(webui, "create_webui_app"), (
        "webui module must define 'create_webui_app'"
    )


def test_config_default_category_known():
    path = REPO_ROOT / "data" / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    categories = config.get("model_categories", {})
    default_cat = config.get("default_category")
    assert default_cat is not None, "config.json missing 'default_category'"
    assert default_cat in categories, (
        f"default_category '{default_cat}' not in model_categories keys "
        f"{list(categories.keys())}"
    )
