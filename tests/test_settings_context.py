"""Tests fuer die Settings-Auflösung (Baseline ⊕ User-Overlay).

Konvention: import proxy, plain def test_*(monkeypatch). Kein TestClient.
"""
import copy

import pytest

import proxy
import users


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    users.reset_store_for_tests(tmp_path / "users.db")
    monkeypatch.setattr(proxy, "PROXY_AUTH_ENABLED", True)
    monkeypatch.setattr(proxy, "PROXY_API_KEY", "localfox-admin")
    # Deterministische Baseline
    monkeypatch.setattr(proxy, "_BASELINE_CONFIG", {
        "model_categories": {
            "local": {"api_url": "http://local:8000/v1", "model_name": "LocalM",
                      "api_key": ""},
            "light": [{"label": "l", "api_url": "http://l", "model_name": "L",
                       "api_key": ""}],
        },
        "default_category": "light",
        "tokens": {
            "loop_guard_enabled": True,
            "loop_guard_mode": "restart",
            "coworker": {"enabled": True, "max_delegations_per_request": 2},
            "local_sampling": {"temperature": 0.7},
        },
    })
    proxy._ctx_user.set(None)
    proxy._ctx_settings.set(None)
    yield
    proxy._ctx_user.set(None)
    proxy._ctx_settings.set(None)


def _user_with(settings):
    return users.User(id=1, username="u", is_admin=False,
                      token_hash="", created_at="", settings=settings)


def test_no_user_returns_baseline():
    eff = proxy._effective_settings(None)
    assert eff["default_category"] == "light"
    assert eff["tokens"]["loop_guard_enabled"] is True


def test_user_overlay_merges():
    u = _user_with({"default_category": "strong",
                    "tokens": {"loop_guard_enabled": False}})
    eff = proxy._effective_settings(u)
    assert eff["default_category"] == "strong"
    # tiefes Merge: loop_guard_enabled aus User, loop_guard_mode bleibt Baseline
    assert eff["tokens"]["loop_guard_enabled"] is False
    assert eff["tokens"]["loop_guard_mode"] == "restart"
    # Baseline unveraendert (kein Seiteneffekt)
    assert proxy._BASELINE_CONFIG["default_category"] == "light"
    assert proxy._BASELINE_CONFIG["tokens"]["loop_guard_enabled"] is True


def test_local_category_always_from_baseline():
    """User darf 'local' NICHT überschreiben — admin-only."""
    u = _user_with({"model_categories": {
        "local": {"api_url": "http://evil", "model_name": "Evil"},
        "light": [{"label": "x", "api_url": "http://x", "model_name": "X",
                   "api_key": ""}],
    }})
    eff = proxy._effective_settings(u)
    assert eff["model_categories"]["local"]["model_name"] == "LocalM"
    assert eff["model_categories"]["local"]["api_url"] == "http://local:8000/v1"
    # light darf der User aber überschreiben
    assert eff["model_categories"]["light"][0]["model_name"] == "X"


def test_deep_merge_no_mutation():
    base = {"a": {"b": 1}, "c": 2}
    over = {"a": {"d": 3}}
    out = proxy._deep_merge(base, over)
    assert out == {"a": {"b": 1, "d": 3}, "c": 2}
    assert base == {"a": {"b": 1}, "c": 2}  # base unveraendert


def test_apply_user_settings_sets_context():
    u = _user_with({"default_category": "vision"})
    proxy._apply_user_settings(u)
    assert proxy._ctx_settings.get()["default_category"] == "vision"


def test_cfg_get_reads_effective():
    u = _user_with({"tokens": {"loop_guard_mode": "note"}})
    proxy._apply_user_settings(u)
    assert proxy._cfg_get("tokens", "loop_guard_mode") == "note"
    assert proxy._cfg_get("tokens", "loop_guard_enabled") is True  # baseline
    assert proxy._cfg_get("tokens", "fehitlend", default="dflt") == "dflt"


def test_cfg_get_no_ctx_returns_default():
    # Ohne aktiven Context wird das Module-Global (hier ueber `default`
    # uebergeben) genutzt — kein Rueckgriff auf die Baseline.
    proxy._ctx_settings.set(None)
    assert proxy._cfg_get("default_category", default="global") == "global"
    assert proxy._cfg_get("nope", "nope2", default=42) == 42


def test_cfg_get_with_ctx_falls_back_to_baseline():
    # Mit aktivem Context: User-Overlay, sonst Baseline, sonst default.
    u = _user_with({"tokens": {"loop_guard_mode": "note"}})
    proxy._apply_user_settings(u)
    assert proxy._cfg_get("default_category", default="global") == "light"  # baseline
    assert proxy._cfg_get("tokens", "fehitlend", default="dflt") == "dflt"
