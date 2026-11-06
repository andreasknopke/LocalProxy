"""Integration-Tests: Benutzer-isolierte Einstellungen.

Prueft, dass die per-Request-Zugriffe (Co-Worker, Loop-Guard, Sampling,
Model-Kategorien) ueber den Settings-Context korrekt pro Benutzer
unterschiedliche Werte liefern — und dass 'local' stets admin-only aus der
Baseline stammt. Konvention wie test_auth_users.py: import proxy, plain
def test_*(monkeypatch), asyncio.run(...) fuer _auth_or_raise.
"""
import asyncio

import pytest

import proxy
import users


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    users.reset_store_for_tests(tmp_path / "users.db")
    monkeypatch.setattr(proxy, "PROXY_AUTH_ENABLED", True)
    monkeypatch.setattr(proxy, "PROXY_API_KEY", "localfox-admin-bootstrap")
    # Deterministische Baseline statt echter config.json.
    monkeypatch.setattr(proxy, "_BASELINE_CONFIG", {
        "model_categories": {
            "local": [{"api_url": "http://local", "model_name": "local-m"}],
            "light": [{"api_url": "http://l1", "model_name": "base-light"}],
        },
        "default_category": "light",
        "tokens": {
            "loop_guard_enabled": True,
            "loop_guard_mode": "restart",
            "loop_guard_max_restarts": 1,
            "coworker": {"enabled": True, "enable_fork_join": True,
                         "max_parallel": 3},
            "local_sampling": {"temperature": 0.7, "thinking_mode": "none"},
        },
    })
    # Module-Globals als Baseline-Spiegel (Fallback ohne Context).
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
    monkeypatch.setattr(proxy, "LOOP_GUARD_ENABLED", True)
    monkeypatch.setattr(proxy, "LOOP_GUARD_MODE", "restart")
    monkeypatch.setattr(proxy, "LOOP_GUARD_MAX_RESTARTS", 1)
    monkeypatch.setattr(proxy, "DEFAULT_CATEGORY", "light")
    monkeypatch.setattr(proxy, "_MODEL_CATEGORIES",
                        proxy._BASELINE_CONFIG["model_categories"])
    proxy._ctx_user.set(None)
    proxy._ctx_settings.set(None)
    yield
    proxy._ctx_user.set(None)
    proxy._ctx_settings.set(None)


def _make_user(username, settings):
    store = users.get_store()
    u = store.create_user(username)
    store.set_settings(u.id, settings)
    return u


def _run_as(token, fn):
    """Fuehrt fn() innerhalb eines authentifizierten Request-Contexts aus."""
    async def _go():
        await proxy._auth_or_raise(_Req(token))
        return fn()
    return asyncio.run(_go())


class _Req:
    def __init__(self, token="", path="/v1/chat/completions"):
        self.url = type("U", (), {"path": path})()
        self._h = {}
        if token:
            self._h["authorization"] = "Bearer " + token

    @property
    def headers(self):
        return self._h


def test_default_user_gets_baseline():
    # Admin (Legacy-Key) ohne Overlay -> Baseline-Werte.
    res = _run_as("localfox-admin-bootstrap", lambda: {
        "cw": proxy._coworker_enabled_flag(),
        "lg": proxy._loop_guard_enabled(),
        "lgm": proxy._loop_guard_mode(),
        "cat": proxy._default_category(),
    })
    assert res["cw"] is True
    assert res["lg"] is True
    assert res["lgm"] == "restart"
    assert res["cat"] == "light"


def test_user_overlay_changes_coworker_and_loopguard():
    u = _make_user("bob", {
        "tokens": {
            "loop_guard_enabled": False,
            "loop_guard_mode": "note",
            "coworker": {"enabled": False},
        },
        "default_category": "strong",
    })
    tok = u._token
    res = _run_as(tok, lambda: {
        "cw": proxy._coworker_enabled_flag(),
        "lg": proxy._loop_guard_enabled(),
        "lgm": proxy._loop_guard_mode(),
        "cat": proxy._default_category(),
    })
    assert res["cw"] is False
    assert res["lg"] is False
    assert res["lgm"] == "note"
    assert res["cat"] == "strong"


def test_users_are_isolated_from_each_other():
    alice = _make_user("alice", {"tokens": {"loop_guard_mode": "note"}})
    bob = _make_user("bob", {"tokens": {"loop_guard_mode": "restart"}})
    a = _run_as(alice._token, lambda: proxy._loop_guard_mode())
    b = _run_as(bob._token, lambda: proxy._loop_guard_mode())
    assert a == "note"
    assert b == "restart"


def test_local_model_is_admin_only():
    # User versucht, local zu ändern -> Baseline bleibt.
    u = _make_user("mallory", {
        "model_categories": {
            "local": [{"api_url": "http://evil", "model_name": "evil-m"}],
            "light": [{"api_url": "http://x", "model_name": "user-light"}],
        },
    })
    res = _run_as(u._token, lambda: {
        "local": [d["model_name"] for d in proxy._model_defs("local")],
        "light": [d["model_name"] for d in proxy._model_defs("light")],
    })
    assert res["local"] == ["local-m"]       # erzwungen aus Baseline
    assert res["light"] == ["user-light"]    # user-spezifisch erlaubt


def test_sampling_is_per_user():
    u = _make_user("carol", {"tokens": {"local_sampling": {"temperature": 0.1}}})
    res = _run_as(u._token, lambda: proxy._local_sampling("temperature", 0.7))
    assert res == 0.1
    # Admin sieht Baseline.
    admin = _run_as("localfox-admin-bootstrap",
                    lambda: proxy._local_sampling("temperature", 0.7))
    assert admin == 0.7


def test_no_context_falls_back_to_globals():
    # Auhalb eines Requests (kein Context) -> Module-Globals.
    proxy._ctx_settings.set(None)
    assert proxy._coworker_enabled_flag() is True
    assert proxy._loop_guard_mode() == "restart"
    assert proxy._default_category() == "light"
