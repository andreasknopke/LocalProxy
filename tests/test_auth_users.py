"""Tests fuer die Benutzer-Authorisierung (Bearer-Token -> User).

Konvention wie test_io_trace.py: import proxy, plain def test_*(monkeypatch),
innere asyncio.run(...) fuer async _auth_or_raise. Kein TestClient.
"""
import asyncio

import pytest

import proxy
import users


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Leerer Benutzer-Store + deterministischer Admin-Token pro Test."""
    users.reset_store_for_tests(tmp_path / "users.db")
    monkeypatch.setattr(proxy, "PROXY_AUTH_ENABLED", True)
    monkeypatch.setattr(proxy, "PROXY_API_KEY", "localfox-admin-bootstrap")
    proxy._ctx_user.set(None)
    yield
    proxy._ctx_user.set(None)


class _Req:
    """Minimaler Request-Stub: nur url.path + authorization-Header."""
    def __init__(self, token: str = "", path: str = "/v1/chat/completions"):
        self.url = type("U", (), {"path": path})()
        self._h = {}
        if token:
            self._h["authorization"] = "Bearer " + token

    def headers_get(self, k, d=""):
        return self._h.get(k.lower(), d)

    # proxy liest request.headers.get(...)
    @property
    def headers(self):
        return self._h


def test_resolve_admin_via_legacy_key():
    u = proxy._resolve_user("localfox-admin-bootstrap")
    assert u is not None
    assert u.is_admin is True
    assert u.username == "admin"


def test_resolve_unknown_token_none():
    assert proxy._resolve_user("localfox-garbage") is None
    assert proxy._resolve_user("") is None


def test_resolve_db_user():
    store = users.get_store()
    u = store.create_user("alice")
    tok = u._token
    got = proxy._resolve_user(tok)
    assert got is not None
    assert got.username == "alice"
    assert got.is_admin is False


def test_auth_or_raise_sets_user_context():
    store = users.get_store()
    u = store.create_user("bob")
    req = _Req(token=u._token)

    async def _run():
        await proxy._auth_or_raise(req)
        return proxy._ctx_user.get()

    got = asyncio.run(_run())
    assert got is not None
    assert got.username == "bob"


def test_auth_or_raise_admin_context():
    req = _Req(token="localfox-admin-bootstrap")

    async def _run():
        await proxy._auth_or_raise(req)
        return proxy._ctx_user.get()

    got = asyncio.run(_run())
    assert got is not None
    assert got.is_admin is True


def test_auth_or_raise_rejects_unknown():
    req = _Req(token="localfox-nope")
    with pytest.raises(proxy.HTTPException) as ei:
        asyncio.run(proxy._auth_or_raise(req))
    assert ei.value.status_code == 401


def test_auth_or_raise_rejects_missing_token():
    req = _Req(token="")
    with pytest.raises(proxy.HTTPException) as ei:
        asyncio.run(proxy._auth_or_raise(req))
    assert ei.value.status_code == 401


def test_auth_disabled_sets_none_user():
    # Auth aus -> kein User, aber kein Fehler
    proxy.PROXY_AUTH_ENABLED = False
    req = _Req(token="")
    asyncio.run(proxy._auth_or_raise(req))
    assert proxy._ctx_user.get() is None


def test_is_authorized_reflects_resolution():
    store = users.get_store()
    u = store.create_user("carol")
    assert proxy._is_authorized(_Req(token=u._token)) is True
    assert proxy._is_authorized(_Req(token="localfox-admin-bootstrap")) is True
    assert proxy._is_authorized(_Req(token="wrong")) is False


# ─── ADMIN_TOKEN (ENV) ─────────────────────────────────────────────────────

def test_admin_token_resolves_admin(monkeypatch):
    monkeypatch.setattr(proxy, "ADMIN_TOKEN", "localfox-fixadmin")
    monkeypatch.setattr(proxy, "PROXY_API_KEY", "localfox-fixadmin")
    u = proxy._resolve_user("localfox-fixadmin")
    assert u is not None and u.is_admin


def test_apply_config_file_does_not_override_admin_token(monkeypatch):
    monkeypatch.setattr(proxy, "ADMIN_TOKEN", "localfox-fixadmin")
    monkeypatch.setattr(proxy, "PROXY_API_KEY", "localfox-fixadmin")
    cfg = {"proxy": {"api_key": "localfox-aus-config"}}
    monkeypatch.setattr(proxy, "_webui_load_config", lambda: cfg)
    proxy._apply_config_file()
    assert proxy.PROXY_API_KEY == "localfox-fixadmin"


def test_apply_config_file_without_admin_token_uses_config(monkeypatch):
    monkeypatch.setattr(proxy, "ADMIN_TOKEN", "")
    monkeypatch.setattr(proxy, "PROXY_API_KEY", "localfox-alt")
    cfg = {"proxy": {"api_key": "localfox-aus-config"}}
    monkeypatch.setattr(proxy, "_webui_load_config", lambda: cfg)
    proxy._apply_config_file()
    assert proxy.PROXY_API_KEY == "localfox-aus-config"


# ── Fallback ohne users-Modul (z.B. users.py fehlt im Deployment-Image) ───

def test_admin_bootstrap_works_without_users_module(monkeypatch):
    monkeypatch.setattr(proxy, "_USERS_AVAILABLE", False)
    monkeypatch.setattr(proxy, "PROXY_API_KEY", "EraseTheMap!")
    u = proxy._resolve_user("EraseTheMap!")
    assert u is not None and u.is_admin and u.username == "admin"
    assert proxy._resolve_user("falsch") is None
