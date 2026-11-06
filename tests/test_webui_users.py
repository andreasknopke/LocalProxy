"""Tests fuer WebUI-Benutzerauthentifizierung, benutzerspezifische Config und
Admin-Benutzerverwaltung. Nutzt FastAPI TestClient gegen create_webui_app()."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import proxy
import users
import webui

ADMIN_TOKEN = "localfox-admin-token-for-tests"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    users.reset_store_for_tests(tmp_path / "users.db")
    cfg = tmp_path / "config.json"
    cfg.write_text(
        '{"model_categories": {'
        '"local": {"api_url": "http://local", "model_name": "local-m", "api_key": "secret-local"},'
        '"light": [{"api_url": "http://l1", "model_name": "base-light", "api_key": "k1"}]},'
        '"default_category": "light",'
        '"tokens": {"loop_guard_enabled": true, "coworker": {"enabled": true}}}',
        encoding="utf-8")
    monkeypatch.setattr(webui, "CONFIG_PATH", cfg)
    monkeypatch.setattr(webui, "WEBUI_USERNAME", "admin")
    monkeypatch.setattr(webui, "_active_tokens", set())
    monkeypatch.setattr(webui, "_session_users", {})
    # Kein echter Proxy-Reload in Tests.
    monkeypatch.setattr(webui, "_reload_proxy_config", lambda: None)
    # Admin-Datensatz mit bekanntem Token anlegen.
    users.get_store().create_user("admin", token=ADMIN_TOKEN, is_admin=True)
    yield


def _client():
    parent = FastAPI()
    webui.mount_webui(parent)
    return TestClient(parent)


def _login_admin(client):
    r = client.post("/webui/api/login",
                    json={"username": "admin", "password": ADMIN_TOKEN})
    assert r.status_code == 200
    return r.json()["token"]


def _login_user(client, username, token):
    return client.post("/webui/api/login", json={"username": username, "password": token})


def _with_token(client, token):
    client.cookies.set(webui.COOKIE_NAME, token, path="/webui")
    return client


def test_legacy_password_login_rejected():
    # Der alte Login per WEBUI_PASSWORD existiert nicht mehr.
    c = _client()
    r = c.post("/webui/api/login",
               json={"username": "admin", "password": "adminpw"})
    assert r.status_code == 401


def test_admin_login_sets_admin_session():
    c = _client()
    tok = _login_admin(c)
    assert webui._is_admin_session(tok) is True


def test_user_login_with_bearer_token():
    store = users.get_store()
    u = store.create_user("alice")
    c = _client()
    r = _login_user(c, "alice", u._token)
    assert r.status_code == 200
    tok = r.json()["token"]
    assert webui._is_admin_session(tok) is False
    su = webui._session_user(tok)
    assert su is not None and su.username == "alice"


def test_user_login_wrong_token_rejected():
    store = users.get_store()
    store.create_user("bob")
    c = _client()
    r = _login_user(c, "bob", "wrong-token")
    assert r.status_code == 401


def test_config_get_masks_keys():
    c = _client()
    _with_token(c, _login_admin(c))
    r = c.get("/webui/api/config")
    assert r.status_code == 200
    cfg = r.json()
    assert "secret-local" not in str(cfg)
    assert "\u2022" in cfg["model_categories"]["local"]["api_key"]


def test_user_config_save_stores_overlay_not_global():
    store = users.get_store()
    u = store.create_user("carol")
    c = _client()
    _with_token(c, _login_user(c, "carol", u._token).json()["token"])
    # User aendert default_category und loop_guard.
    c.put("/webui/api/config", json={"default_category": "strong",
                               "tokens": {"loop_guard_enabled": False}})
    overlay = store.get_settings(u.id)
    assert overlay["default_category"] == "strong"
    assert overlay["tokens"]["loop_guard_enabled"] is False
    # Globale config bleibt unveraendert.
    assert webui._load_config()["default_category"] == "light"


def test_user_cannot_change_local_category():
    store = users.get_store()
    u = store.create_user("dave")
    c = _client()
    _with_token(c, _login_user(c, "dave", u._token).json()["token"])
    c.put("/webui/api/config", json={
        "model_categories": {"local": {"api_url": "http://evil",
                                       "model_name": "evil", "api_key": "x"}}})
    overlay = store.get_settings(u.id) or {}
    assert "local" not in overlay.get("model_categories", {})


def test_admin_config_save_writes_global():
    c = _client()
    _with_token(c, _login_admin(c))
    c.put("/webui/api/config", json={"default_category": "vision"})
    assert webui._load_config()["default_category"] == "vision"


def test_admin_user_crud():
    c = _client()
    _with_token(c, _login_admin(c))
    # create
    r = c.post("/webui/api/users", json={"username": "erin"})
    assert r.status_code == 200
    uid = r.json()["id"]
    assert "token" in r.json()
    # list
    r = c.get("/webui/api/users")
    assert any(u["username"] == "erin" for u in r.json()["users"])
    # rotate
    r = c.post(f"/webui/api/users/{uid}/rotate")
    assert r.status_code == 200 and "token" in r.json()
    # delete
    r = c.delete(f"/webui/api/users/{uid}")
    assert r.status_code == 200
    r = c.get("/webui/api/users")
    assert not any(u["id"] == uid for u in r.json()["users"])


def test_me_endpoint_reports_admin():
    c = _client()
    _with_token(c, _login_admin(c))
    r = c.get("/webui/api/me")
    assert r.status_code == 200
    assert r.json()["is_admin"] is True
    assert r.json()["username"] == "admin"


def test_me_endpoint_reports_regular_user():
    store = users.get_store()
    u = store.create_user("gina")
    c = _client()
    _with_token(c, _login_user(c, "gina", u._token).json()["token"])
    r = c.get("/webui/api/me")
    assert r.json()["is_admin"] is False
    assert r.json()["username"] == "gina"


def test_user_can_rotate_own_token():
    store = users.get_store()
    u = store.create_user("heidi")
    old_token = u._token
    c = _client()
    _with_token(c, _login_user(c, "heidi", old_token).json()["token"])
    r = c.post("/webui/api/me/rotate")
    assert r.status_code == 200
    new_token = r.json()["token"]
    assert new_token != old_token
    # Neuer Token funktioniert, alter nicht mehr.
    assert store.get_by_token(new_token) is not None
    assert store.get_by_token(old_token) is None


def test_rotate_own_token_requires_login():
    c = _client()
    # Ohne gueltige Session leitet das Middleware-Redirect um (302) oder 403.
    r = c.post("/webui/api/me/rotate", follow_redirects=False)
    assert r.status_code in (302, 403)


def test_non_admin_cannot_access_user_admin():
    store = users.get_store()
    u = store.create_user("frank")
    c = _client()
    _with_token(c, _login_user(c, "frank", u._token).json()["token"])
    r = c.get("/webui/api/users")
    assert r.status_code == 403
