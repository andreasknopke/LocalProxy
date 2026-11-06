"""Tests fuer das users.py Modul (DB-Schicht)."""
import json
import tempfile
from pathlib import Path

import pytest

import users


@pytest.fixture
def store(tmp_path):
    return users.UserStore(tmp_path / "users.db")


def test_generate_token_format():
    t = users.generate_token()
    assert t.startswith("localfox-")
    assert len(t) > 20
    assert users.generate_token() != users.generate_token()


def test_hash_token_stable_and_hex():
    h = users.hash_token("abc")
    assert h == users.hash_token("abc")
    assert h != users.hash_token("abd")
    assert len(h) == 64
    int(h, 16)  # valid hex


def test_create_and_get_by_token(store):
    u = store.create_user("alice")
    assert u.id > 0
    assert u.username == "alice"
    assert u.is_admin is False
    # Token liegt nicht im Objekt (nur temporaer)
    assert store.get_by_token(u._token) is not None
    assert store.get_by_token(u._token).id == u.id


def test_token_stored_hashed_not_plaintext(store):
    u = store.create_user("bob", token="localfox-deadbeef")
    # Im rohen DB soll der Klartext nicht stehen
    raw = store.path.read_bytes()
    assert b"deadbeef" not in raw
    assert users.hash_token("localfox-deadbeef").encode() in raw


def test_get_by_token_unknown(store):
    assert store.get_by_token("nope") is None
    assert store.get_by_token("") is None


def test_unique_username(store):
    store.create_user("carol")
    with pytest.raises(Exception):
        store.create_user("carol")


def test_unique_token(store):
    store.create_user("dave", token="localfox-same")
    with pytest.raises(Exception):
        store.create_user("erin", token="localfox-same")


def test_ensure_admin_idempotent(store):
    a1 = store.ensure_admin("admin", "localfox-admin1")
    assert a1.is_admin is True
    # Zweiter Aufruf legt keinen zweiten Admin an, gibt bestehenden zurueck
    a2 = store.ensure_admin("admin2", "localfox-admin2")
    assert a2.id == a1.id
    assert len(store.list_users()) == 1


def test_list_users(store):
    store.create_user("x")
    store.create_user("y")
    names = [u.username for u in store.list_users()]
    assert names == ["x", "y"]


def test_settings_roundtrip(store):
    u = store.create_user("f", settings={"default_category": "strong"})
    assert store.get_settings(u.id) == {"default_category": "strong"}
    store.set_settings(u.id, {"tokens": {"loop_guard_enabled": False}})
    assert store.get_settings(u.id) == {"tokens": {"loop_guard_enabled": False}}
    store.set_settings(u.id, None)
    assert store.get_settings(u.id) is None


def test_rotate_token(store):
    u = store.create_user("g", token="localfox-old")
    assert store.get_by_token("localfox-old") is not None
    new = store.rotate_token(u.id)
    assert new and new != "localfox-old"
    assert store.get_by_token("localfox-old") is None
    assert store.get_by_token(new) is not None


def test_delete_user(store):
    u = store.create_user("h")
    assert store.delete_user(u.id) is True
    assert store.get(u.id) is None
    assert store.delete_user(u.id) is False


def test_get_by_username(store):
    store.create_user("findme")
    assert store.get_by_username("findme") is not None
    assert store.get_by_username("ghost") is None


def test_persistence_across_reopen(store, tmp_path):
    u = store.create_user("persist", token="localfox-persist")
    store2 = users.UserStore(tmp_path / "users.db")
    assert store2.get_by_token("localfox-persist") is not None
    assert store2.get_by_token("localfox-persist").username == "persist"
