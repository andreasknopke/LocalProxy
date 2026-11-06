"""Benutzer- und Einstellungs-Verwaltung fuer LocalProxy.

Jeder Benutzer wird durch einen eigenen Bearer-Token identifiziert. Der Token
wird nur als SHA-256-Hash in SQLite abgelegt. Pro Benutzer koennen eigene
Einstellungen (Modelle, Co-Worker, Loop-Guard, Sampling, Prompt-Injection) als
JSON hinterlegt werden; fehlt ein Wert, wird auf die globale Baseline
(config.json / Module-Globals) zurueckgefallen.

Das Modul ist absichtlich frei von FastAPI-/proxy-Abhaengigkeiten, damit es
isoliert getestet werden kann.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Token-/Hash-Werkzeug ────────────────────────────────────────────────────

_TOKEN_PREFIX = "localfox-"


def generate_token() -> str:
    """Generiert einen neuen, eindeutigen Bearer-Token."""
    return _TOKEN_PREFIX + secrets.token_hex(16)


def hash_token(token: str) -> str:
    """SHA-256-Hex-Digest eines Tokens. Tokens werden nie Klartext gespeichert."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ── Datenmodell ─────────────────────────────────────────────────────────────

@dataclass
class User:
    id: int
    username: str
    is_admin: bool
    token_hash: str
    created_at: str
    settings: Optional[Dict[str, Any]] = field(default=None)

    def public(self) -> Dict[str, Any]:
        """Ohne Token-Hash — fuer Status-/Admin-Ausgaben."""
        return {
            "id": self.id,
            "username": self.username,
            "is_admin": self.is_admin,
            "created_at": self.created_at,
            "has_settings": self.settings is not None,
        }


# ── Schema ──────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    username     TEXT    NOT NULL UNIQUE,
    token_hash   TEXT    NOT NULL UNIQUE,
    is_admin     INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    settings_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_users_token ON users(token_hash);
"""


def _default_db_path() -> Path:
    env = os.getenv("LOCALPROXY_USERS_DB")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "data" / "users.db"


# ── Store ───────────────────────────────────────────────────────────────────

class UserStore:
    """SQLite-gestuetzter Benutzer-Store. Thread-sicher ueber eine Lock."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript(_SCHEMA)

    # -- Connection ---------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    # -- Zeilen -> User -----------------------------------------------------
    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> User:
        settings = None
        raw = row["settings_json"]
        if raw:
            try:
                settings = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                settings = None
        return User(
            id=row["id"],
            username=row["username"],
            is_admin=bool(row["is_admin"]),
            token_hash=row["token_hash"],
            created_at=row["created_at"],
            settings=settings,
        )

    # -- Erzeugen -----------------------------------------------------------
    def create_user(self, username: str, token: Optional[str] = None,
                    is_admin: bool = False,
                    settings: Optional[Dict[str, Any]] = None) -> User:
        """Anlegt einen Benutzer. `token` wird generiert, wenn None.
        `token_hash` muss eindeutig sein — ein bereits vergebener Token scheitert."""
        if token is None:
            token = generate_token()
        thash = hash_token(token)
        settings_json = json.dumps(settings) if settings is not None else None
        with self._lock, self._connect() as con:
            cur = con.execute(
                "INSERT INTO users (username, token_hash, is_admin, settings_json) "
                "VALUES (?, ?, ?, ?)",
                (username, thash, 1 if is_admin else 0, settings_json),
            )
            uid = cur.lastrowid
            row = con.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
        user = self._row_to_user(row)
        user._token = token  # nur fuer Rueckgabe an Aufrufer, nie gespeichert
        return user

    def ensure_admin(self, username: str, token: str) -> User:
        """Legt einen Admin an, falls es noch keinen gibt. Idempotent.
        Existiert bereits ein Admin, wird nichts geaendert und dieser zurueckgegeben."""
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT * FROM users WHERE is_admin = 1 ORDER BY id LIMIT 1"
            ).fetchone()
            if row:
                return self._row_to_user(row)
        return self.create_user(username, token=token, is_admin=True)

    # -- Lesen --------------------------------------------------------------
    def get_by_token(self, token: str) -> Optional[User]:
        if not token:
            return None
        thash = hash_token(token)
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM users WHERE token_hash = ?", (thash,)
            ).fetchone()
        return self._row_to_user(row) if row else None

    def get(self, user_id: int) -> Optional[User]:
        with self._connect() as con:
            row = con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._row_to_user(row) if row else None

    def get_by_username(self, username: str) -> Optional[User]:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
        return self._row_to_user(row) if row else None

    def list_users(self) -> List[User]:
        with self._connect() as con:
            rows = con.execute("SELECT * FROM users ORDER BY id").fetchall()
        return [self._row_to_user(r) for r in rows]

    # -- Schreiben ----------------------------------------------------------
    def set_settings(self, user_id: int, settings: Optional[Dict[str, Any]]) -> None:
        sj = json.dumps(settings) if settings is not None else None
        with self._lock, self._connect() as con:
            con.execute("UPDATE users SET settings_json = ? WHERE id = ?", (sj, user_id))

    def get_settings(self, user_id: int) -> Optional[Dict[str, Any]]:
        u = self.get(user_id)
        return u.settings if u else None

    def rotate_token(self, user_id: int) -> Optional[str]:
        """Neuer Token fuer den User. Gibt den Klartext-Token zurueck."""
        token = generate_token()
        with self._lock, self._connect() as con:
            cur = con.execute(
                "UPDATE users SET token_hash = ? WHERE id = ?",
                (hash_token(token), user_id),
            )
            if cur.rowcount == 0:
                return None
        return token

    def set_token(self, user_id: int, token: str) -> bool:
        """Setzt den Token eines Benutzers explizit (Heal/Reparatur)."""
        with self._lock, self._connect() as con:
            cur = con.execute(
                "UPDATE users SET token_hash = ? WHERE id = ?",
                (hash_token(token), user_id),
            )
            return cur.rowcount > 0

    def delete_user(self, user_id: int) -> bool:
        with self._lock, self._connect() as con:
            cur = con.execute("DELETE FROM users WHERE id = ?", (user_id,))
            return cur.rowcount > 0


# ── Modulweiter Singleton (vom Proxy genutzt) ───────────────────────────────

_STORE: Optional[UserStore] = None
_STORE_LOCK = threading.Lock()


def get_store(path: Optional[Path | str] = None) -> UserStore:
    """Singleton-Zugriff auf den Standard-Store. Pfad nur bei Erstaufruf relevant."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = UserStore(path or _default_db_path())
    return _STORE


def reset_store_for_tests(path: Path | str) -> UserStore:
    """Nur fuer Tests: setzt den Singleton zurueck und zeigt auf einen Testpfad."""
    global _STORE
    with _STORE_LOCK:
        _STORE = UserStore(path)
    return _STORE
