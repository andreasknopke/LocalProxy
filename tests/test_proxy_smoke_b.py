"""Smoke tests (part B) for the LocalProxy project.

Lightweight, self-contained checks that verify basic repo structure
and documentation sanity.
"""

import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent


def test_requirements_txt_exists():
    """requirements.txt must exist at the repo root and must not be empty."""
    req = REPO_ROOT / "requirements.txt"
    assert req.is_file(), f"requirements.txt not found at {req}"
    content = req.read_text(encoding="utf-8").strip()
    assert len(content) > 0, "requirements.txt is empty"


def test_readme_mentions_coworker():
    """README.md must contain the string 'coworker' (case-insensitive)."""
    readme = REPO_ROOT / "README.md"
    assert readme.is_file(), f"README.md not found at {readme}"
    content = readme.read_text(encoding="utf-8").lower()
    assert "coworker" in content, "README.md does not mention 'coworker'"


def test_io_trace_dir_structure():
    """data/io_traces/ must exist; any subdirectory must contain
    both events.jsonl and a valid meta.json."""
    io_traces = REPO_ROOT / "data" / "io_traces"
    assert io_traces.is_dir(), f"data/io_traces/ not found at {io_traces}"

    entries = list(io_traces.iterdir())
    if not entries:
        # Empty directory is acceptable
        return

    for entry in entries:
        if not entry.is_dir():
            continue
        events = entry / "events.jsonl"
        meta = entry / "meta.json"
        assert events.is_file(), f"{entry.name}/events.jsonl is missing"
        assert meta.is_file(), f"{entry.name}/meta.json is missing"
        with open(meta, encoding="utf-8") as f:
            json.load(f)
