"""No em dash (U+2014) in any committed text file - CLAUDE.md hard rule.
Uses `git ls-files` so this checks exactly what's committed, not the
working tree (which may have scratch files) and not files git itself
ignores (data/, mlruns/, etc.)."""

import subprocess
from pathlib import Path

import pytest

EM_DASH = chr(0x2014)  # built from the codepoint, not a literal character, so this file does not trip its own check
TEXT_EXTENSIONS = {".md", ".py", ".csv", ".json", ".txt", ".ipynb"}


def committed_text_files() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True, cwd=Path(__file__).resolve().parent.parent,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [f for f in result.stdout.splitlines() if Path(f).suffix in TEXT_EXTENSIONS]


FILES = committed_text_files()


@pytest.mark.skipif(not FILES, reason="not running inside a git checkout, or git is unavailable")
@pytest.mark.parametrize("filepath", FILES)
def test_no_em_dash(filepath):
    repo_root = Path(__file__).resolve().parent.parent
    full_path = repo_root / filepath
    try:
        content = full_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        pytest.skip(f"{filepath}: not readable as utf-8 text, or missing from working tree")
        return
    assert EM_DASH not in content, f"{filepath} contains an em dash (U+2014)"
