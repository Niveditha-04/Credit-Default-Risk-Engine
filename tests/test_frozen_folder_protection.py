"""Frozen-folder write protection (src.models.gbm). A locked file must
refuse a plain write, and promote_candidate_artifacts must only ever be
reachable when --promote was passed, so a run of gbm.py that is not
explicitly promoting a new champion can never overwrite the committed
frozen artifacts."""

import stat
import sys

import pytest

from src.models.gbm import (
    lock_frozen_file,
    new_run_id,
    promote_candidate_artifacts,
    promote_requested,
)


def test_lock_frozen_file_makes_it_read_only(tmp_path):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"fake model bytes")
    lock_frozen_file(path)
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o444, f"expected read-only 0o444, got {oct(mode)}"


def test_locked_file_rejects_a_plain_overwrite(tmp_path):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"original")
    lock_frozen_file(path)
    with pytest.raises(PermissionError):
        path.write_bytes(b"silently overwritten")
    assert path.read_bytes() == b"original"


def test_promote_requested_reads_argv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gbm.py"])
    assert promote_requested() is False
    monkeypatch.setattr(sys, "argv", ["gbm.py", "--promote"])
    assert promote_requested() is True


def test_promote_candidate_artifacts_copies_and_locks(tmp_path):
    candidate_dir = tmp_path / "candidates" / "run1" / "lightgbm"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "model_chosen.joblib").write_bytes(b"candidate model v2")

    frozen_dir = tmp_path / "frozen" / "lightgbm"
    frozen_dir.mkdir(parents=True)
    old_frozen_file = frozen_dir / "model_chosen.joblib"
    old_frozen_file.write_bytes(b"old frozen model v1")
    lock_frozen_file(old_frozen_file)

    promote_candidate_artifacts(candidate_dir, frozen_dir)

    promoted = frozen_dir / "model_chosen.joblib"
    assert promoted.read_bytes() == b"candidate model v2"
    mode = stat.S_IMODE(promoted.stat().st_mode)
    assert mode == 0o444, "promoted file must end up read-only too"


def test_new_run_id_is_a_sortable_timestamp_safe_for_a_directory_name():
    import re

    run_id = new_run_id()
    assert re.fullmatch(r"\d{8}T\d{6}", run_id), f"expected YYYYMMDDTHHMMSS, got {run_id!r}"
    assert "/" not in run_id and " " not in run_id
