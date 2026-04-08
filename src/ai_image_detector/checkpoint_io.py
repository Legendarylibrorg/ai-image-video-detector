from __future__ import annotations

"""Safe checkpoint file reads: staging with ``O_NOFOLLOW`` (or fallback) before parse."""

import errno
import os
import shutil
import tempfile
from pathlib import Path

from .io_limits import check_file_size, reject_symlink


def checkpoint_load_staging_enabled() -> bool:
    """When true, checkpoint loads copy via ``O_NOFOLLOW`` (or a safe fallback) to reduce TOCTOU."""
    v = (os.environ.get("AID_CHECKPOINT_LOAD_STAGING") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def materialize_checkpoint_file(path: str | Path, *, max_bytes: int) -> Path:
    """Open the file without following a symlink leaf, then copy exactly ``st_size`` bytes to a temp path.

    Caller must ``unlink`` the returned path when finished. Falls back to ``reject_symlink`` +
    ``shutil.copyfile`` when ``O_NOFOLLOW`` is unavailable (e.g. some Windows builds).
    """
    p = Path(path)
    suffix = p.suffix or ".bin"
    fd_out, tmp_name = tempfile.mkstemp(prefix="aid-ckpt-", suffix=suffix)
    os.close(fd_out)
    tmp_path = Path(tmp_name)
    try:
        if hasattr(os, "chmod"):
            try:
                os.chmod(tmp_path, 0o600)
            except OSError:
                pass
        if not hasattr(os, "O_NOFOLLOW"):
            reject_symlink(p)
            check_file_size(p, max_bytes=max_bytes)
            shutil.copyfile(p, tmp_path, follow_symlinks=False)
            st = tmp_path.stat()
            if st.st_size > max_bytes:
                raise ValueError(f"file_too_large path={tmp_path} size={st.st_size} max={max_bytes}")
            return tmp_path

        try:
            fd_in = os.open(str(p), os.O_RDONLY | os.O_NOFOLLOW)
        except OSError as exc:
            # Linux: ELOOP. Some platforms report EINVAL for O_NOFOLLOW on a symlink leaf.
            if exc.errno == errno.ELOOP or (exc.errno == errno.EINVAL and p.is_symlink()):
                raise ValueError(f"symlink_not_allowed path={p}") from exc
            raise
        try:
            st = os.fstat(fd_in)
            size = int(st.st_size)
            if size > max_bytes:
                raise ValueError(f"file_too_large path={p} size={size} max={max_bytes}")
            remaining = size
            with open(tmp_path, "wb") as out_f:
                while remaining > 0:
                    chunk = os.read(fd_in, min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError(f"checkpoint_short_read path={p}")
                    out_f.write(chunk)
                    remaining -= len(chunk)
            if tmp_path.stat().st_size != size:
                raise ValueError("checkpoint_size_mismatch_after_stage")
            return tmp_path
        finally:
            os.close(fd_in)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
