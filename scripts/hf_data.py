from __future__ import annotations

import json
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    from datasets import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars, load_dataset
except ImportError:  # pragma: no cover - optional dependency path
    load_dataset = None  # type: ignore[assignment]

    def are_progress_bars_disabled() -> bool:  # type: ignore[override]
        return True

    def disable_progress_bars() -> None:  # type: ignore[override]
        return None

    def enable_progress_bars() -> None:  # type: ignore[override]
        return None


PREFERRED_SPLITS = ("train", "validation", "test")

_MAX_HF_SOURCE_ID_LEN = 256
HF_DATASET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

# Hub dataset search strings (``HfApi.list_datasets(search=...)``).
_MAX_HF_DISCOVERY_QUERY_CHARS = int(os.environ.get("AID_MAX_HF_DISCOVERY_QUERY_CHARS", "512"))

# JSONL dataset source manifest (append-only under ``.local`` / ``out``).
_MAX_SOURCE_MANIFEST_BYTES = int(os.environ.get("AID_MAX_SOURCE_MANIFEST_BYTES", str(64 * 1024 * 1024)))


def validate_hf_repo_blob_path(filename: str) -> str:
    """Reject path traversal in Hub ``filename`` / relative paths passed to ``hf_hub_download``."""
    fn = str(filename).strip().replace("\\", "/")
    if not fn or fn.startswith("/"):
        raise ValueError(f"invalid_hf_repo_filename={filename!r}")
    parts = fn.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"invalid_hf_repo_filename={filename!r}")
    return fn


def validate_hf_dataset_source_id(source_id: str) -> str:
    """Reject malformed or path-like strings before they reach Hub APIs."""
    s = str(source_id).strip()
    if not s or len(s) > _MAX_HF_SOURCE_ID_LEN:
        raise ValueError(f"invalid_hf_dataset_id length={len(s)}")
    if "\n" in s or "\r" in s or "\x00" in s:
        raise ValueError("invalid_hf_dataset_id control_chars")
    if ".." in s or s.startswith("/") or "\\" in s:
        raise ValueError(f"invalid_hf_dataset_id path_tokens source_id={s!r}")
    if not HF_DATASET_ID_RE.match(s):
        raise ValueError(f"invalid_hf_dataset_id format={s!r}")
    return s


def validate_hf_discovery_query(query: str) -> str:
    """Reject control characters and oversized strings before Hub discovery APIs.

    Discovery queries come from CLI flags and environment-driven CSV; this bounds
    attacker-controlled growth and keeps newlines out of Hub search parameters.
    """
    q = str(query).strip()
    if not q:
        raise ValueError("empty_hf_discovery_query")
    if len(q) > _MAX_HF_DISCOVERY_QUERY_CHARS:
        raise ValueError(f"hf_discovery_query_too_long len={len(q)} max={_MAX_HF_DISCOVERY_QUERY_CHARS}")
    if any(ch in q for ch in ("\n", "\r", "\x00")):
        raise ValueError("hf_discovery_query_illegal_control_char")
    return q


def _hf_trust_allowlist() -> frozenset[str] | None:
    """When set, only these repo ids may use ``trust_remote_code`` (requires trust env + risk accept)."""
    raw = (os.environ.get("AID_HF_TRUST_REMOTE_ALLOWLIST") or "").strip()
    if not raw:
        return None
    out = {validate_hf_dataset_source_id(x) for x in raw.split(",") if x.strip()}
    return frozenset(out)


def _hf_trust_remote_code_from_env() -> bool:
    """When false (default), Hugging Face ``datasets`` will not run custom Hub loading scripts."""
    v = (os.environ.get("AID_HF_TRUST_REMOTE_CODE") or "").strip().lower()
    return v in ("1", "true", "yes")


def _hf_accept_trust_remote_risk_from_env() -> bool:
    """Explicit acknowledgement before Hub ``trust_remote_code=True`` (any dataset).

    Required for **both** allowlisted datasets and legacy global trust so
    ``AID_HF_TRUST_REMOTE_CODE=1`` alone cannot enable remote loading scripts.
    """
    for key in ("AID_ACCEPT_HF_TRUST_REMOTE_RISK", "I_ACCEPT_HF_TRUST_RISK"):
        v = (os.environ.get(key) or "").strip().lower()
        if v in ("1", "true", "yes"):
            return True
    return False


def _hf_trust_remote_unsafe_global_raw_from_env() -> bool:
    """Legacy: opt in to ``trust_remote_code`` for **every** dataset (still needs accept; see load path)."""
    v = (os.environ.get("AID_HF_TRUST_REMOTE_UNSAFE_GLOBAL") or "").strip().lower()
    return v in ("1", "true", "yes")


@dataclass(frozen=True)
class LoadedDatasetSource:
    source_id: str
    split_name: str
    split: object
    streaming: bool


def read_noncomment_lines(path: Path) -> list[str]:
    """Read ``#``-stripped non-empty lines with byte/line caps (same env as ``io_limits`` line lists)."""
    from ai_image_detector.io_limits import (
        MAX_NONEMPTY_LINES_COUNT,
        MAX_NONEMPTY_LINES_FILE_BYTES,
        read_bytes_limited,
        reject_symlink,
    )

    items: list[str] = []
    if not path.exists():
        return items
    p = Path(path)
    reject_symlink(p)
    raw = read_bytes_limited(p, max_bytes=MAX_NONEMPTY_LINES_FILE_BYTES)
    text = raw.decode("utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(line)
        if len(items) > MAX_NONEMPTY_LINES_COUNT:
            raise ValueError(
                f"sources_list_too_many_lines path={p} max_lines={MAX_NONEMPTY_LINES_COUNT}"
            )
    return items


def write_noncomment_lines(path: Path, items: Sequence[str]) -> None:
    """Write sources list; path must stay under the collection workspace (same anchor as reads)."""
    from ai_image_detector.collection_paths import collection_workspace_root, require_under_collection_workspace
    from ai_image_detector.io_limits import reject_symlink

    p = Path(path).expanduser()
    w = collection_workspace_root()
    require_under_collection_workspace(p, w)
    if p.exists():
        reject_symlink(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(item).strip() for item in items if str(item).strip())
    p.write_text(text + ("\n" if text else ""), encoding="utf-8")


def normalize_hf_token(token: str | None) -> str | None:
    cleaned = (token or "").strip()
    return cleaned or None


def resolve_hf_token_value(token_env: str = "HF_TOKEN") -> tuple[str | None, str]:
    env_names = [token_env, "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"]
    seen: set[str] = set()
    for name in env_names:
        normalized_name = str(name or "").strip()
        if not normalized_name or normalized_name in seen:
            continue
        seen.add(normalized_name)
        token = normalize_hf_token(os.environ.get(normalized_name))
        if token:
            return token, f"env:{normalized_name}"
    try:
        from huggingface_hub import get_token

        token = normalize_hf_token(get_token())
        if token:
            return token, "hf_auth_login"
    except ImportError:
        pass
    return None, "missing"


def load_hf_dataset_source(
    source_id: str,
    *,
    token: str | None = None,
    streaming: bool = True,
    cache_dir: str | None = None,
    preferred_splits: Sequence[str] = PREFERRED_SPLITS,
) -> LoadedDatasetSource:
    if load_dataset is None:
        raise RuntimeError("datasets package is required to load Hugging Face datasets")
    source_id = validate_hf_dataset_source_id(source_id)
    token = normalize_hf_token(token)
    base_kw: dict[str, object] = {
        "streaming": bool(streaming),
        "cache_dir": cache_dir or None,
    }
    if token:
        base_kw["token"] = token
    trust_env = _hf_trust_remote_code_from_env()
    allow = _hf_trust_allowlist()
    unsafe_global = _hf_trust_remote_unsafe_global_raw_from_env()
    accept = _hf_accept_trust_remote_risk_from_env()
    if not trust_env:
        trust_requested = False
    elif not accept:
        trust_requested = False
    elif unsafe_global:
        trust_requested = True
    elif allow is not None:
        trust_requested = source_id in allow
    else:
        trust_requested = False
    if trust_requested:
        mode = "unsafe_global" if unsafe_global else "allowlist"
        print(
            f"notice_hf_trust_remote_code source_id={source_id} mode={mode}",
            file=sys.stderr,
            flush=True,
        )
    attempts: list[dict[str, object]] = []
    for use_token in (True, False):
        for pass_trust_flag in (True, False):
            kw = dict(base_kw)
            if not use_token:
                kw.pop("token", None)
            if pass_trust_flag:
                kw["trust_remote_code"] = trust_requested
            attempts.append(kw)

    last_type_error: TypeError | None = None
    ds = None
    for kw in attempts:
        try:
            ds = load_dataset(source_id, **kw)
            break
        except TypeError as exc:
            last_type_error = exc
            continue
        except Exception as exc:
            raise RuntimeError(f"failed to load dataset {source_id}: {exc}") from exc

    if ds is None:
        raise RuntimeError(
            f"failed to load dataset {source_id}: incompatible load_dataset() arguments"
        ) from last_type_error

    split_name = next((name for name in preferred_splits if name in ds), None)
    if split_name is None:
        keys = list(ds.keys())
        if not keys:
            raise RuntimeError(f"dataset {source_id} returned no splits")
        split_name = str(keys[0])
    return LoadedDatasetSource(
        source_id=source_id,
        split_name=split_name,
        split=ds[split_name],
        streaming=bool(streaming),
    )


def iter_source_examples(
    source: LoadedDatasetSource,
    *,
    seed: int,
    shuffle_buffer_size: int,
    max_samples: int,
) -> Iterator[object]:
    limit = max(1, int(max_samples))
    if source.streaming:
        split = source.split
        try:
            split = split.shuffle(seed=seed, buffer_size=max(500, int(shuffle_buffer_size)))
        except (AttributeError, TypeError, ValueError):
            pass
        yield from split.take(limit)
        return

    split = source.split
    indices = list(range(len(split)))
    import random

    random.Random(seed).shuffle(indices)
    for idx in indices[:limit]:
        yield split[idx]


def normalize_image_dataset_split(
    split,
    *,
    label_field: str,
    resolve_label,
    label_column: str = "_normalized_label",
    batch_size: int = 128,
    show_progress: bool = False,
):
    keep_labels = {"ai", "real"}

    def normalize_batch(batch):
        return {label_column: [resolve_label(value) for value in batch[label_field]]}

    with datasets_progress_bars(enabled=show_progress):
        normalized = split.map(
            normalize_batch,
            batched=True,
            batch_size=max(1, int(batch_size)),
        )
        return normalized.filter(lambda ex: ex[label_column] in keep_labels)


@contextmanager
def datasets_progress_bars(enabled: bool):
    was_disabled = are_progress_bars_disabled()
    if enabled:
        enable_progress_bars()
    else:
        disable_progress_bars()
    try:
        yield
    finally:
        if was_disabled:
            disable_progress_bars()
        else:
            enable_progress_bars()


def unique_preserve(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def snapshot_dataset_repo(
    repo_id: str,
    *,
    allow_patterns: Sequence[str] | None = None,
    token: str | None = None,
    cache_dir: str | None = None,
    max_workers: int = 4,
) -> str:
    from huggingface_hub import snapshot_download

    validate_hf_dataset_source_id(repo_id)

    return snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=normalize_hf_token(token),
        allow_patterns=list(allow_patterns or []),
        max_workers=max(1, int(max_workers)),
        cache_dir=cache_dir or None,
    )


def list_dataset_repo_files(repo_id: str, *, token: str | None = None) -> list[str]:
    from huggingface_hub import list_repo_files

    validate_hf_dataset_source_id(repo_id)

    return list_repo_files(repo_id, repo_type="dataset", token=normalize_hf_token(token))


def download_dataset_file(
    repo_id: str,
    filename: str,
    *,
    token: str | None = None,
    cache_dir: str | None = None,
) -> str:
    from huggingface_hub import hf_hub_download

    validate_hf_dataset_source_id(repo_id)
    fn = validate_hf_repo_blob_path(filename)

    return hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=fn,
        token=normalize_hf_token(token),
        cache_dir=cache_dir or None,
    )


def load_latest_source_manifest(path: Path) -> dict[str, dict]:
    """Parse JSONL manifest with a bounded read (prefix only if file exceeds cap)."""
    from ai_image_detector.io_limits import read_bytes_limited, reject_symlink

    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    p = Path(path)
    reject_symlink(p)
    raw = read_bytes_limited(p, max_bytes=_MAX_SOURCE_MANIFEST_BYTES)
    text = raw.decode("utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        source = str(entry.get("source", "")).strip()
        if source:
            latest[source] = entry
    return latest


def append_source_manifest_entry(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
