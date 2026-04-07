from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    from datasets import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars, load_dataset
except Exception:  # pragma: no cover - optional dependency path
    load_dataset = None  # type: ignore[assignment]

    def are_progress_bars_disabled() -> bool:  # type: ignore[override]
        return True

    def disable_progress_bars() -> None:  # type: ignore[override]
        return None

    def enable_progress_bars() -> None:  # type: ignore[override]
        return None


PREFERRED_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class LoadedDatasetSource:
    source_id: str
    split_name: str
    split: object
    streaming: bool


def read_noncomment_lines(path: Path) -> list[str]:
    items: list[str] = []
    if not path.exists():
        return items
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(line)
    return items


def write_noncomment_lines(path: Path, items: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(item).strip() for item in items if str(item).strip())
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


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
    except Exception:
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
    token = normalize_hf_token(token)
    kwargs = {
        "streaming": bool(streaming),
        "cache_dir": cache_dir or None,
    }
    if token:
        kwargs["token"] = token
    try:
        ds = load_dataset(source_id, **kwargs)
    except TypeError:
        kwargs.pop("token", None)
        ds = load_dataset(source_id, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"failed to load dataset {source_id}: {exc}") from exc

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
        except Exception:
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

    return list_repo_files(repo_id, repo_type="dataset", token=normalize_hf_token(token))


def download_dataset_file(
    repo_id: str,
    filename: str,
    *,
    token: str | None = None,
    cache_dir: str | None = None,
) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        token=normalize_hf_token(token),
        cache_dir=cache_dir or None,
    )


def load_latest_source_manifest(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
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
