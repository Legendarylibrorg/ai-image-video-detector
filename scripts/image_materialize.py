from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Iterable

from PIL import Image

from ai_image_detector.io_limits import open_image_rgb


@dataclass(frozen=True)
class ImageQualityPolicy:
    min_side: int
    max_aspect_ratio: float
    min_entropy: float


class ImageDeduper:
    def __init__(self, seen_exact: set[str] | None = None, seen_dhash_by_cls: dict[str, list[str]] | None = None):
        self.seen_exact = seen_exact or set()
        self.seen_dhash_by_cls = seen_dhash_by_cls or defaultdict(list)

    @classmethod
    def from_output(
        cls,
        out: Path,
        *,
        splits: Iterable[str],
        classes: Iterable[str],
    ) -> "ImageDeduper":
        seen_exact: set[str] = set()
        seen_dhash_by_cls: dict[str, list[str]] = defaultdict(list)
        for split in splits:
            for label in classes:
                split_dir = out / split / label
                if not split_dir.exists():
                    continue
                for path in split_dir.glob("*.jpg"):
                    if path.name.startswith("hardneg="):
                        continue
                    img = open_local_image(path)
                    if img is None:
                        continue
                    seen_exact.add(hash_img_bytes(img))
                    seen_dhash_by_cls[label].append(dhash_hex(img))
        return cls(seen_exact=seen_exact, seen_dhash_by_cls=seen_dhash_by_cls)

    def duplicate_reason(
        self,
        img: Image.Image,
        *,
        cls: str,
        near_hamming: int,
        near_window: int,
    ) -> str | None:
        img_hash = hash_img_bytes(img)
        if img_hash in self.seen_exact:
            return "dup_exact"

        img_dhash = dhash_hex(img)
        prevs = self.seen_dhash_by_cls.get(cls, [])
        if near_window > 0:
            start = max(0, len(prevs) - near_window)
            for prev in prevs[start:]:
                if hamming_hex(img_dhash, prev) <= int(near_hamming):
                    return "dup_near"
        return None

    def remember(self, img: Image.Image, *, cls: str) -> None:
        self.seen_exact.add(hash_img_bytes(img))
        self.seen_dhash_by_cls.setdefault(cls, []).append(dhash_hex(img))


def hash_img_bytes(img: Image.Image) -> str:
    return hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()


def dhash_hex(img: Image.Image) -> str:
    g = img.convert("L").resize((9, 8), Image.BILINEAR)
    px = list(g.tobytes())
    bits: list[str] = []
    for y in range(8):
        row = px[y * 9 : (y + 1) * 9]
        for x in range(8):
            bits.append("1" if row[x] > row[x + 1] else "0")
    return f"{int(''.join(bits), 2):016x}"


def hamming_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def image_entropy_bits(img: Image.Image) -> float:
    hist = img.convert("L").histogram()
    total = float(sum(hist))
    if total <= 0:
        return 0.0
    ent = 0.0
    for n in hist:
        if n <= 0:
            continue
        p = float(n) / total
        ent -= p * math.log2(p)
    return ent


def passes_quality_filters(img: Image.Image, policy: ImageQualityPolicy) -> tuple[bool, str]:
    w, h = img.size
    if min(w, h) < int(policy.min_side):
        return False, "too_small"
    aspect = float(max(w, h)) / float(max(1, min(w, h)))
    if aspect > float(policy.max_aspect_ratio):
        return False, "bad_aspect"
    ent = image_entropy_bits(img)
    if ent < float(policy.min_entropy):
        return False, "low_entropy"
    return True, "ok"


def open_example_image(example: dict, image_field: str) -> Image.Image | None:
    img = example.get(image_field)
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    try:
        return Image.fromarray(img).convert("RGB")
    except Exception:
        return None


def open_local_image(path: Path) -> Image.Image | None:
    try:
        return open_image_rgb(path)
    except Exception:
        return None


def save_img(img: Image.Image, path: Path, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, quality=quality)
