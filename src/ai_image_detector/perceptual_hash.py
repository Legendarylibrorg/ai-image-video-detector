from __future__ import annotations

from pathlib import Path

from .io_limits import MAX_IMAGE_FILE_BYTES, check_file_size, configure_pil_limits


def dhash_hex(img) -> str:
    from PIL import Image

    g = img.convert("L").resize((9, 8), Image.BILINEAR)
    px = list(g.tobytes())
    bits: list[str] = []
    for y in range(8):
        row = px[y * 9 : (y + 1) * 9]
        for x in range(8):
            bits.append("1" if row[x] > row[x + 1] else "0")
    return f"{int(''.join(bits), 2):016x}"


def hamming_hex(a: str, b: str) -> int:
    if not a or not b:
        return 64
    return (int(a, 16) ^ int(b, 16)).bit_count()


def dhash_path(path: Path) -> str:
    configure_pil_limits()
    try:
        check_file_size(path, max_bytes=MAX_IMAGE_FILE_BYTES)
    except (OSError, ValueError):
        return ""
    try:
        from PIL import Image

        with Image.open(path) as img:
            return dhash_hex(img)
    except OSError:
        return ""
