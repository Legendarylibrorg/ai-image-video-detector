from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any
from typing import Optional

import piexif
from PIL import Image


METADATA_FEATURE_NAMES = (
    "metadata_score",
    "has_exif",
    "has_software_tag",
    "has_suspicious_software_marker",
    "has_generation_prompt_trace",
    "has_capture_time",
    "has_camera_id",
    "has_gps",
    "is_jpeg_like",
    "log_file_size_norm",
    "aspect_ratio_norm",
    "megapixels_norm",
)

WEB_EXPORT_FORMATS = {"WEBP", "PNG", "GIF"}


def _load_exif_dict(image_path: str) -> dict:
    with Image.open(image_path) as img:
        exif_bytes = img.info.get("exif", b"")
    if exif_bytes:
        return piexif.load(exif_bytes)
    return {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}


def _decode_exif_value(value: Any) -> str:
    if isinstance(value, bytes):
        # EXIF strings are often bytes and may include ASCII prefix markers.
        return value.decode("utf-8", errors="ignore").replace("\x00", " ").strip()
    return str(value)


def _extract_fields(exif_dict: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    ifd0 = exif_dict.get("0th", {})
    ifd_exif = exif_dict.get("Exif", {})

    software = ifd0.get(piexif.ImageIFD.Software)
    artist = ifd0.get(piexif.ImageIFD.Artist)
    datetime = ifd0.get(piexif.ImageIFD.DateTime)
    user_comment = ifd_exif.get(piexif.ExifIFD.UserComment)
    model = ifd0.get(piexif.ImageIFD.Model)
    make = ifd0.get(piexif.ImageIFD.Make)

    if software is not None:
        out["software"] = _decode_exif_value(software)
    if artist is not None:
        out["artist"] = _decode_exif_value(artist)
    if datetime is not None:
        out["datetime"] = _decode_exif_value(datetime)
    if user_comment is not None:
        out["user_comment"] = _decode_exif_value(user_comment)
    if model is not None:
        out["camera_model"] = _decode_exif_value(model)
    if make is not None:
        out["camera_make"] = _decode_exif_value(make)
    return out


def analyze_metadata(image_path: str) -> dict[str, Any]:
    with Image.open(image_path) as image:
        image_format = (image.format or "").upper()
    exif_dict = _load_exif_dict(image_path)
    fields = _extract_fields(exif_dict)
    if image_format:
        fields.setdefault("file_format", image_format.lower())

    score = 0.0
    flags: list[str] = []
    is_web_export_format = image_format in WEB_EXPORT_FORMATS

    has_any_exif = any(bool(exif_dict.get(k)) for k in ("0th", "Exif", "GPS", "Interop", "1st"))
    if not has_any_exif:
        score += 0.08 if is_web_export_format else 0.22
        flags.append("missing_exif")

    software = fields.get("software", "").lower()
    user_comment = fields.get("user_comment", "").lower()

    if software:
        score += 0.10
        flags.append("edited_with_software_tag")

    suspicious_software_markers = (
        "stable diffusion",
        "midjourney",
        "dall",
        "comfyui",
        "automatic1111",
        "invokeai",
        "adobe firefly",
        "novelai",
        "leonardo",
    )
    if any(marker in software for marker in suspicious_software_markers):
        score += 0.35
        flags.append("synthetic_software_marker")

    suspicious_comment_markers = (
        "steps:",
        "sampler:",
        "cfg scale",
        "negative prompt",
        "seed:",
        "model hash",
        "prompt:",
    )
    if any(marker in user_comment for marker in suspicious_comment_markers):
        score += 0.35
        flags.append("generation_prompt_trace")

    if "datetime" not in fields:
        score += 0.02 if is_web_export_format else 0.06
        flags.append("missing_capture_time")

    if "camera_make" not in fields and "camera_model" not in fields:
        score += 0.03 if is_web_export_format else 0.08
        flags.append("missing_camera_id")

    score = min(1.0, score)
    return {"metadata_score": score, "metadata_flags": flags, "metadata_fields": fields}


def extract_metadata_features(image_path: str) -> list[float]:
    with Image.open(image_path) as image:
        width, height = image.size
        image_format = (image.format or "").upper()
    exif_dict = _load_exif_dict(image_path)
    fields = _extract_fields(exif_dict)
    analysis = analyze_metadata(image_path)

    software = fields.get("software", "").lower()
    user_comment = fields.get("user_comment", "").lower()
    has_any_exif = any(bool(exif_dict.get(k)) for k in ("0th", "Exif", "GPS", "Interop", "1st"))
    suspicious_software_markers = (
        "stable diffusion",
        "midjourney",
        "dall",
        "comfyui",
        "automatic1111",
        "invokeai",
        "adobe firefly",
        "novelai",
        "leonardo",
    )
    suspicious_comment_markers = (
        "steps:",
        "sampler:",
        "cfg scale",
        "negative prompt",
        "seed:",
        "model hash",
        "prompt:",
    )
    file_size = Path(image_path).stat().st_size if Path(image_path).exists() else 0
    aspect_ratio = width / max(float(height), 1.0)
    aspect_ratio_norm = min(abs(math.log(max(aspect_ratio, 1e-6))), math.log(4.0)) / math.log(4.0)
    megapixels = (width * height) / 1_000_000.0

    return [
        float(analysis["metadata_score"]),
        1.0 if has_any_exif else 0.0,
        1.0 if software else 0.0,
        1.0 if any(marker in software for marker in suspicious_software_markers) else 0.0,
        1.0 if any(marker in user_comment for marker in suspicious_comment_markers) else 0.0,
        1.0 if "datetime" in fields else 0.0,
        1.0 if ("camera_make" in fields or "camera_model" in fields) else 0.0,
        1.0 if bool(exif_dict.get("GPS")) else 0.0,
        1.0 if image_format in {"JPEG", "JPG"} else 0.0,
        min(math.log1p(float(file_size)) / 20.0, 1.0),
        float(aspect_ratio_norm),
        min(megapixels / 12.0, 1.0),
    ]


def metadata_feature_dim() -> int:
    return len(METADATA_FEATURE_NAMES)


def inspect_metadata(image_path: str) -> None:
    exif_dict = _load_exif_dict(image_path)

    print(f"image={image_path}")
    for ifd in ("0th", "Exif", "GPS", "Interop", "1st"):
        if not exif_dict.get(ifd):
            continue
        print(f"[{ifd}]")
        for tag, value in exif_dict[ifd].items():
            name = piexif.TAGS[ifd][tag]["name"] if tag in piexif.TAGS[ifd] else str(tag)
            print(f"  {name}: {value}")


def strip_metadata(input_path: str, output_path: str) -> None:
    img = Image.open(input_path).convert("RGB")
    img.save(output_path, quality=95)


def modify_metadata(
    input_path: str,
    output_path: str,
    software: Optional[str],
    artist: Optional[str],
    user_comment: Optional[str],
) -> None:
    img = Image.open(input_path).convert("RGB")
    exif_dict = _load_exif_dict(input_path)

    if software is not None:
        exif_dict["0th"][piexif.ImageIFD.Software] = software.encode("utf-8", errors="ignore")
    if artist is not None:
        exif_dict["0th"][piexif.ImageIFD.Artist] = artist.encode("utf-8", errors="ignore")
    if user_comment is not None:
        prefix = b"ASCII\x00\x00\x00"
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = prefix + user_comment.encode("ascii", errors="ignore")

    exif_bytes = piexif.dump(exif_dict)
    img.save(output_path, exif=exif_bytes, quality=95)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect or modify image EXIF metadata")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="Print EXIF metadata")
    p_inspect.add_argument("--image", required=True)

    p_strip = sub.add_parser("strip", help="Remove EXIF metadata")
    p_strip.add_argument("--input", required=True)
    p_strip.add_argument("--output", required=True)

    p_mod = sub.add_parser("modify", help="Modify EXIF metadata fields")
    p_mod.add_argument("--input", required=True)
    p_mod.add_argument("--output", required=True)
    p_mod.add_argument("--software")
    p_mod.add_argument("--artist")
    p_mod.add_argument("--comment")

    args = ap.parse_args()

    if args.cmd == "inspect":
        inspect_metadata(args.image)
        analysis = analyze_metadata(args.image)
        print(f"[analysis] metadata_score={analysis['metadata_score']:.3f} flags={analysis['metadata_flags']}")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.cmd == "strip":
        strip_metadata(args.input, args.output)
        print(f"saved stripped image to {args.output}")
        return

    if args.cmd == "modify":
        modify_metadata(args.input, args.output, args.software, args.artist, args.comment)
        print(f"saved modified image to {args.output}")
        return


if __name__ == "__main__":
    main()
