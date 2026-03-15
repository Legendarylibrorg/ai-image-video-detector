from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from typing import Optional

import piexif
from PIL import Image


def _load_exif_dict(image_path: str) -> dict:
    img = Image.open(image_path)
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
    exif_dict = _load_exif_dict(image_path)
    fields = _extract_fields(exif_dict)

    score = 0.0
    flags: list[str] = []

    has_any_exif = any(bool(exif_dict.get(k)) for k in ("0th", "Exif", "GPS", "Interop", "1st"))
    if not has_any_exif:
        score += 0.30
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
        score += 0.10
        flags.append("missing_capture_time")

    if "camera_make" not in fields and "camera_model" not in fields:
        score += 0.15
        flags.append("missing_camera_id")

    score = min(1.0, score)
    return {"metadata_score": score, "metadata_flags": flags, "metadata_fields": fields}


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
