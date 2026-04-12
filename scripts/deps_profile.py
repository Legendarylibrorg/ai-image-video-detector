#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json


PIPELINE_EXTRAS = ("inference", "training", "collection", "video")

# Must match ``[project.optional-dependencies]`` keys in ``pyproject.toml``.
ALLOWED_DEPS_EXTRAS = frozenset(("pipeline", "inference", "training", "collection", "video"))

PYTHON_MODULES_BY_EXTRA: dict[str, tuple[str, ...]] = {
    "base": ("ai_image_detector",),
    "inference": ("numpy", "PIL", "safetensors", "torch", "torchvision"),
    "training": ("numpy", "PIL", "safetensors", "torch", "torchvision", "piexif", "sklearn"),
    "collection": ("datasets", "huggingface_hub", "PIL"),
    "video": ("cv2",),
}


def normalize_requested_extras(raw_extra: str) -> list[str]:
    requested: list[str] = []
    seen: set[str] = set()
    for item in raw_extra.split(","):
        extra = item.strip()
        if not extra:
            continue
        if extra not in ALLOWED_DEPS_EXTRAS:
            allowed = ",".join(sorted(ALLOWED_DEPS_EXTRAS))
            raise SystemExit(f"invalid_deps_extra_token token={extra!r} allowed={allowed}")
        if extra == "pipeline":
            return ["pipeline"]
        if extra in seen:
            continue
        seen.add(extra)
        requested.append(extra)
    if not requested:
        return ["pipeline"]
    return requested


def expanded_extras(raw_extra: str) -> list[str]:
    requested = normalize_requested_extras(raw_extra)
    if "pipeline" in requested:
        return list(PIPELINE_EXTRAS)
    return requested


def required_python_modules(raw_extra: str) -> list[str]:
    modules: list[str] = list(PYTHON_MODULES_BY_EXTRA["base"])
    for extra in expanded_extras(raw_extra):
        for module_name in PYTHON_MODULES_BY_EXTRA.get(extra, ()):
            if module_name not in modules:
                modules.append(module_name)
    return modules


def check_imports(raw_extra: str) -> None:
    for module_name in required_python_modules(raw_extra):
        importlib.import_module(module_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile-aware dependency helpers")
    parser.add_argument("--extras", default="pipeline", help="Comma-separated dependency extras")
    parser.add_argument(
        "--emit",
        choices=("json", "modules", "expanded-extras"),
        default="modules",
        help="Emit the normalized dependency view",
    )
    parser.add_argument(
        "--check-imports",
        action="store_true",
        help="Import each required Python module and exit nonzero on failure",
    )
    args = parser.parse_args()

    if args.check_imports:
        check_imports(args.extras)
        return 0

    normalized = normalize_requested_extras(args.extras)
    expanded = expanded_extras(args.extras)
    modules = required_python_modules(args.extras)

    if args.emit == "json":
        print(
            json.dumps(
                {
                    "requested_extras": normalized,
                    "expanded_extras": expanded,
                    "python_modules": modules,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.emit == "expanded-extras":
        print("\n".join(expanded))
        return 0

    print("\n".join(modules))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
