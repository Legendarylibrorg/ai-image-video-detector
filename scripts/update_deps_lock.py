#!/usr/bin/env python3
"""Pin the pipeline stack: newest stable release per dependency on PyPI (no prereleases).

Versions match requires-python in pyproject.toml. Torch and torchvision stay aligned
using TORCHVISION_SERIES_BY_TORCH_SERIES in this file. Wheel artifacts prefer manylinux
x86_64 with the highest compatible ``cp`` tag up to MANIFEST_MAX_WHEEL_CP (CI Python).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from urllib.error import URLError
from urllib.request import urlopen

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

try:  # pragma: no cover - packaging is usually available via pip
    from packaging.specifiers import SpecifierSet
    from packaging.version import InvalidVersion, Version
except ModuleNotFoundError:  # pragma: no cover
    from pip._vendor.packaging.specifiers import SpecifierSet  # type: ignore[no-redef]
    from pip._vendor.packaging.version import InvalidVersion, Version  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK_FILE = ROOT / "requirements.lock"
DEFAULT_MANIFEST_FILE = ROOT / "requirements.lock.json"
PYPI_JSON = "https://pypi.org/pypi/{project}/json"
PYPI_VERSION_JSON = "https://pypi.org/pypi/{project}/{version}/json"

# TorchVision publishes an explicit compatibility table on PyPI. We mirror the
# stable torch->torchvision major/minor mapping here so the updater can keep the
# pair aligned while still taking the newest compatible patch release.
# Prefer manylinux x86_64 wheels tagged at or below this CPython ABI so the manifest
# matches GitHub Actions (see `.github/ci-python-version.txt` and workflows) and typical
# `pip install` on 3.11–3.14. If PyPI only ships newer `cp` wheels, selection falls back
# to the highest available tag (bump this when CI moves to a newer interpreter).
MANIFEST_MAX_WHEEL_CP = Version("3.14")

TORCHVISION_SERIES_BY_TORCH_SERIES = {
    "2.13": "0.28",
    "2.12": "0.27",
    "2.11": "0.26",
    "2.10": "0.25",
    "2.9": "0.24",
    "2.8": "0.23",
    "2.7": "0.22",
    "2.6": "0.21",
    "2.5": "0.20",
    "2.4": "0.19",
    "2.3": "0.18",
    "2.2": "0.17",
}


def fetch_json(url: str) -> dict:
    try:
        with urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (URLError, OSError):
        out = subprocess.check_output(["curl", "-sS", "--max-time", "30", url], text=True)
        return json.loads(out)


def normalize_requirement_name(requirement: str) -> str:
    requirement = requirement.split(";", 1)[0].strip()
    match = re.match(r"([A-Za-z0-9_.-]+)", requirement)
    if match is None:
        raise ValueError(f"unsupported_requirement={requirement}")
    return match.group(1)


def pypi_project_name(requirement_name: str) -> str:
    return requirement_name.replace("_", "-")


def load_project_config(pyproject_path: Path) -> dict:
    return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))


def repo_python_floor(project_cfg: dict) -> Version:
    requires_python = project_cfg["project"]["requires-python"]
    match = re.search(r">=\s*([0-9]+(?:\.[0-9]+)*)", requires_python)
    if match is None:
        raise ValueError(f"unsupported_requires_python={requires_python}")
    return Version(match.group(1))


def pipeline_requirements(project_cfg: dict) -> list[str]:
    pipeline = project_cfg["project"]["optional-dependencies"]["pipeline"]
    return [normalize_requirement_name(item) for item in pipeline]


def is_version_compatible(requires_python: str | None, python_floor: Version) -> bool:
    if not requires_python:
        return True
    try:
        return python_floor in SpecifierSet(requires_python)
    except (InvalidVersion, ValueError):
        return True


def non_yanked_files(files: list[dict], python_floor: Version) -> list[dict]:
    compatible: list[dict] = []
    for item in files:
        if item.get("yanked"):
            continue
        if not is_version_compatible(item.get("requires_python"), python_floor):
            continue
        compatible.append(item)
    return compatible


def parse_version(version: str) -> Version | None:
    try:
        return Version(version)
    except InvalidVersion:
        return None


def wheel_abi_python_version(filename: str) -> Version | None:
    """Parse the first ``cpXYZ`` ABI segment from a wheel filename (PEP 425)."""
    match = re.search(r"-cp(\d+)-", filename)
    if match is None:
        return None
    nodot = match.group(1)
    try:
        if len(nodot) == 2:
            return Version(f"{nodot[0]}.{nodot[1]}")
        if len(nodot) == 3:
            return Version(f"{nodot[0]}.{nodot[1:]}")
    except InvalidVersion:
        return None
    return None


def select_preferred_artifact(files: list[dict]) -> dict:
    sdists = [item for item in files if item.get("packagetype") == "sdist"]
    if sdists:
        return sorted(sdists, key=lambda item: item["filename"])[0]
    universal = [
        item
        for item in files
        if item.get("packagetype") == "bdist_wheel" and "py3-none-any" in item.get("filename", "")
    ]
    if universal:
        return sorted(universal, key=lambda item: item["filename"])[0]
    manylinux_x86_64 = [
        item
        for item in files
        if item.get("packagetype") == "bdist_wheel"
        and "manylinux" in item.get("filename", "")
        and "x86_64" in item.get("filename", "")
    ]
    tagged = [(wheel_abi_python_version(item["filename"]), item) for item in manylinux_x86_64]
    tagged = [(ver, item) for ver, item in tagged if ver is not None]
    capped = [(ver, item) for ver, item in tagged if ver <= MANIFEST_MAX_WHEEL_CP]
    tier = capped if capped else tagged
    if tier:
        max_ver = max(ver for ver, _ in tier)
        best = [item for ver, item in tier if ver == max_ver]
        return sorted(best, key=lambda item: item["filename"])[0]
    if manylinux_x86_64:
        return sorted(manylinux_x86_64, key=lambda item: item["filename"])[0]
    linux_wheels = [item for item in files if "manylinux" in item.get("filename", "")]
    if linux_wheels:
        return sorted(linux_wheels, key=lambda item: item["filename"])[0]
    cp_wheels = [
        item
        for item in files
        if item.get("packagetype") == "bdist_wheel" and re.search(r"-cp\d+-", item.get("filename", ""))
    ]
    if cp_wheels:
        return sorted(cp_wheels, key=lambda item: item["filename"])[0]
    return sorted(files, key=lambda item: item["filename"])[0]


def latest_compatible_release(project_name: str, python_floor: Version, *, prefix: str | None = None) -> tuple[str, dict]:
    payload = fetch_json(PYPI_JSON.format(project=project_name))
    releases = payload.get("releases", {})
    candidates: list[tuple[Version, str]] = []
    for version_text in releases:
        parsed = parse_version(version_text)
        if parsed is None or parsed.is_prerelease or parsed.is_devrelease:
            continue
        if prefix is not None and not version_text.startswith(prefix + "."):
            continue
        candidates.append((parsed, version_text))
    for _, version_text in sorted(candidates, reverse=True):
        files = non_yanked_files(releases.get(version_text, []), python_floor)
        if files:
            return version_text, payload
    raise SystemExit(f"no_compatible_release project={project_name} python_floor={python_floor}")


def lock_entry(requirement_name: str, project_name: str, version_text: str, payload: dict, python_floor: Version) -> dict:
    files = non_yanked_files(payload["releases"][version_text], python_floor)
    if not files:
        raise SystemExit(f"no_compatible_files project={project_name} version={version_text}")
    artifact = select_preferred_artifact(files)
    return {
        "name": requirement_name,
        "project": project_name,
        "version": version_text,
        "requires_python": artifact.get("requires_python") or payload.get("info", {}).get("requires_python"),
        "artifact": {
            "filename": artifact["filename"],
            "packagetype": artifact["packagetype"],
            "sha256": artifact["digests"]["sha256"],
            "url": artifact["url"],
        },
    }


def resolve_torchvision_version(torch_version: str, python_floor: Version) -> tuple[str, dict]:
    series = ".".join(torch_version.split(".")[:2])
    if series not in TORCHVISION_SERIES_BY_TORCH_SERIES:
        raise SystemExit(f"unsupported_torch_series={series} update_torchvision_map=1")
    prefix = TORCHVISION_SERIES_BY_TORCH_SERIES[series]
    return latest_compatible_release("torchvision", python_floor, prefix=prefix)


def resolve_lock_entries(pyproject_path: Path) -> tuple[dict, list[dict]]:
    project_cfg = load_project_config(pyproject_path)
    python_floor = repo_python_floor(project_cfg)
    requirement_names = pipeline_requirements(project_cfg)
    results: list[dict] = []
    resolved_versions: dict[str, str] = {}
    for requirement_name in requirement_names:
        project_name = pypi_project_name(requirement_name)
        if requirement_name == "torchvision":
            torch_version = resolved_versions.get("torch")
            if torch_version is None:
                raise SystemExit("torchvision_resolution_requires_torch_first")
            version_text, payload = resolve_torchvision_version(torch_version, python_floor)
        else:
            version_text, payload = latest_compatible_release(project_name, python_floor)
        resolved_versions[requirement_name] = version_text
        results.append(lock_entry(requirement_name, project_name, version_text, payload, python_floor))
    return project_cfg, results


def write_lock(lock_file: Path, entries: list[dict]) -> None:
    body = "".join(f"{entry['name']}=={entry['version']}\n" for entry in entries)
    lock_file.write_text(body, encoding="utf-8")


def write_manifest(manifest_file: Path, project_cfg: dict, entries: list[dict]) -> None:
    payload = {
        "generated_from": "scripts/update_deps_lock.py",
        "python_requires": project_cfg["project"]["requires-python"],
        "packages": entries,
    }
    manifest_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_lock_file(lock_file: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw_line in lock_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, version = line.partition("==")
        if not sep:
            raise SystemExit(f"invalid_lock_line={line}")
        rows.append((name.strip(), version.strip()))
    return rows


def verify_manifest(lock_file: Path, manifest_file: Path, *, require_current: bool) -> None:
    lock_rows = parse_lock_file(lock_file)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_rows = manifest.get("packages", [])
    if len(lock_rows) != len(manifest_rows):
        raise SystemExit("lock_manifest_length_mismatch")
    for (lock_name, lock_version), manifest_entry in zip(lock_rows, manifest_rows):
        if lock_name != manifest_entry["name"] or lock_version != manifest_entry["version"]:
            raise SystemExit(f"lock_manifest_mismatch package={lock_name}")
        payload = fetch_json(PYPI_VERSION_JSON.format(project=manifest_entry["project"], version=lock_version))
        matched = None
        for item in payload.get("urls", []):
            if item["filename"] == manifest_entry["artifact"]["filename"]:
                matched = item
                break
        if matched is None:
            raise SystemExit(f"missing_manifest_artifact package={lock_name} version={lock_version}")
        if matched.get("yanked"):
            raise SystemExit(f"yanked_manifest_artifact package={lock_name} version={lock_version}")
        if matched["digests"]["sha256"] != manifest_entry["artifact"]["sha256"]:
            raise SystemExit(f"hash_mismatch package={lock_name} version={lock_version}")
    if require_current:
        project_cfg, latest_entries = resolve_lock_entries(ROOT / "pyproject.toml")
        current = [(name, version) for name, version in lock_rows]
        latest = [(entry["name"], entry["version"]) for entry in latest_entries]
        if current != latest:
            raise SystemExit("lock_out_of_date")
        if manifest.get("python_requires") != project_cfg["project"]["requires-python"]:
            raise SystemExit("manifest_python_requires_mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description="Update and verify the dependency lock.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    update = subparsers.add_parser("update", help="Refresh lock and hash manifest")
    update.add_argument("--lock-file", default=str(DEFAULT_LOCK_FILE))
    update.add_argument("--manifest-file", default=str(DEFAULT_MANIFEST_FILE))
    update.add_argument("--pyproject", default=str(ROOT / "pyproject.toml"))

    verify = subparsers.add_parser("verify", help="Verify lock/hash manifest against PyPI")
    verify.add_argument("--lock-file", default=str(DEFAULT_LOCK_FILE))
    verify.add_argument("--manifest-file", default=str(DEFAULT_MANIFEST_FILE))
    verify.add_argument("--require-current", action="store_true")

    args = parser.parse_args()
    if args.command == "update":
        pyproject_path = Path(args.pyproject)
        project_cfg, entries = resolve_lock_entries(pyproject_path)
        write_lock(Path(args.lock_file), entries)
        write_manifest(Path(args.manifest_file), project_cfg, entries)
        print(f"deps_lock=updated file={args.lock_file}")
        print(f"deps_manifest=updated file={args.manifest_file}")
        return 0

    verify_manifest(Path(args.lock_file), Path(args.manifest_file), require_current=args.require_current)
    print(f"deps_lock=verified file={args.lock_file}")
    print(f"deps_manifest=verified file={args.manifest_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
