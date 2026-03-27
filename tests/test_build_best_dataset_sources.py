from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_best_dataset_sources


class _FakeDataset:
    def __init__(
        self,
        ds_id: str,
        downloads: int,
        likes: int,
        tags: list[str],
        *,
        cardData: dict[str, object] | None = None,
    ) -> None:
        self.id = ds_id
        self.downloads = downloads
        self.likes = likes
        self.tags = tags
        self.cardData = cardData or {}


class _FakeApi:
    calls: list[dict[str, object]] = []
    init_tokens: list[str | None] = []

    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.init_tokens.append(token)

    def list_datasets(self, *, search=None, limit=None, sort=None):
        self.calls.append({"search": search, "limit": limit, "sort": sort})
        return [
            _FakeDataset(
                "org/real-fake-images",
                downloads=500,
                likes=20,
                tags=["image-classification", "image", "license:apache-2.0"],
            )
        ]


class _FailingIterable:
    def __iter__(self):
        raise RuntimeError("network down")


class _FailingApi:
    def __init__(self, token: str | None = None) -> None:
        self.token = token

    def list_datasets(self, *, search=None, limit=None, sort=None):
        return _FailingIterable()


class BuildBestDatasetSourcesTests(unittest.TestCase):
    def test_discover_hf_sources_uses_supported_list_datasets_signature(self) -> None:
        old_api = build_best_dataset_sources.HfApi
        try:
            _FakeApi.calls = []
            _FakeApi.init_tokens = []
            build_best_dataset_sources.HfApi = _FakeApi
            found = build_best_dataset_sources.discover_hf_sources(
                queries=["real fake images"],
                per_query_limit=5,
                max_sources=10,
                min_downloads=10,
                min_likes=1,
                min_quality_score=0.0,
                print_top_n=0,
            )
        finally:
            build_best_dataset_sources.HfApi = old_api

        self.assertEqual(found, ["org/real-fake-images"])
        self.assertEqual(
            _FakeApi.calls,
            [{"search": "real fake images", "limit": 5, "sort": "downloads"}],
        )

    def test_discover_hf_sources_handles_iteration_failures(self) -> None:
        old_api = build_best_dataset_sources.HfApi
        try:
            build_best_dataset_sources.HfApi = _FailingApi
            found = build_best_dataset_sources.discover_hf_sources(
                queries=["real fake images"],
                per_query_limit=5,
                max_sources=10,
                min_downloads=10,
                min_likes=1,
                min_quality_score=0.0,
                print_top_n=0,
            )
        finally:
            build_best_dataset_sources.HfApi = old_api

        self.assertEqual(found, [])

    def test_discover_hf_sources_treats_blank_token_as_missing(self) -> None:
        old_api = build_best_dataset_sources.HfApi
        try:
            _FakeApi.calls = []
            _FakeApi.init_tokens = []
            build_best_dataset_sources.HfApi = _FakeApi
            build_best_dataset_sources.discover_hf_sources(
                queries=["real fake images"],
                per_query_limit=5,
                max_sources=10,
                min_downloads=10,
                min_likes=1,
                min_quality_score=0.0,
                print_top_n=0,
                token="  ",
            )
        finally:
            build_best_dataset_sources.HfApi = old_api

        self.assertEqual(_FakeApi.init_tokens, [None])

    def test_discover_hf_sources_filters_out_non_open_license_datasets(self) -> None:
        class _LicenseApi:
            def __init__(self, token: str | None = None) -> None:
                self.token = token

            def list_datasets(self, *, search=None, limit=None, sort=None):
                return [
                    _FakeDataset(
                        "org/open-real-fake-images",
                        downloads=500,
                        likes=20,
                        tags=["image-classification", "image", "license:apache-2.0"],
                    ),
                    _FakeDataset(
                        "org/restricted-real-fake-images",
                        downloads=800,
                        likes=30,
                        tags=["image-classification", "image", "license:cc-by-nc-4.0"],
                    ),
                ]

        old_api = build_best_dataset_sources.HfApi
        try:
            build_best_dataset_sources.HfApi = _LicenseApi
            found = build_best_dataset_sources.discover_hf_sources(
                queries=["real fake images"],
                per_query_limit=5,
                max_sources=10,
                min_downloads=10,
                min_likes=1,
                min_quality_score=0.0,
                print_top_n=0,
            )
        finally:
            build_best_dataset_sources.HfApi = old_api

        self.assertEqual(found, ["org/open-real-fake-images"])

    def test_discover_hf_sources_keeps_real_only_visual_datasets(self) -> None:
        class _VisualRealApi:
            def __init__(self, token: str | None = None) -> None:
                self.token = token

            def list_datasets(self, *, search=None, limit=None, sort=None):
                return [
                    _FakeDataset(
                        "org/dslr-photo-corpus",
                        downloads=700,
                        likes=25,
                        tags=["computer-vision", "license:apache-2.0"],
                    ),
                    _FakeDataset(
                        "org/audio-corpus",
                        downloads=900,
                        likes=40,
                        tags=["audio", "license:apache-2.0"],
                    ),
                ]

        old_api = build_best_dataset_sources.HfApi
        try:
            build_best_dataset_sources.HfApi = _VisualRealApi
            found = build_best_dataset_sources.discover_hf_sources(
                queries=["dslr photo dataset"],
                per_query_limit=5,
                max_sources=10,
                min_downloads=10,
                min_likes=1,
                min_quality_score=0.0,
                print_top_n=0,
            )
        finally:
            build_best_dataset_sources.HfApi = old_api

        self.assertEqual(found, ["org/dslr-photo-corpus"])

    def test_build_source_list_falls_back_to_live_discovery_when_cache_is_empty(self) -> None:
        old_discover = build_best_dataset_sources.discover_hf_sources
        try:
            build_best_dataset_sources.discover_hf_sources = lambda **_: ["org/real-fake-images"]
            with tempfile.TemporaryDirectory() as tmpdir:
                args = SimpleNamespace(
                    no_default_sources=True,
                    sources_file="",
                    extra_source=[],
                    discover_hf=True,
                    hf_cache_file=f"{tmpdir}/empty_cache.txt",
                    hf_cache_only_if_present=True,
                    hf_query=["real fake images"],
                    hf_discovery_limit=5,
                    hf_max_sources=10,
                    hf_min_downloads=10,
                    hf_min_likes=1,
                    hf_min_quality_score=0.0,
                    hf_print_top=0,
                    hf_query_pause_ms=0,
                    token_env="HF_TOKEN",
                )
                open(args.hf_cache_file, "w", encoding="utf-8").close()

                found = build_best_dataset_sources.build_source_list(args)
        finally:
            build_best_dataset_sources.discover_hf_sources = old_discover

        self.assertEqual(found, ["org/real-fake-images"])

    def test_build_source_list_refetches_when_cache_policy_is_stale(self) -> None:
        old_discover = build_best_dataset_sources.discover_hf_sources
        try:
            build_best_dataset_sources.discover_hf_sources = lambda **_: ["org/new-stronger-source"]
            with tempfile.TemporaryDirectory() as tmpdir:
                cache_path = Path(tmpdir) / "sources.txt"
                cache_path.write_text("org/old-source\n", encoding="utf-8")
                build_best_dataset_sources.save_cache_policy(
                    cache_path,
                    {
                        "queries": ["old query"],
                        "hf_discovery_limit": 5,
                        "hf_max_sources": 10,
                        "hf_min_downloads": 10,
                        "hf_min_likes": 1,
                        "hf_min_quality_score": 0.0,
                        "hf_print_top": 0,
                        "hf_query_pause_ms": 0,
                    },
                )
                args = SimpleNamespace(
                    no_default_sources=True,
                    sources_file="",
                    extra_source=[],
                    discover_hf=True,
                    hf_cache_file=str(cache_path),
                    hf_cache_only_if_present=True,
                    hf_query=["better query"],
                    hf_discovery_limit=5,
                    hf_max_sources=10,
                    hf_min_downloads=10,
                    hf_min_likes=1,
                    hf_min_quality_score=1.0,
                    hf_print_top=0,
                    hf_query_pause_ms=0,
                    token_env="HF_TOKEN",
                )

                found = build_best_dataset_sources.build_source_list(args)
                cached_sources = build_best_dataset_sources.read_sources_file(cache_path)
                cached_policy = build_best_dataset_sources.load_cache_policy(cache_path)
        finally:
            build_best_dataset_sources.discover_hf_sources = old_discover

        self.assertEqual(found, ["org/new-stronger-source"])
        self.assertEqual(cached_sources, ["org/new-stronger-source"])
        self.assertEqual(cached_policy, build_best_dataset_sources.discovery_policy(args))


if __name__ == "__main__":
    unittest.main()
