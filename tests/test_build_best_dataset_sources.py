from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


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


class _AuditSplit:
    def __init__(self, split: str, num_rows: int) -> None:
        self.split = split
        self.num_rows = num_rows


class _AuditInfo:
    def __init__(self, *, dataset_info: dict[str, object] | None = None) -> None:
        self.downloads = 1234
        self.likes = 44
        self.tags = ["image", "computer-vision", "license:apache-2.0", "screenshot"]
        self.description = "Browser screenshot real fake image dataset"
        self.cardData = {"license": "apache-2.0"}
        self.dataset_info = dataset_info or {
            "config_name": "default",
            "features": [
                {"name": "image", "type": {"_type": "Image"}},
                {"name": "label", "type": {"names": ["real", "ai"]}},
            ],
        }
        self.splits = [_AuditSplit("train", 120), _AuditSplit("validation", 30)]
        self.siblings = []
        self.config = "default"


class _AuditApi:
    def __init__(self, token: str | None = None) -> None:
        self.token = token

    def dataset_info(self, source_id: str):
        return _AuditInfo()


class _AuditApiMappingFeatures:
    def __init__(self, token: str | None = None) -> None:
        self.token = token

    def dataset_info(self, source_id: str):
        return _AuditInfo(
            dataset_info={
                "config_name": "default",
                "features": {
                    "image": {"_type": "Image"},
                    "label": {"_type": "ClassLabel", "names": ["real", "ai"]},
                },
            }
        )


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

    def test_audit_hf_sources_infers_fields_domains_and_label_map(self) -> None:
        old_api = build_best_dataset_sources.HfApi
        try:
            build_best_dataset_sources.HfApi = _AuditApi
            audited = build_best_dataset_sources.audit_hf_sources(["org/browser-dataset"], min_rows=100)
        finally:
            build_best_dataset_sources.HfApi = old_api

        self.assertEqual(len(audited), 1)
        entry = audited[0]
        self.assertTrue(entry.approved)
        self.assertEqual(entry.image_field, "image")
        self.assertEqual(entry.label_field, "label")
        self.assertEqual(entry.label_map, {"0": "real", "1": "ai"})
        self.assertEqual(entry.total_rows, 150)
        self.assertIn("screen", entry.domain_tags)

    def test_audit_hf_sources_accepts_mapping_feature_shape(self) -> None:
        old_api = build_best_dataset_sources.HfApi
        try:
            build_best_dataset_sources.HfApi = _AuditApiMappingFeatures
            audited = build_best_dataset_sources.audit_hf_sources(["org/browser-dataset"], min_rows=100)
        finally:
            build_best_dataset_sources.HfApi = old_api

        self.assertEqual(len(audited), 1)
        entry = audited[0]
        self.assertTrue(entry.approved)
        self.assertEqual(entry.image_field, "image")
        self.assertEqual(entry.label_field, "label")
        self.assertEqual(entry.label_map, {"0": "real", "1": "ai"})

    def test_build_source_list_filters_to_audit_approved_sources_and_writes_manifest(self) -> None:
        old_discover = build_best_dataset_sources.discover_hf_sources
        old_audit = build_best_dataset_sources.audit_hf_sources
        try:
            build_best_dataset_sources.discover_hf_sources = lambda **_: ["org/approved", "org/rejected"]
            build_best_dataset_sources.audit_hf_sources = lambda *args, **kwargs: [
                build_best_dataset_sources.AuditedSource(
                    source_id="org/approved",
                    score=2.0,
                    downloads=100,
                    likes=10,
                    license_markers=("apache-2.0",),
                    matched_groups=("photo",),
                    domain_tags=("photo",),
                    image_field="image",
                    label_field="label",
                    label_map={"0": "real", "1": "ai"},
                    split_names=("train",),
                    total_rows=1000,
                    approved=True,
                    rejection_reasons=(),
                    config_name="default",
                ),
                build_best_dataset_sources.AuditedSource(
                    source_id="org/rejected",
                    score=1.0,
                    downloads=20,
                    likes=1,
                    license_markers=("apache-2.0",),
                    matched_groups=(),
                    domain_tags=(),
                    image_field="",
                    label_field="",
                    label_map={},
                    split_names=(),
                    total_rows=50,
                    approved=False,
                    rejection_reasons=("missing_image_field",),
                    config_name="default",
                ),
            ]
            with tempfile.TemporaryDirectory() as tmpdir:
                audit_path = Path(tmpdir) / "audit.jsonl"
                args = SimpleNamespace(
                    no_default_sources=True,
                    sources_file="",
                    extra_source=[],
                    discover_hf=True,
                    hf_cache_file="",
                    hf_cache_only_if_present=False,
                    hf_query=["real fake images"],
                    hf_discovery_limit=5,
                    hf_max_sources=10,
                    hf_min_downloads=10,
                    hf_min_likes=1,
                    hf_min_quality_score=0.0,
                    hf_print_top=0,
                    hf_query_pause_ms=0,
                    token_env="HF_TOKEN",
                    hf_discovery_workers=1,
                    hf_require_open_license=True,
                    hf_license_allow=list(build_best_dataset_sources.DEFAULT_ALLOWED_LICENSE_TAGS),
                    hf_audit_sources=True,
                    hf_audit_file=str(audit_path),
                    hf_audit_min_rows=100,
                    hf_audit_require_image_field=True,
                    hf_audit_require_label_field=True,
                    hf_audit_filter_to_approved=True,
                )

                found = build_best_dataset_sources.build_source_list(args)
                audit_text = audit_path.read_text(encoding="utf-8")
        finally:
            build_best_dataset_sources.discover_hf_sources = old_discover
            build_best_dataset_sources.audit_hf_sources = old_audit

        self.assertEqual(found, ["org/approved"])
        self.assertIn("\"source_id\": \"org/approved\"", audit_text)
        self.assertIn("\"source_id\": \"org/rejected\"", audit_text)

    def test_build_source_list_applies_audit_even_when_cache_only_path_is_used(self) -> None:
        old_audit = build_best_dataset_sources.audit_hf_sources
        try:
            build_best_dataset_sources.audit_hf_sources = lambda *args, **kwargs: [
                build_best_dataset_sources.AuditedSource(
                    source_id="org/cached-approved",
                    score=2.0,
                    downloads=100,
                    likes=10,
                    license_markers=("apache-2.0",),
                    matched_groups=("photo",),
                    domain_tags=("photo",),
                    image_field="image",
                    label_field="label",
                    label_map={"0": "real", "1": "ai"},
                    split_names=("train",),
                    total_rows=1000,
                    approved=True,
                    rejection_reasons=(),
                    config_name="default",
                ),
                build_best_dataset_sources.AuditedSource(
                    source_id="org/cached-rejected",
                    score=1.0,
                    downloads=20,
                    likes=1,
                    license_markers=("apache-2.0",),
                    matched_groups=(),
                    domain_tags=(),
                    image_field="",
                    label_field="",
                    label_map={},
                    split_names=(),
                    total_rows=50,
                    approved=False,
                    rejection_reasons=("missing_image_field",),
                    config_name="default",
                ),
            ]
            with tempfile.TemporaryDirectory() as tmpdir:
                cache_path = Path(tmpdir) / "sources.txt"
                cache_path.write_text("org/cached-approved\norg/cached-rejected\n", encoding="utf-8")
                build_best_dataset_sources.save_cache_policy(
                    cache_path,
                    {
                        "queries": ["real fake images"],
                        "hf_discovery_limit": 5,
                        "hf_max_sources": 10,
                        "hf_min_downloads": 10,
                        "hf_min_likes": 1,
                        "hf_min_quality_score": 0.0,
                        "hf_print_top": 0,
                        "hf_discovery_workers": 1,
                        "hf_query_pause_ms": 0,
                        "hf_require_open_license": True,
                        "hf_license_allow": list(build_best_dataset_sources.DEFAULT_ALLOWED_LICENSE_TAGS),
                    },
                )
                audit_path = Path(tmpdir) / "audit.jsonl"
                args = SimpleNamespace(
                    no_default_sources=True,
                    sources_file="",
                    extra_source=[],
                    discover_hf=True,
                    hf_cache_file=str(cache_path),
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
                    hf_discovery_workers=1,
                    hf_require_open_license=True,
                    hf_license_allow=list(build_best_dataset_sources.DEFAULT_ALLOWED_LICENSE_TAGS),
                    hf_audit_sources=True,
                    hf_audit_file=str(audit_path),
                    hf_audit_min_rows=100,
                    hf_audit_require_image_field=True,
                    hf_audit_require_label_field=True,
                    hf_audit_filter_to_approved=True,
                )

                found = build_best_dataset_sources.build_source_list(args)
                audit_text = audit_path.read_text(encoding="utf-8")
        finally:
            build_best_dataset_sources.audit_hf_sources = old_audit

        self.assertEqual(found, ["org/cached-approved"])
        self.assertIn("\"source_id\": \"org/cached-rejected\"", audit_text)


if __name__ == "__main__":
    unittest.main()
