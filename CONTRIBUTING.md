# Contributing

Thanks for contributing.

## Before You Start

- Open an issue or draft PR when the change is large, user-facing, or changes the training/data flow.
- Do not open public issues for security problems. Use the process in [SECURITY.md](SECURITY.md).
- Keep changes focused. Small PRs are much easier to review and safer to merge.

## Development Setup

Native Linux:

```bash
./local.sh setup
```

Narrower local installs are also supported:

```bash
DEPS_EXTRA=collection ./local.sh deps
DEPS_EXTRA=training ./local.sh deps
```

## Expectations For Changes

- Preserve the existing `ai` vs `real` training/data flow unless the change explicitly updates that contract.
- Avoid committing datasets, model artifacts, caches, or secrets.
- Keep docs in sync when command behavior, paths, or bootstrap flow changes.
- Add or update tests for behavior changes, especially around shell wrappers and install paths.

## Checks To Run

Run the full test suite before opening a PR:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Useful targeted checks:

```bash
bash -n local.sh scripts/install_deps.sh scripts/doctor.sh
python3 -m py_compile src/ai_image_detector/cli.py
python3 scripts/update_deps_lock.py verify --require-current
```

If you use pre-commit, install it and run:

```bash
pre-commit run --all-files
```

## Pull Requests

- Branch from `main`.
- Explain what changed and why.
- Call out behavior changes, dependency changes, and docs updates.
- Mention any skipped checks or environment-specific limitations.

The PR template in [`.github/pull_request_template.md`](.github/pull_request_template.md) is the default checklist.
