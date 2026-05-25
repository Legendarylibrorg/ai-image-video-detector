# Security Policy

## Supported Versions

Security updates are applied to the `main` branch.

## Reporting a Vulnerability

Please report security issues privately via [GitHub Security Advisories](https://github.com/Legendarylibrorg/ai-image-video-detector/security/advisories/new).

## Threat model (local pipeline)

This repository targets **local or VM-isolated** training and data collection. It is **not** a hardened multi-tenant service. Assume an attacker who can place files in dataset directories, supply checkpoints, or influence environment variables has significant leverage on the machine that runs the pipeline.

## Supply chain and bootstrap

- **Prefer a pinned revision** when using **`install.sh`**: set **`INSTALL_REV`** to a tag or branch name so **`git clone --depth 1 --branch`** pins the initial checkout (see **`install_security_notice`** on stderr when **`INSTALL_REV`** is unset). For arbitrary commit SHAs, clone manually from **docs/STARTUP.md** instead of relying on **`--branch`**. See also README **Safer bootstrap** for **`curl | bash`** flows.
- **`install.sh`** only clones the **official** GitHub URL by default. For other remotes, set **`INSTALL_ALLOW_CUSTOM_REPO=1`** (HTTPS only). Non-official **`github.com`** paths (forks) require **`INSTALL_ALLOW_NON_OFFICIAL_GITHUB_REPO=1`**. Other hosts must appear in **`INSTALL_REPO_HOST_ALLOWLIST`** (comma-separated, non-empty when custom clone is used; the validator defaults the env var to `github.com` when unset), or set **`INSTALL_ALLOW_ANY_HTTPS_HOST=1`** to restore broad HTTPS (not recommended). **`install_validate.py`** also rejects **`INSTALL_DIR`** values that resolve under forbidden system prefixes (see **`FORBIDDEN_INSTALL_PREFIXES`** in that file—e.g. `/etc/`, `/bin/`, `/boot/`), **embedded credentials in `REPO_URL`** (use credential helpers instead), and **oversized** URL/path strings.
- Setup runs **`env SETUP_INSTALL_SYSTEM_DEPS=0 ./local.sh setup`** without a shell parsing a dynamic command string.
- **`APT_PACKAGES`** is validated so tokens match Debian package-name characters only, blocking shell injection through that variable.
- **Python 3.11+** (GitHub **Security Checks** / **Dependency Updates** install the exact version from **`.github/ci-python-version.txt`** on `ubuntu-latest`; **`MANIFEST_MAX_WHEEL_CP`** in **`scripts/update_deps_lock.py`** must match that interpreter for wheel-tag selection). **`requirements.lock`** + **`requirements.lock.json`** are authoritative pins (per-artifact **SHA256**, verified vs PyPI in CI); **markdown is not**. The JSON records **one** PyPI file per package—another OS may still install the same **version** from a different wheel; regenerate on **Linux x86_64** (or the lock-refresh workflow) if you need the manifest to match CI.

## Hugging Face token and Hub trust

- Use a **read-scoped** Hugging Face token unless you truly need write access. The token is passed to libraries and containers that talk to the Hub; treat `.env` and shell environments as **secret-bearing**.
- By default, **`datasets.load_dataset`** is called with **`trust_remote_code=False`**. Hub datasets that require custom loading scripts need **`AID_HF_TRUST_REMOTE_CODE=1`**, a comma-separated **`AID_HF_TRUST_REMOTE_ALLOWLIST`** of **`namespace/dataset`** ids, **and** **`AID_ACCEPT_HF_TRUST_REMOTE_RISK=1`** or **`I_ACCEPT_HF_TRUST_RISK=1`** before **`trust_remote_code=True`** is passed for allowlisted ids. The legacy behavior (trust flag applies to **every** dataset) additionally requires **`AID_HF_TRUST_REMOTE_UNSAFE_GLOBAL=1`** with the same explicit accept flags (not recommended).
- Collection scripts require **`--out`**, **`--sources-file`**, **`--hf-cache-file`**, **`--hf-audit-file`**, and **`--cache-dir`** to resolve under **`AID_WORKSPACE_ROOT`** when set, otherwise under the process **cwd**. Relative **`../`** escapes and absolute paths outside that anchor are rejected. Existing path arguments that are symlink leaves are rejected before resolve. Docker Compose sets **`AID_WORKSPACE_ROOT=/workspace`**. **`scripts/ingest_model_outputs.py`** applies the same rules to **`--src`**, **`--dst`**, and **`--archive`**, walks image trees with **`followlinks=False`**, and rejects symlink image leaves.
- HF **discovery search strings** are validated in **`scripts/hf_data.validate_hf_discovery_query`** (max length **`AID_MAX_HF_DISCOVERY_QUERY_CHARS`**, default 512; no newlines / NUL). **Source-id list files** and **dataset source manifests** use bounded reads (same byte/line caps as **`AID_MAX_NONEMPTY_LINES_FILE_BYTES`** / **`AID_MAX_NONEMPTY_LINES_COUNT`** for lists; **`AID_MAX_SOURCE_MANIFEST_BYTES`** default 64 MiB for JSONL manifests) and reject symlink file leaves where applied.

## Checkpoints and media

- By default, **`.safetensors`** and **`.pt`** loads go through **`ai_image_detector.checkpoint_io`**, which uses **`O_NOFOLLOW`** (where available) to open the leaf path, then copies **exactly** the validated byte length into a private temp file before parsing (temp file mode **`0600`** on POSIX where supported). Staging briefly needs **up to ~2× the checkpoint size** in temp space. Set **`AID_CHECKPOINT_LOAD_STAGING=0`** only on trusted paths to skip the extra copy. Caps: **`AID_MAX_SAFETENSORS_FILE_BYTES`** and **`AID_MAX_TRAINING_CHECKPOINT_BYTES`** (default 2 GiB each).
- Training resumes from **`.pt`** files using PyTorch **`weights_only=True`** when the installed PyTorch supports it.

## Docker and isolation

- Compose uses dropped capabilities and a non-root UID/GID, but the **repository and data directories are bind-mounted**. Compromise inside the container can still read and write those host paths. The README’s **dedicated Linux VM** model remains the primary isolation boundary.

## Detection outputs

Model and heuristic outputs are **probabilistic**. Do not rely on them alone for high-stakes enforcement; use human review where misuse has real consequences.

## Data poisoning

Directories such as **`incoming_model_outputs`** feed training. Malicious or mislabeled content can **degrade or backdoor** models. ClamAV is **best-effort**, not a guarantee.
