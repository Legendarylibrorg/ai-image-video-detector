# Security Policy

## Supported Versions

Security updates are applied to the `main` branch.

## Reporting a Vulnerability

Please report security issues privately via [GitHub Security Advisories](https://github.com/Legendarylibrorg/ai-image-video-detector/security/advisories/new).

## Threat model (local pipeline)

This repository targets **local or VM-isolated** training and data collection. It is **not** a hardened multi-tenant service. Assume an attacker who can place files in dataset directories, supply checkpoints, or influence environment variables has significant leverage on the machine that runs the pipeline.

## Supply chain and bootstrap

- **Prefer a pinned revision** when using the remote installer, instead of always following `main` (see README **Safer bootstrap**).
- **`install.sh`** only clones the **official** GitHub URL by default. For other remotes, set **`INSTALL_ALLOW_CUSTOM_REPO=1`** (HTTPS only). Non-official **`github.com`** paths (forks) require **`INSTALL_ALLOW_NON_OFFICIAL_GITHUB_REPO=1`**. Other hosts must appear in **`INSTALL_REPO_HOST_ALLOWLIST`** (comma-separated, non-empty when custom clone is used; the validator defaults the env var to `github.com` when unset), or set **`INSTALL_ALLOW_ANY_HTTPS_HOST=1`** to restore broad HTTPS (not recommended). **`install_validate.py`** also rejects **`INSTALL_DIR`** values that resolve under forbidden system prefixes (see **`FORBIDDEN_INSTALL_PREFIXES`** in that file—e.g. `/etc/`, `/bin/`, `/boot/`), **embedded credentials in `REPO_URL`** (use credential helpers instead), and **oversized** URL/path strings.
- Setup runs **`env SETUP_INSTALL_SYSTEM_DEPS=0 ./local.sh setup`** without a shell parsing a dynamic command string.
- **`APT_PACKAGES`** is validated so tokens match Debian package-name characters only, blocking shell injection through that variable.
- **Python dependencies** are pinned in `requirements.lock` with hash verification in CI.

## Hugging Face token and Hub trust

- Use a **read-scoped** Hugging Face token unless you truly need write access. The token is passed to libraries and containers that talk to the Hub; treat `.env` and shell environments as **secret-bearing**.
- By default, **`datasets.load_dataset`** is called with **`trust_remote_code=False`**. Hub datasets that require custom loading scripts need **`AID_HF_TRUST_REMOTE_CODE=1`** plus a comma-separated **`AID_HF_TRUST_REMOTE_ALLOWLIST`** of **`namespace/dataset`** ids. Only listed repos receive **`trust_remote_code=True`**. For the legacy behavior (trust flag applies to every dataset), set **`AID_HF_TRUST_REMOTE_UNSAFE_GLOBAL=1`** (not recommended).
- Collection scripts require **`--out`**, **`--sources-file`**, **`--hf-cache-file`**, **`--hf-audit-file`**, and **`--cache-dir`** to resolve under **`AID_WORKSPACE_ROOT`** when set, otherwise under the process **cwd**. Relative **`../`** escapes and absolute paths outside that anchor are rejected. Docker Compose sets **`AID_WORKSPACE_ROOT=/workspace`**. **`scripts/ingest_model_outputs.py`** applies the same rules to **`--src`**, **`--dst`**, and **`--archive`**, walks image trees with **`followlinks=False`**, and rejects symlink image leaves.

## Checkpoints and media

- By default, **`.safetensors`** and **`.pt`** loads go through **`ai_image_detector.checkpoint_io`**, which uses **`O_NOFOLLOW`** (where available) to open the leaf path, then copies **exactly** the validated byte length into a private temp file before parsing (temp file mode **`0600`** on POSIX where supported). Staging briefly needs **up to ~2× the checkpoint size** in temp space. Set **`AID_CHECKPOINT_LOAD_STAGING=0`** only on trusted paths to skip the extra copy. Caps: **`AID_MAX_SAFETENSORS_FILE_BYTES`** and **`AID_MAX_TRAINING_CHECKPOINT_BYTES`** (default 2 GiB each).
- Training resumes from **`.pt`** files using PyTorch **`weights_only=True`** when the installed PyTorch supports it.

## Docker and isolation

- Compose uses dropped capabilities and a non-root UID/GID, but the **repository and data directories are bind-mounted**. Compromise inside the container can still read and write those host paths. The README’s **dedicated Linux VM** model remains the primary isolation boundary.

## Detection outputs

Model and heuristic outputs are **probabilistic**. Do not rely on them alone for high-stakes enforcement; use human review where misuse has real consequences.

## Data poisoning

Directories such as **`incoming_model_outputs`** feed training. Malicious or mislabeled content can **degrade or backdoor** models. ClamAV is **best-effort**, not a guarantee.
