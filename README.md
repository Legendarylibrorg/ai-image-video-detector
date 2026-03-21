# Local AI Image And Video Training Pipeline

This repository is for one job:
- collect Hugging Face image and video data locally
- train detectors locally
- rerun safely if a long setup stops partway through

It is not a production serving repo in the current mode.

## Open Source Notes

- License: MIT (see `LICENSE`).
- Security reporting: see `SECURITY.md`.
- Do not commit secrets (tokens, keys) or private datasets.
- Dataset and model licenses vary by source; verify each source license before commercial or production use.
- Detection outputs are probabilistic and can be wrong; review high-risk decisions with human oversight.

## Startup

Linux is the best-supported host.

### Linux quick start

1. Enter the repo:

```bash
cd /path/to/image-spam
```

2. Install system packages if your machine does not already have them. These are the commands that normally need `sudo`:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
```

3. Bootstrap the repo as your normal user:

```bash
./local.sh setup
```

4. Add your Hugging Face token to `.env`:

```bash
HF_TOKEN='your_token_here'
```

5. Run a quick sanity check:

```bash
./local.sh smoke
```

6. Start the resumable pipeline:

```bash
./local.sh run
```

Important notes:
- `./local.sh setup` already tries `apt-get` automatically on supported Linux hosts and uses `sudo` when available.
- Keep `sudo` on package-manager commands only. Run `./local.sh ...` and `bash scripts/...` as your normal user.
- `./local.sh run` is resumable: completed stages are skipped, training locks are waited out, and transient failures are retried.

### Startup commands

Use these first:

```bash
./local.sh setup
./local.sh smoke
./local.sh run
./local.sh status
```

Use these when you want more control:

```bash
./local.sh collect
./local.sh train
./local.sh retrain
./local.sh continuous
./local.sh check
./local.sh setup-full
```

### One-command startup

If you want setup plus the full collect-and-train flow in one command:

```bash
HF_TOKEN='your_token_here' ./local.sh setup-full
```

### Manual Linux bootstrap

Only use this if you do not want `./local.sh setup`:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
python3 -m venv .venv
source .venv/bin/activate
bash scripts/install_deps.sh
./local.sh check
```

### Startup troubleshooting

If setup stopped:

```bash
./local.sh setup-full
```

If collection seems slow:

```bash
./local.sh status
./local.sh smoke
```

If you only want to retrain:

```bash
./local.sh train
./local.sh retrain
```

If you changed dependencies:

```bash
./local.sh deps-update
bash scripts/install_deps.sh
```

## Docs

Use the docs for deeper explanations and lower-level reference:

- [docs/STARTUP.md](docs/STARTUP.md)
  Full Linux startup flow, manual bootstrap, setup options, and troubleshooting.
- [docs/COMMANDS.md](docs/COMMANDS.md)
  `./local.sh`, `scripts/do.sh`, wrappers, and lower-level command surfaces.
- [docs/REFERENCE.md](docs/REFERENCE.md)
  Higher-level reference notes for datasets, training, evaluation, video, and pipeline modes.
- [CONTRIBUTING.md](CONTRIBUTING.md)
  Contribution guidance.
- [SECURITY.md](SECURITY.md)
  Security reporting guidance.
