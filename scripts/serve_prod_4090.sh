#!/usr/bin/env bash
set -euo pipefail

echo "serve_disabled=1 reason=pipeline_only_mode"
echo "use_pipeline_commands=./local.sh setup|setup-full|collect|train|status"
exit 2
