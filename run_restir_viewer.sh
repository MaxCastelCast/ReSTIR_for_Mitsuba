#!/usr/bin/env bash
set -euo pipefail

# Find the repository root
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

RESTIR_DIR="$REPO_ROOT/src/python/ReSTIR"

# Check Pixi installation
if ! command -v pixi >/dev/null 2>&1; then
    echo "Error: Pixi is not installed or is not on PATH." >&2
    echo "Install Pixi first: https://pixi.sh/latest/installation/" >&2
    exit 1
fi

# Check project structure
if [[ ! -f "$RESTIR_DIR/pixi.toml" ]]; then
    echo "Error: Cannot find $RESTIR_DIR/pixi.toml" >&2
    exit 1
fi

# Navigate to the Pixi project
cd "$RESTIR_DIR"

echo "== Checking/installing ReSTIR Pixi dependencies =="

pixi install

echo "== Launching ReSTIR viewer =="

pixi run viewer