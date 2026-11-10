#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

if [ "$#" -ne 0 ]; then
    echo "Edit the defaults in tools/replay_hphy_mechanism.py, or run its Python command directly." >&2
    exit 2
fi

# Defaults are defined near the top of tools/replay_hphy_mechanism.py.
echo "===== goji26RGB2goji30RGB H-Phy forward inference and mechanism figures ====="
python tools/replay_hphy_mechanism.py

echo "All goji26RGB2goji30RGB H-Phy mechanism outputs finished."
