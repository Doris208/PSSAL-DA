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


D="goji26IRRG2goji30RGB"
ROOT="/root/shared-nvme/work_dirs/doris_struct_distinct_v2/$D"
TILE="/root/shared-nvme/mydataset/30/rgb_dir/val/3_1536_640_1792_896.tif"

# Baseline
python tools/replay_baseline_prediction.py \
  --direction "$D" \
  --checkpoint "$ROOT/${D}_hrda_3vi/iter_2000.pth" \
  --target-file "$TILE"

# H-Phy
python tools/replay_hphy_mechanism.py \
  --direction "$D" \
  --checkpoint "$ROOT/${D}_hrda_3vi_phy_gate/best_Veg_mIoU_iter_11000.pth" \
  --checkpoint-baseline "$ROOT/${D}_hrda_3vi/iter_2000.pth" \
  --target-file "$TILE"

# Structural V2
python tools/replay_struct_mechanism.py \
  --direction "$D" \
  --checkpoint "$ROOT/${D}_hrda_3vi_struct_gate/best_NonVeg_mIoU_iter_9000.pth" \
  --checkpoint-baseline "$ROOT/${D}_hrda_3vi/iter_2000.pth" \
  --target-file "$TILE"
