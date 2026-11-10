#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
RUN_ROOT="/root/shared-nvme/work_dirs/doris_stage1"

# Run only the two additions; do not restart the original six experiments.
python tools/train.py configs/memoryadaptnet/doris_goji26IRRG2goji30RGB_memoryadaptnet_3vi.py --work-dir "$RUN_ROOT/goji26IRRG2goji30RGB_memoryadaptnet_3vi"
python tools/train.py configs/daformer/doris_goji26RGB2goji30RGB_daformer_3vi.py --work-dir "$RUN_ROOT/goji26RGB2goji30RGB_daformer_3vi"
