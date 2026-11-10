#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
# All three use 20000 iters, matching the reference runs confirmed by the user.
python tools/train.py configs/hrda/baseline_resnet/doris_goji30IRRG2goji26IRRG_hrda_r101_no3vi.py
python tools/train.py configs/memoryadaptnet/doris_goji26IRRG2goji30RGB_memoryadaptnet_r101_no3vi.py
python tools/train.py configs/daformer/doris_goji26RGB2goji30RGB_daformer_r101_no3vi.py
