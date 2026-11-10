#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
# Future DORIS runs use 15000 iterations; the filename is retained for command compatibility.
python tools/train.py configs/hrda/baseline_resnet/doris_goji30IRRG2goji26IRRG_hrda_r101_no3vi.py
python tools/train.py configs/memoryadaptnet/doris_goji26IRRG2goji30RGB_memoryadaptnet_r101_no3vi.py
python tools/train.py configs/daformer/doris_goji26RGB2goji30RGB_daformer_r101_no3vi.py
