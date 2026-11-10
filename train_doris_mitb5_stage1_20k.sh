#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
# Group HRDA experiments by transfer direction under the DORIS stage-1 directory.
RUN_ROOT="/root/shared-nvme/work_dirs/doris_stage1"

# python tools/train.py configs/hrda/baseline_resnet/doris_goji30IRRG2goji26IRRG_hrda_r101_no3vi.py
# python tools/train.py configs/memoryadaptnet/doris_goji26IRRG2goji30RGB_memoryadaptnet_r101_no3vi.py
# python tools/train.py configs/daformer/doris_goji26RGB2goji30RGB_daformer_r101_no3vi.py

# python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi.py --work-dir "$RUN_ROOT/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi"
# python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge.py --work-dir "$RUN_ROOT/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge"
# python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi.py --work-dir "$RUN_ROOT/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi"
# python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py --work-dir "$RUN_ROOT/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge"
# python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi.py --work-dir "$RUN_ROOT/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi"
# python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py --work-dir "$RUN_ROOT/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge"

# Additional comparisons: one direction per method, as requested.
python tools/train.py configs/memoryadaptnet/doris_goji26IRRG2goji30RGB_memoryadaptnet_3vi.py --work-dir "$RUN_ROOT/goji26IRRG2goji30RGB_memoryadaptnet_3vi"
python tools/train.py configs/daformer/doris_goji26RGB2goji30RGB_daformer_3vi.py --work-dir "$RUN_ROOT/goji26RGB2goji30RGB_daformer_3vi"


