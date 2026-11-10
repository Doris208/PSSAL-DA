#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "===== goji30IRRG2goji26IRRG 1/8 Baseline: HRDA + 3VI ====="
# python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi.py --work-dir "/root/shared-nvme/work_dirs/doris_stage1/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi"

echo "===== goji30IRRG2goji26IRRG 2/8 Physiology ====="
python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_phy_gate.py --work-dir "/root/shared-nvme/work_dirs/doris_stage1/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_phy_gate"

echo "===== goji30IRRG2goji26IRRG 3/8 Structural ====="
python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_struct_gate.py --work-dir "/root/shared-nvme/work_dirs/doris_stage1/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_struct_gate"

echo "===== goji30IRRG2goji26IRRG 4/8 MPARC ====="
python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_phy_struct_gate.py --work-dir "/root/shared-nvme/work_dirs/doris_stage1/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_phy_struct_gate"

echo "===== goji30IRRG2goji26IRRG 5/8 Prototype ====="
python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_ps_proto.py --work-dir "/root/shared-nvme/work_dirs/doris_stage1/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_ps_proto"

echo "===== goji30IRRG2goji26IRRG 6/8 Edge ====="
python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_struct_edge.py --work-dir "/root/shared-nvme/work_dirs/doris_stage1/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_struct_edge"

echo "===== goji30IRRG2goji26IRRG 7/8 PBCR ====="
python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_pbcr.py --work-dir "/root/shared-nvme/work_dirs/doris_stage1/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_pbcr"

echo "===== goji30IRRG2goji26IRRG 8/8 MPARC + PBCR (Full) ====="
python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge.py --work-dir "/root/shared-nvme/work_dirs/doris_stage1/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge"

echo "All goji30IRRG2goji26IRRG DORIS aligned 15k ablations finished."
