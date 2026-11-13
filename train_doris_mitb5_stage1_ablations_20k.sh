#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Includes all eight experiments per direction; 1 and 8 are existing configurations.

echo "===== goji26RGB2goji30RGB 1/8 Baseline: HRDA + 3VI ====="
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi"

echo "===== goji26RGB2goji30RGB 2/8 Physiology ====="
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_phy_gate"

echo "===== goji26RGB2goji30RGB 3/8 Structural ====="
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_struct_gate.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_struct_gate"

echo "===== goji26RGB2goji30RGB 4/8 MPARC ====="
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate"

echo "===== goji26RGB2goji30RGB 5/8 Prototype ====="
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_ps_proto.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_ps_proto"

echo "===== goji26RGB2goji30RGB 6/8 Edge ====="
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_struct_edge.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_struct_edge"

echo "===== goji26RGB2goji30RGB 7/8 PBCR ====="
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_pbcr.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_pbcr"

echo "===== goji26RGB2goji30RGB 8/8 MPARC + PBCR (Full) ====="
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge"

echo "All goji26RGB2goji30RGB DORIS aligned 15k ablations finished."

echo "===== goji26IRRG2goji30RGB 1/8 Baseline: HRDA + 3VI ====="
python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi"

echo "===== goji26IRRG2goji30RGB 2/8 Physiology ====="
python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_phy_gate.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_phy_gate"

echo "===== goji26IRRG2goji30RGB 3/8 Structural ====="
python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_struct_gate.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_struct_gate"

echo "===== goji26IRRG2goji30RGB 4/8 MPARC ====="
python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_phy_struct_gate.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_phy_struct_gate"

echo "===== goji26IRRG2goji30RGB 5/8 Prototype ====="
python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_ps_proto.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_ps_proto"

echo "===== goji26IRRG2goji30RGB 6/8 Edge ====="
python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_struct_edge.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_struct_edge"

echo "===== goji26IRRG2goji30RGB 7/8 PBCR ====="
python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_pbcr.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_pbcr"

echo "===== goji26IRRG2goji30RGB 8/8 MPARC + PBCR (Full) ====="
python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge"

echo "All goji26IRRG2goji30RGB DORIS aligned 15k ablations finished."

echo "===== goji30IRRG2goji26IRRG 1/8 Baseline: HRDA + 3VI ====="
python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi.py --work-dir "/root/shared-nvme/work_dirs/doris_stage1/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi"

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
