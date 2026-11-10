# DORIS Stage 1 新版 24 组消融实验

> 当前Physiology/MPARC/Full共9组已启用balanced_high_conf；9组指3个方向各自的Physiology/MPARC/Full。用户已确认当前消融尚未运行，直接使用这些配置和原定输出目录首次训练即可。具体公式、采集和验证见[doris_physiology_optimization_zh.md](doris_physiology_optimization_zh.md)。

## 本轮修改

新版共24份带`doris_`前缀的配置：每方向编号2–7使用较新的18份中间消融配置，编号1 Baseline和编号8 Full保留现有六份的训练设置。此次目录整理将24份全部移到`configs/hrda/doris/<迁移方向>/`，同步更新相关脚本、检查工具和文档引用，旧版无前缀配置不动。

新配置沿用对应旧消融配置的有效定义方式（方向对应的ablation_common、make_model_override、make_uda_override），并采用当前对应方向doris Full的新版模块参数；省略旧文件中已被末尾override替代的历史定义。每个配置独立列明其启用模块及新版覆盖值，未修改公共模型源码或旧helper默认值。

|编号|后缀|生理|结构|原型|边界|
|---|---|---|---|---|---|
|1|无|关|关|关|关|
|2|_phy_gate|开|关|关|关|
|3|_struct_gate|关|开|关|关|
|4|_phy_struct_gate|开|开|关|关|
|5|_ps_proto|关|关|开|关|
|6|_struct_edge|关|关|关|开|
|7|_pbcr|关|关|开|开|
|8|_proto_struct_edge|开|开|开|开|

同一方向的所有启用模块参数与当前Full逐项相同：结构融合doris_confidence_guided；边界响应doris_reliable/valid_boundary_only；边界损失doris_balanced；原型tau_proto=0.25、tau_conf_proto=0.88。没有额外调参。

|迁移方向|原型更新起点|原型加权/损失起点|lambda_proto_tgt|proto_min_weight / power|边界源/目标强度|
|---|---|---|---|---|---|
|26RGB→30RGB|3000|6000|0.001|0.88 / 0.70|0.01 / 0.006|
|26IRRG→30RGB|8000|14000|0.0002|0.98 / 0.35|0.004 / 0.002|
|30IRRG→26IRRG|8000|12000|0.0003|0.95 / 0.35|0.006 / 0.003|

三组均为MiT-B5、6通道、5类、20K、batch=2、lr=6e-5。数据划分、通道顺序、归一化、增强、种子、优化器、学习率时序、验证与checkpoint协议均与对应Full一致。这里的对齐是同一迁移任务内，不宣称三个方向使用相同模块强度。

## 当前目录结构

```text
configs/hrda/doris/
  goji26RGB2goji30RGB/       # 8份doris_配置
  goji26IRRG2goji30RGB/      # 8份doris_配置
  goji30IRRG2goji26IRRG/     # 8份doris_配置

/root/shared-nvme/work_dirs/doris_stage1/
  goji26RGB2goji30RGB/<实验名>/
  goji26IRRG2goji30RGB/<实验名>/
  goji30IRRG2goji26IRRG/<实验名>/
```

Baseline三个文件原本已有doris_前缀，因此直接迁移现有副本，避免重复。子目录中的配置通过`../../pssalda_*_ablation_common.py`继承原公共配置；绝对Python helper导入保持不变。

## 18份中间消融配置清单

- `configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_pbcr.py`
- `configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_phy_gate.py`
- `configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_phy_struct_gate.py`
- `configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_ps_proto.py`
- `configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_struct_edge.py`
- `configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_struct_gate.py`
- `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_pbcr.py`
- `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py`
- `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate.py`
- `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_ps_proto.py`
- `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_struct_edge.py`
- `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_struct_gate.py`
- `configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_pbcr.py`
- `configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_phy_gate.py`
- `configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_phy_struct_gate.py`
- `configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_ps_proto.py`
- `configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_struct_edge.py`
- `configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_struct_gate.py`

## 日志与输出

所有24组使用PrecisionLoggerHook、每50步基础日志和显存，验证每1000步；按实际启用模块输出损失。编号2–8均开启mechanism_logging/fixed_probe_enabled，按实际启用模块分配500步CSV/固定探针/候选图，不为未启用模块计算或伪造响应；编号1仍仅保留基础日志。具体采集矩阵、字段和文件数量见[doris_module_collection_zh.md](doris_module_collection_zh.md)。

24份配置的默认work_dir与批次脚本的`--work-dir`统一为`/root/shared-nvme/work_dirs/doris_stage1/<迁移方向>/<方向>_hrda_3vi<后缀>/`。只调整目录，不改变训练设置，也不移动/合并已有实验结果。例如：

`/root/shared-nvme/work_dirs/doris_stage1/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_struct_gate/`

这些目录用于各组独立日志、验证指标与checkpoint；实际Full的analysis也跟随该目录。本轮没有运行训练或触碰已有实验结果。新脚本完整启用24条命令，不自动跳过已经跑过的1/8和8/8；若复用其已有结果，只手动执行每方向2–7的命令即可。

## 完整执行命令

项目根目录新增`train_doris_mitb5_stage1_ablations_20k.sh`，与以下内容一致。可以逐条执行，也可在云端项目根目录运行`bash train_doris_mitb5_stage1_ablations_20k.sh`。脚本会切换到自身所在目录。

```bash
#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
RUN_ROOT="/root/shared-nvme/work_dirs/doris_stage1"

# Includes all eight experiments per direction; 1 and 8 are existing configurations.

echo "===== goji26RGB2goji30RGB 1/8 Baseline: HRDA + 3VI ====="
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi.py --work-dir "$RUN_ROOT/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi"

echo "===== goji26RGB2goji30RGB 2/8 Physiology ====="
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py --work-dir "$RUN_ROOT/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_phy_gate"

echo "===== goji26RGB2goji30RGB 3/8 Structural ====="
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_struct_gate.py --work-dir "$RUN_ROOT/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_struct_gate"

echo "===== goji26RGB2goji30RGB 4/8 MPARC ====="
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate.py --work-dir "$RUN_ROOT/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate"

echo "===== goji26RGB2goji30RGB 5/8 Prototype ====="
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_ps_proto.py --work-dir "$RUN_ROOT/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_ps_proto"

echo "===== goji26RGB2goji30RGB 6/8 Edge ====="
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_struct_edge.py --work-dir "$RUN_ROOT/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_struct_edge"

echo "===== goji26RGB2goji30RGB 7/8 PBCR ====="
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_pbcr.py --work-dir "$RUN_ROOT/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_pbcr"

echo "===== goji26RGB2goji30RGB 8/8 MPARC + PBCR (Full) ====="
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py --work-dir "$RUN_ROOT/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge"

echo "All goji26RGB2goji30RGB DORIS aligned 20k ablations finished."

echo "===== goji26IRRG2goji30RGB 1/8 Baseline: HRDA + 3VI ====="
python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi.py --work-dir "$RUN_ROOT/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi"

echo "===== goji26IRRG2goji30RGB 2/8 Physiology ====="
python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_phy_gate.py --work-dir "$RUN_ROOT/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_phy_gate"

echo "===== goji26IRRG2goji30RGB 3/8 Structural ====="
python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_struct_gate.py --work-dir "$RUN_ROOT/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_struct_gate"

echo "===== goji26IRRG2goji30RGB 4/8 MPARC ====="
python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_phy_struct_gate.py --work-dir "$RUN_ROOT/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_phy_struct_gate"

echo "===== goji26IRRG2goji30RGB 5/8 Prototype ====="
python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_ps_proto.py --work-dir "$RUN_ROOT/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_ps_proto"

echo "===== goji26IRRG2goji30RGB 6/8 Edge ====="
python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_struct_edge.py --work-dir "$RUN_ROOT/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_struct_edge"

echo "===== goji26IRRG2goji30RGB 7/8 PBCR ====="
python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_pbcr.py --work-dir "$RUN_ROOT/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_pbcr"

echo "===== goji26IRRG2goji30RGB 8/8 MPARC + PBCR (Full) ====="
python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py --work-dir "$RUN_ROOT/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge"

echo "All goji26IRRG2goji30RGB DORIS aligned 20k ablations finished."

echo "===== goji30IRRG2goji26IRRG 1/8 Baseline: HRDA + 3VI ====="
python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi.py --work-dir "$RUN_ROOT/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi"

echo "===== goji30IRRG2goji26IRRG 2/8 Physiology ====="
python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_phy_gate.py --work-dir "$RUN_ROOT/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_phy_gate"

echo "===== goji30IRRG2goji26IRRG 3/8 Structural ====="
python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_struct_gate.py --work-dir "$RUN_ROOT/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_struct_gate"

echo "===== goji30IRRG2goji26IRRG 4/8 MPARC ====="
python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_phy_struct_gate.py --work-dir "$RUN_ROOT/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_phy_struct_gate"

echo "===== goji30IRRG2goji26IRRG 5/8 Prototype ====="
python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_ps_proto.py --work-dir "$RUN_ROOT/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_ps_proto"

echo "===== goji30IRRG2goji26IRRG 6/8 Edge ====="
python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_struct_edge.py --work-dir "$RUN_ROOT/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_struct_edge"

echo "===== goji30IRRG2goji26IRRG 7/8 PBCR ====="
python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_pbcr.py --work-dir "$RUN_ROOT/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_pbcr"

echo "===== goji30IRRG2goji26IRRG 8/8 MPARC + PBCR (Full) ====="
python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge.py --work-dir "$RUN_ROOT/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge"

echo "All goji30IRRG2goji26IRRG DORIS aligned 20k ablations finished."
```

## 必要核查

新增`tools/check_doris_stage1_ablation_configs.py`，使用真实MMCV Config.fromfile核查24个有效配置：模块开关、启用模块的完整参数、去掉未启用结构交互/边界loss后的模型、数据/增强/预训练/优化器/学习率/评估协议与Full一致；同时核对24条命令顺序、输出路径和LF行尾。

本地验收：24个真实MMCV有效配置和24条命令核查通过；只跳过模型custom_imports，未构建网络。Python3.8编译检查和本轮文件空白检查通过。原24配置、既有doris配置和原批次脚本的SHA256与修改前一致。未运行CPU/GPU训练，未连接或同步远端，未commit/push。

云端手动静态复核（不启动训练）：

```bash
python tools/check_doris_stage1_ablation_configs.py
```

本地缺少完整模型依赖时可加`--skip-custom-imports`，仅跳过配置custom_imports的模型导入；这不跳过配置解析与参数比对，也不能替代GPU运行验收。

目录迁移验证：24份迁移前后有效配置逐项比对，除work_dir外完全一致；90份旧版无前缀配置SHA256未变。原位置不保留重复doris_配置，新目录每方向严格8份。
