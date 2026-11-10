# H-Phy 两份调度指令合并验收

## 最终参数与验收边界

本报告合并核查《H-Phy 门控调度提前启用及离线机制响应同步修订》和《H-Phy 门控调度最终调整为 4000 Iter》。最终参数采用补充指令。`4000` 表示门控完全生效节点，训练总长度为 `15000`。原指令中 3000 Iter 调度表的数值由 4000 Iter 公式替代。

```python
phy_gate_schedule = 'early_nonzero_linear'
phy_gate_initial_strength = 0.10
phy_gate_full_iter = 4000
phy_min_weight = 0.75
max_iters = 15000
```

```text
rho(t) = 0.10 + 0.90 * clip(t / 4000, 0, 1)
G_phy = 1 - 0.25 * rho(t) * (1 - S_phy)
A_phy = 1 - G_phy
W_after = W_before * G_phy
```

执行环境为本地 Windows、Python 3.8.20、PyTorch 2.3.0 CPU、MMCV 1.4.0。工作基于分支 `codex/hphy15k` 和现有未提交重构，HEAD 为 `ef089d4dc036b4b1ac3069d5b4e6964734be3da5`。本次执行了短步合成数据训练测试。正式 15000 Iter 训练由作者后续手动执行。

## 两份指令逐项核查

| 要求 | 实现与测试证据 | 结果 |
|---|---|---|
| Early Non-Zero Linear Ramp-up | `phy_gate_strength` 统一计算 rho，公共早期调度默认 full_iter 为 4000 | 已实现 |
| H-Phy 真实训练权重应用 | `_apply_phy_gate` 返回实际前后权重，H-Phy 分支先于 Legacy 提前返回逻辑执行 | CPU 完整模型通过 |
| Goji/Vegetation 类别条件作用域 | Teacher 0/1 与有效区域控制门控，非植被门控为 1 | 单元测试及非植被 Smoke 通过 |
| 三方向九组配置 | Physiology、MPARC、Full 均显式指定 0.10、4000、0.75、15000 | 九组解析通过 |
| 八组消融独立性 | Baseline、Structural、Prototype、Edge、PBCR 的原配置文本一致性与模块开关检查 | 通过 |
| 双头结构和辅助损失 | 修改前快照 AST 对照验证两个 Head、源域损失、目标域 BCE 函数一致 | 通过 |
| 源域双头梯度 | Iter 0 的真实完整模型反向，两个源域 Head 均有非零梯度 | 三组通过 |
| 目标辅助监督 | Iter 0 即有正的高置信目标植被 BCE，目标直接 GV 梯度隔离单测 | 通过 |
| detach 与单次门控 | 分割加权路径对 H-Phy 无梯度，检查 W_after、A 和 ClassMix 实收权重 | 通过 |
| EMA、ClassMix、MPARC、PBCR 接口 | `forward_train`、EMA 更新与相关结构/边缘方法的 AST 一致，完整模型执行正常 | 通过 |
| 训练日志 | 保留 loss_phy_src、loss_phy_tgt，新增 phy_gate_iter、phy_gate_rho、phy_gate_active_ratio、phy_S_mean_active、phy_A_mean_active、phy_A_max_active | 最小日志过滤测试通过 |
| Checkpoint 参数追溯 | 训练 meta 保存实际 phy_cfg 与规范化 phy_gate，Checkpoint 节点规划读取 full_iter | 通过 |
| 断点恢复 local_iter | 原恢复逻辑读取已完成迭代，新门控在恢复值 2000 时得到 rho=0.55 | 恢复与门控单测通过 |
| 历史调度回放 | 恢复保存配置中的调度字段，记录 requested/effective 参数与历史标识 | 3K、延迟 6K/4K、Legacy 测试通过 |
| 新实验与旧权重隔离 | 新训练拒绝不同调度的 resume_from/load_from，目录使用 hphy_early4k | 通过 |
| 回放 Iter 可靠性 | completed_iter、规范编号、经身份核验的最佳记录、同字节编号别名及显式 Iter | 缺失/冲突测试通过 |
| 训练与回放公式一致 | 同一个 hierarchical_phy_gate，核对 A=1-W_after/W_before | 配对回放测试通过 |
| 机制元数据 | phy_mechanism_meta.json 保存全部九个指定字段与空作用域原因，保留兼容文件 | 通过 |
| 独立掩膜 | Teacher 植被、正基础权重、二者交集、实际衰减区域单独保存 | 掩膜及数值检查通过 |
| 1×5 与独立图 | 固定 S 色标 0–1、A 色标 0–0.25，256 原网格与中心 196、ROI、clean/labeled | 渲染测试及合成图目视检查通过 |
| Bash 启动入口 | 八组平铺命令，单组复制对应 Python 命令，日志和目录检查由训练入口执行 | 平铺脚本专项 9 项测试通过 |
| 主干和训练协议 | MiT-B5、六通道、相同划分/种子/优化器/学习率、15000 Iter | 八组及三方向审计通过 |
| GPU 前后向 | 本地 PyTorch 为 CPU 构建，用户指定仅本地开发并手动上传 | 待作者在 GPU 环境验收 |
| 真实数据完整回放 | 既有真实 TIFF 测试缺少 DORIS_PROBE_DATA_ROOT | 待真实数据验收 |

## 调度实际输出

固定 `S_phy=0.4`。表中 A 为理论值，实际 float32 输出的绝对误差小于 `1e-6`。

| Iter | 实际 rho | A_phy=0.15 rho |
|---|---|---|
| 0 | 0.1000 | 0.01500 |
| 1000 | 0.3250 | 0.04875 |
| 2000 | 0.5500 | 0.08250 |
| 3000 | 0.7750 | 0.11625 |
| 4000 | 1.0000 | 0.15000 |
| 7000 | 1.0000 | 0.15000 |
| 10000 | 1.0000 | 0.15000 |
| 15000 | 1.0000 | 0.15000 |

单元测试还覆盖原指令中的 500、1500、2500 Iter，在新公式下分别为 0.2125、0.4375、0.6625。参数边界、空组、无正基础权重、ignore、非植被与零权重场景均在测试范围内。[实际数值输出](D:/PSSAL-DA/outputs/hphy_early4k_schedule.json)

## 八组 Table IV 配置

方向为 `goji26RGB2goji30RGB`。所有配置均采用 MiT-B5、六通道、seed 0、相同数据划分和 15000 Iter。表中开关顺序为 Phy / Structural / Prototype / Edge。

| 编号 | 实验 | 开关 | H-Phy 完整生效 Iter | 配置 |
|---|---|---|---|---|
| 1/8 | Baseline | 0 / 0 / 0 / 0 | — | [Baseline](D:/PSSAL-DA/configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi.py) |
| 2/8 | Physiology | 1 / 0 / 0 / 0 | 4000 | [Physiology](D:/PSSAL-DA/configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py) |
| 3/8 | Structural | 0 / 1 / 0 / 0 | — | [Structural](D:/PSSAL-DA/configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_struct_gate.py) |
| 4/8 | MPARC | 1 / 1 / 0 / 0 | 4000 | [MPARC](D:/PSSAL-DA/configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate.py) |
| 5/8 | Prototype | 0 / 0 / 1 / 0 | — | [Prototype](D:/PSSAL-DA/configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_ps_proto.py) |
| 6/8 | Edge | 0 / 0 / 0 / 1 | — | [Edge](D:/PSSAL-DA/configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_struct_edge.py) |
| 7/8 | PBCR | 0 / 0 / 1 / 1 | — | [PBCR](D:/PSSAL-DA/configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_pbcr.py) |
| 8/8 | Full | 1 / 1 / 1 / 1 | 4000 | [Full](D:/PSSAL-DA/configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py) |

| 方向 | Physiology | MPARC | Full | 总 Iter |
|---|---|---|---|---|
| goji26RGB2goji30RGB | 4000 | 4000 | 4000 | 15000 |
| goji26IRRG2goji30RGB | 4000 | 4000 | 4000 | 15000 |
| goji30IRRG2goji26IRRG | 4000 | 4000 | 4000 | 15000 |

继承解析还核对 Structural 特征交互、StructEdgeLoss、Prototype 阈值、损失系数和实际输出路径。三方向 24 组的原有审计入口也已适配当前 H-Phy 定义。[配置解析结果](D:/PSSAL-DA/outputs/hphy_early4k_config_audit.json) [24 组审计输出](D:/PSSAL-DA/outputs/hphy_early4k_all24_audit_v2.txt)

## 测试结果

- 全量回归为 **186 passed、1 skipped**，耗时 215.03 秒。唯一跳过项是缺少真实数据根目录的既有 TIFF 测试。[最终测试日志](D:/PSSAL-DA/outputs/hphy_early4k_verified_tests.txt)
- H-Phy、配对回放、Checkpoint 与启动脚本专项测试为 48 passed。[专项日志](D:/PSSAL-DA/outputs/hphy_early4k_targeted_v2.txt)
- Python 3.8 静态编译覆盖 27 个修改文件，Bash 两个入口语法检查及 git diff --check 通过。[静态检查](D:/PSSAL-DA/outputs/hphy_early4k_static_checks.txt)
- 早期启动器的 all --check 解析记录保留作为该阶段验收证据。当前平铺脚本通过独立的 `python tools/check_doris_hphy_configs.py --all-directions` 检查配置。[早期检查记录](D:/PSSAL-DA/outputs/hphy_early4k_launcher_check.txt)
- 完整 CPU Smoke Test 的三组分别执行 Iter 0、4000、10000。每组均通过真实 MiT-B5/HRDA/DACS 前向、源域/混合/辅助损失、反向、优化器更新、有限值检查、双头梯度、ClassMix 权重、Checkpoint/优化器往返、只读机制导出与 Student 验证。[完整 Smoke 报告](D:/PSSAL-DA/outputs/hphy_early4k_full_cpu_smoke_v2/report.json)

| 模型 | 总参数量，含 EMA | H-Phy 参数量 | Iter 0 A 均值，作用像素 | Iter 4000 A 均值，作用像素 |
|---|---|---|---|---|
| Physiology | 171381606 | 5202 | 0.019725 | 0.165049 |
| MPARC | 171512678 | 5202 | 0.019675 | 0.196529 |
| Full | 171512678 | 5202 | 0.019675 | 0.196529 |

Smoke 使用随机初始化，并在测试进程内清空预训练路径。高置信 Teacher 通过确定性分类器参数构造，Iter 0 的 Student 分类器同步设置，以适配 DACS 首步复制 EMA 的实际行为。真实 Teacher 前向、EMA、ClassMix 及辅助损失均执行。Iter 10000 构造非植被预测，其 H-Phy 活跃比例和 A 均值为 0。图像与这些数值均属于合成验收数据。

合成 1×5 图在 rho=0.55 时经过共享门控生成，完成 PNG/PDF、独立图、掩膜与 196 中心裁剪的目视检查。[合成预览](D:/PSSAL-DA/outputs/hphy_early4k_render_qa/rendered/phy_mechanism_1x5.png)

现有真实 TIFF 用例沿用数据环境变量条件。GPU 和真实数据项的验收结果应在作者上传后补充。

## 兼容调整与历史保护

新实验输出根目录为 `/root/shared-nvme/work_dirs/doris_hphy_early4k/`。八组启动目录使用原 stage1 脚本的实验名称，放在该根目录的 `goji26RGB2goji30RGB/` 下。训练入口在各实验目录内记录独立日志。当前 `hphy_early4k` 标识描述门控节点，训练协议为 15K。

历史 3K H-Phy 回放恢复 full_iter=3000。更早的 H-Phy 恢复保存的 6000/4000 延迟调度。历史单头和 20K 回放由通用 replay 入口按保存配置处理，配对 1×5 入口保持既定 15K 协议验证。源 checkpoint 和只读回放状态的 hash 前后相同。

MMCV 1.4 的 ConfigDict 需要在恢复调度时保留容器类型，已对该兼容点增加测试。历史配置审计工具的旧生理参数断言已按实际 H-Phy 调用链更新，归档启动脚本按文本核对，新脚本单独检查 LF 行尾。归档 20K 命令保留供来源追溯，新实验使用下述专用入口。

当前五个 phy-disabled 组的配置一致性测试通过。与本轮修改前快照比较，H-Phy 两个 Head、源域/目标域辅助损失及 DACS forward_train、EMA、相关结构/边缘方法的 AST 相同。训练门控的新日志和权重快照用于验证单次加权与实际作用域。

新 checkpoint 节点在 full_iter=4000 附近规划，继承的 Structural、Prototype、Edge 节点仍取各自配置。训练入口在写配置或日志前检查旧目录占用和调度不一致。历史 `doris_stage1`、3K 和旧 15K 实验文件保留在各自路径。

## 作者手动执行命令

脚本路径为 [run_doris_tableiv_26rgb_to_30rgb_hphy15k.sh](D:/PSSAL-DA/scripts/run_doris_tableiv_26rgb_to_30rgb_hphy15k.sh)。以下命令在手动上传后的 Linux 项目根目录执行。

```bash
# 检查八组
python tools/check_doris_hphy_configs.py --all-directions

# 单组检查或运行，例如 Physiology
python tools/check_doris_hphy_configs.py --experiment 2
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py --work-dir "/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_phy_gate"

# 八组顺序运行
bash scripts/run_doris_tableiv_26rgb_to_30rgb_hphy15k.sh

# 独立 GPU 短步验收，输出应为全新目录
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tools/smoke_doris_hphy.py --device cuda:0 --experiments 2 4 8 \
  --output /root/shared-nvme/work_dirs/doris_hphy_early4k/gpu_smoke_01
```

离线机制脚本为 [run_hphy_mechanism_26rgb2_30rgb.sh](D:/PSSAL-DA/scripts/run_hphy_mechanism_26rgb2_30rgb.sh)。默认路径和六张目标图列表集中定义在 tools/replay_hphy_mechanism.py 顶部，可用 `python tools/replay_hphy_mechanism.py --checkpoint iter_10000.pth` 指定 H-Phy 权重，逐图独立完成响应保存与绘图。Baseline 默认使用对应训练目录内的 latest.pth，可通过 --checkpoint-baseline 覆盖。输出放入带时间戳的新目录，每张图具有独立子目录和统计。

```bash
bash scripts/run_hphy_mechanism_26rgb2_30rgb.sh

# 迭代证据不足时才补充 --iter N / --iter-baseline N
# 可选，已有数组的独立重绘
python tools/build_hphy_mechanism_figure.py --input-npz "/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/hphy_mechanism_26rgb2_30rgb/responses_full_256.npz" --meta-json "/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/hphy_mechanism_26rgb2_30rgb/phy_mechanism_meta.json" --output-dir "/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/hphy_mechanism_26rgb2_30rgb" --render-only --export-group --export-panels --export-full256
```

## 文献与设计依据

Mean Teacher、Temporal Ensembling 为逐步引入辅助监督提供训练原则。DACS 与 DAFormer 对应本项目的 Teacher、混合伪监督和 UDA 稳定训练基础。0.10 与 4000 是当前工程设置，参数效果由新 15K 实验验证。[Mean Teacher](https://arxiv.org/abs/1703.01780) [Temporal Ensembling](https://arxiv.org/abs/1610.02242) [DACS](https://openaccess.thecvf.com/content/WACV2021/html/Tranheden_DACS_Domain_Adaptation_via_Cross-Domain_Mixed_Sampling_WACV_2021_paper.html) [DAFormer](https://openaccess.thecvf.com/content/CVPR2022/html/Hoyer_DAFormer_Improving_Network_Architectures_and_Training_Strategies_for_Domain-Adaptive_Semantic_CVPR_2022_paper.html)

## 两份指令累计修改文件

文件清单以本轮调度任务前的本地快照为基准，覆盖此前 Early Ramp-up 整合及最终 4K 修订。仓库中更早重构留下的其他未提交修改单独保留。[本轮关键 diff](D:/PSSAL-DA/outputs/hphy_early4k_schedule_task.patch)

- [configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_phy_gate.py](D:/PSSAL-DA/configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_phy_gate.py)
- [configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_phy_struct_gate.py](D:/PSSAL-DA/configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_phy_struct_gate.py)
- [configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py](D:/PSSAL-DA/configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py)
- [configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py](D:/PSSAL-DA/configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py)
- [configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate.py](D:/PSSAL-DA/configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate.py)
- [configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py](D:/PSSAL-DA/configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py)
- [configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_phy_gate.py](D:/PSSAL-DA/configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_phy_gate.py)
- [configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_phy_struct_gate.py](D:/PSSAL-DA/configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_phy_struct_gate.py)
- [configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge.py](D:/PSSAL-DA/configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge.py)
- [docs/hphy_design_and_evidence.md](D:/PSSAL-DA/docs/hphy_design_and_evidence.md)
- [docs/hphy_early4k_acceptance.md](D:/PSSAL-DA/docs/hphy_early4k_acceptance.md)
- [docs/hphy_phy_figure_protocol.md](D:/PSSAL-DA/docs/hphy_phy_figure_protocol.md)
- [docs/hphy_tableiv_alignment.md](D:/PSSAL-DA/docs/hphy_tableiv_alignment.md)
- [mmseg/apis/train.py](D:/PSSAL-DA/mmseg/apis/train.py)
- [mmseg/models/uda/dacs.py](D:/PSSAL-DA/mmseg/models/uda/dacs.py)
- [mmseg/models/uda/phy_reliability.py](D:/PSSAL-DA/mmseg/models/uda/phy_reliability.py)
- [mmseg/utils/mechanism.py](D:/PSSAL-DA/mmseg/utils/mechanism.py)
- [mmseg/utils/precision_logger.py](D:/PSSAL-DA/mmseg/utils/precision_logger.py)
- [scripts/run_doris_tableiv_26rgb_to_30rgb_hphy15k.sh](D:/PSSAL-DA/scripts/run_doris_tableiv_26rgb_to_30rgb_hphy15k.sh)
- [scripts/run_hphy_mechanism_26rgb2_30rgb.sh](D:/PSSAL-DA/scripts/run_hphy_mechanism_26rgb2_30rgb.sh)
- [tests/test_doris_checkpoint_policy.py](D:/PSSAL-DA/tests/test_doris_checkpoint_policy.py)
- [tests/test_doris_stage1.py](D:/PSSAL-DA/tests/test_doris_stage1.py)
- [tests/test_hphy.py](D:/PSSAL-DA/tests/test_hphy.py)
- [tests/test_hphy_launcher.py](D:/PSSAL-DA/tests/test_hphy_launcher.py)
- [tests/test_hphy_mechanism.py](D:/PSSAL-DA/tests/test_hphy_mechanism.py)
- [tools/build_hphy_mechanism_figure.py](D:/PSSAL-DA/tools/build_hphy_mechanism_figure.py)
- [tools/check_doris_hphy_configs.py](D:/PSSAL-DA/tools/check_doris_hphy_configs.py)
- [tools/check_doris_stage1_ablation_configs.py](D:/PSSAL-DA/tools/check_doris_stage1_ablation_configs.py)
- [tools/doris_checkpoint_plan.py](D:/PSSAL-DA/tools/doris_checkpoint_plan.py)
- [tools/replay_doris_mechanisms.py](D:/PSSAL-DA/tools/replay_doris_mechanisms.py)
- [tools/replay_hphy_mechanism.py](D:/PSSAL-DA/tools/replay_hphy_mechanism.py)
- [tools/smoke_doris_hphy.py](D:/PSSAL-DA/tools/smoke_doris_hphy.py)
- [tools/train.py](D:/PSSAL-DA/tools/train.py)
