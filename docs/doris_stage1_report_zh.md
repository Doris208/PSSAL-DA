# DORIS-PSSAL-DA 第一阶段：实现、核验与运行说明

> 当前日志与机制采集以 [最终整合版工程交付](doris_mechanism_collection_zh.md) 为准；下文保留此前方案的算法核查与阶段记录。

> 2026-11-05 后续更新：Full 的融合、tau_proto 和过程日志设置以 [论文代码日志对照](doris_stage1_paper_alignment_zh.md) 为准；下文保留此前阶段记录。

日期：2026-11-05。代码基线：本地 `doris`，HEAD `8b587700c012d14cbccaddcc94ffd6662ab7468c`，与此前推送的 `origin/doris-01` 提交一致。本轮开始时工作区干净；未连接 GPU、安装依赖、执行完整训练、commit 或 push。

本轮原位修改 8 个源码文件，新增 6 个实验配置、配置核验工具、独立响应导出工具、六组批次脚本、单元测试及本报告。原有实验配置、公共配置工具、批次脚本和历史数据未改写。

后续补充：用户另指定 MemoryAdaptNet＋MiT-B5＋3VI（26IRRG→30RGB）和 DAFormer＋MiT-B5＋3VI（26RGB→30RGB），当前总脚本已扩展为八组，配置核验工具检查 32 个配置。原六组正在运行或已经完成时，使用 `train_doris_mitb5_stage1_comparisons_20k.sh` 只补两组。详见 [新增两组说明](D:/PSSAL-DA/docs/doris_stage1_comparisons_zh.md)。下文的六组设计与初次测试记录保留为原阶段记录。

## 1. 最终有效调用关系

既有配置的有效链路为：各方向最后生效的 `_base_` 和赋值 → `pssalda_*_ablation_common.py` 与 `pssalda_26rgb2_30rgb_ablation_utils.py` → `make_model_override/make_uda_override` → DACS。未按文件前半段的失效设置判断。

三个方向为 T1：30IRRG→26IRRG；T2：26IRRG→30RGB；T3：26RGB→30RGB。每个方向既有 8 组的最终开关均通过 Python 配置执行/继承的静态检查：

| 配置后缀 | phy | struct | proto | edge | 默认算法 |
|---|---:|---:|---:|---:|---|
| 无后缀（HRDA） | 0 | 0 | 0 | 0 | legacy |
| `_phy_gate` | 1 | 0 | 0 | 0 | legacy |
| `_struct_gate` | 0 | 1 | 0 | 0 | legacy |
| `_phy_struct_gate` | 1 | 1 | 0 | 0 | legacy |
| `_ps_proto` | 0 | 0 | 1 | 0 | legacy |
| `_struct_edge` | 0 | 0 | 0 | 1 | legacy |
| `_pbcr` | 0 | 0 | 1 | 1 | legacy |
| `_proto_struct_edge` | 1 | 1 | 1 | 1 | legacy |

**本地没有 MMCV，真正的 `Config.fromfile` 检查尚未执行，不能把静态检查当作该项通过。** 新增 `tools/check_doris_stage1_configs.py` 使用真正的 MMCV `Config.fromfile` 核验旧 24 组与新 6 组共 30 个配置；同步后应先在现有 hrda 环境运行它。测试也保留了这项检查，本地明确跳过。

## 2. MPARC：新旧 R_struct

默认 `fusion_mode='legacy_product'`，保留旧乘积：

```text
R_struct = clamp(R_conf^confidence_power
                 × R_continuity^continuity_power
                 × R_adj^adjacency_power
                 × R_sem^semantic_power, 0, 1)
```

公共配置中的对应 power 仍为 1.0、1.0、0.8、0.5。

只有新 Full 指定 `fusion_mode='doris_confidence_guided'`：

```text
全部先验可用：
R_struct = R_conf × (0.40 R_continuity + 0.30 R_adj + 0.30 R_sem)

部分先验不可用：
R_struct = R_conf × Σ(w_i × available_i × R_i) / Σ(w_i × available_i)
```

权重按像元及其预测类别的先验初始化状态归一化。adjacency 必须对应源域类别行已经初始化，semantic 必须对应源域概率语义原型已经初始化；关闭的先验、非有限响应不参与。默认返回 1 的未初始化响应不会作为证据计入均值。没有有效分量时返回 0，输出限制在 [0,1]，非有限输入被安全排除/清理。

四个原有响应的计算、邻域内 ignore 处理、源域邻接/概率语义 EMA 更新均保留；DORIS 从 DACS 接收已计算的 Teacher confidence，不重复求最大概率。旧模式继续走原始计算分支。

结构类别范围仍为 2/3/4；伪监督门控、原有特征交互位置、`gamma_struct=0.05`、6000/4000 warmup/rampup 均未调整。新配置没有第五个结构响应、额外网络层或可学习融合系数。

## 3. PBCR：新旧 R_edge 与 R_pair

默认 `edge_fusion_mode='legacy'`，原 `paper_pbcr` 计算保留。公共配置旧系数为 0.4/0.4/0.2；旧 Source 缺少概率/类别对分量时，原实现相当于 `0.4 E_img`。旧 R_pair 的非交界默认行为也保留在 legacy 中。

新 Full 指定 `edge_fusion_mode='doris_reliable'`，仍使用 `edge_weight_mode='paper_pbcr'`：

```text
Target 全部分量可用：R_edge = 0.45 E_prob + 0.35 E_img + 0.20 R_pair
Source 只有影像分量：R_edge = E_img
Target 类别对先验未就绪：R_edge = (0.45 E_prob + 0.35 E_img) / 0.80
```

各分量按实际可用状态归一化。`E_prob` 来自 EMA Teacher 类别概率梯度；新模式 `E_img` 只使用基础前三通道，VI 三通道不参与影像边界梯度。legacy 保留其原有通道计算。

新 R_pair：

- 保留原源域类别对矩阵及 EMA 更新，没有目标域 GT 输入。
- 仅使用原先的上下左右四邻域；邻居必须有效且异类，不跨图像边界 wrap，不将 255 当类别。
- 对每个像元的有效类别对分数 `prior[pair]^power` 求均值；未见类别对的源域先验分数为 0，不设置可靠性下限。
- 非交界、ignore、未初始化全局先验的响应均为 0。
- 保留类别对 warmup 时长，以从 0 到 1 的 ramp 缩放真实交界响应，不从默认 1 开始。

Target 边界损失权重单独采用：

```text
reliability = detach(clamp((Teacher confidence - 0.60) / 0.35, 0, 1))
edge_weight = detach(R_edge × reliability × spatial_valid × lambda_edge × warmup)
```

未再次乘入全局语义伪标签权重；不修改普通语义分割的 pseudo_threshold、pseudo_weight 或其计算。Source 不额外进行模型前向，边界响应仍来自已有影像。

## 4. Boundary Loss：正负分开归一化

默认 `boundary_loss_mode='legacy'`：旧代码和类名、入口保留；旧 BCE 主要在有效正边界上计算，原 CE/Dice 分支不变。

新 Full 使用 `boundary_loss_mode='doris_balanced'`，配置 `bce_weight=1`、`ce_weight=0.1`、`dice_weight=0`、外部 `loss_weight=0.1`：

```text
L = loss_weight × (0.75 L_positive_BCE
                 + 0.25 L_local_negative_BCE
                 + 0.10 L_boundary_CE)
```

正样本是原 `label_to_edge(edge_radius=1)` 得到的有效边界；负样本是距有效正边界三像素范围内的非边界像元，使用 7×7 pooling 表达 Chebyshev 距离，不扩大 R_pair 邻域。ignore 和无效区域均不参与正负样本计数。

正样本权重为已有 `edge_weight`；局部负环带通过 7×7 max pooling 传播邻近有效正边界权重，再限制在负环带和有效支持区域。它不会因为环带本来没有边界响应而始终为零。

两个 BCE 分母分别使用有效正/负样本数量，CE 使用正边界数量；不除以权重和。`lambda_edge`、warmup、`StructEdgeLoss.loss_weight` 因而均保留实际线性缩放作用，没有重复乘入。没有正/负样本、全 ignore、零权重时均可安全返回有限值和零梯度。

### 为什么还修改了 HRDA 的四个入口文件

仅将权重置 0 不够：负环带权重传播可能重新填入被裁剪的区域或 ClassMix 拼接缝。因此增加可选 `struct_edge_valid_mask`，经过 segmentor → decode head → HR 分支裁剪 → StructEdgeLoss 传递。

该掩膜只约束新边界辅助损失；语义 Loss 不读取它。ClassMix 使用同一 domain mask 混合 source/target 有效区域，并去掉拼接缝附近的边界标签/负环带感受区域。HR 分支按同一 crop box 裁剪，缩放使用最近邻。

这些参数均追加在旧参数之后，默认 None；旧模式不传有效掩膜，旧计算保持。没有增加网络层或修改滑窗推理。

## 5. 六个新增配置与候选参数

所有文件位于 `D:\PSSAL-DA\configs\hrda\`，仅保留最终有效的公共 base 和赋值；不复制整套 common 或算法源文件。

| 方向 | 新 HRDA | 新 Full |
|---|---|---|
| T1 | `doris_goji30IRRG2goji26IRRG_hrda_3vi.py` | `doris_goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge.py` |
| T2 | `doris_goji26IRRG2goji30RGB_hrda_3vi.py` | `doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py` |
| T3 | `doris_goji26RGB2goji30RGB_hrda_3vi.py` | `doris_goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py` |

三个新 HRDA 的模型、UDA、数据、优化器、学习率、runner、checkpoint、evaluation、fp16、seed、crop 参数与各自旧 HRDA 的最终有效设置相同，四模块均关闭。新 Full 四模块均开启，joint、Spatial Layout、Neighbor Patch、Spectral Boundary、额外 confidence/prototype boundary gate 继续关闭。

| Full 参数 | T1 | T2 Safe | T3 |
|---|---:|---:|---:|
| target_update_start | 8000 | 8000 | 3000 |
| weight_start / loss_start | 12000 | 14000 | 6000 |
| lambda_proto_tgt | 0.0003 | 0.0002 | 0.001 |
| proto_min_weight | 0.95 | 0.98 | 0.88 |
| proto_power | 0.35 | 0.35 | 0.70 |
| loss_weight_src | 0.006 | 0.004 | 0.010 |
| loss_weight_tgt | 0.003 | 0.002 | 0.006 |
| StructEdgeLoss.loss_weight | 0.1 | 0.1 | 0.1 |

T2 还保留 Safe 的 `target_weight_thr=0.30` 和 `boundary_pair_min_weight=0.80` 等其余设置；后者在新 R_pair 中不再作为非交界默认高值使用。

这些是第一轮候选值，没有声称最优或承诺类别精度提高。其余 MPARC/Prototype 公共参数保留。MiT-B5、6 通道、crop=256×256、20K、batch=2、seed=0、lr=6e-5、优化器、pseudo_threshold=0.968、目标区域裁剪、标签顺序和评价协议保持。

当前24份HRDA消融配置位于`configs/hrda/doris/<迁移方向>/`，默认输出与批次脚本的`--work-dir`一致：`/root/shared-nvme/work_dirs/doris_stage1/<迁移方向>/<配置名去掉doris_前缀>/`。旧输出结果未移动，不建立batch时间戳层。

## 6. 修改文件清单

路径均相对于 `D:\PSSAL-DA`。

| 原位修改文件 | 位置和原因 |
|---|---|
| `mmseg/models/uda/struct_prior_gate.py` | 构造增加兼容默认模式；`fuse_doris_responses/apply` 实现新融合、先验可用性与缓存置信度。原响应与 EMA 更新函数未改。 |
| `mmseg/models/uda/dacs.py` | 新模式判断、R_pair、可用分量融合、Teacher 平滑可靠性、ClassMix 支持掩膜、按需标量。保留所有 legacy 分支。 |
| `mmseg/models/losses/struct_edge_loss.py` | 模式开关与 `_doris_balanced_loss`，旧 forward 主体保留。 |
| `mmseg/models/segmentors/hrda_encoder_decoder.py` | 训练入口追加并传递可选边界有效掩膜。 |
| `mmseg/models/segmentors/encoder_decoder.py` | 仅在提供掩膜时向 decode head 传递。 |
| `mmseg/models/decode_heads/decode_head.py` | 可选掩膜继续传递，仅送入边界 Loss，语义 Loss 不使用。 |
| `mmseg/models/decode_heads/hrda_head.py` | 各分支传递掩膜，HR crop 使用原 crop box 对齐。 |
| `mmseg/utils/precision_logger.py` | 允许显式模型侧采样产生的 `doris.*` 标量通过 minimal 过滤；其余日志与 Acc 格式保留。 |

新增：上表六个配置；`tools/check_doris_stage1_configs.py`；`tools/export_doris_response_maps.py`；`train_doris_mitb5_stage1_20k.sh`；`tests/test_doris_stage1.py`；本报告。

没有修改原 24 个配置、三个 common、原 ablation utils、旧批次脚本、Prototype 算法、生理头、Teacher/Student EMA 更新函数、初始化预训练路径、数据集/标签处理、正式 test/evaluation 工具或历史文件。未创建 `doris_dacs.py` 等源码双版本。

## 7. 可选过程统计

默认 `uda.doris_process_logging=False`。明确设置 True 后，按 `uda.doris_log_interval=50` 采样当步统计，送入原有 minimal logger 的 `stats` 行及 JSON。它不是每 50 步重新做一次完整验证，也不是额外前向，更不是 50 步窗口均值。

| 字段 | 分母或阶段 |
|---|---|
| `doris.teacher_spatial_valid_ratio` | 空间有效像元 / 全图像元。 |
| `doris.teacher_high_conf_ratio_valid` | 空间有效区域中 confidence≥0.968 的比例。 |
| `doris.teacher_goji_prediction_ratio_valid`、vegetation 对应字段 | Teacher 预测该类的有效像元 / 全部空间有效像元。 |
| `doris.teacher_goji_high_conf_ratio_given_class`、vegetation 对应字段 | 有效且预测为该类的像元中，高置信度的比例。 |
| `doris.goji_final_pseudo_weight_mean_valid`、vegetation 对应字段 | MPARC 与 Prototype 加权之后、ClassMix 之前，该类有效像元的最终语义伪标签权重均值。 |
| `doris.target_prototype_valid_classes` | PBCR 特征 Prototype 的 Target bank 已初始化类别数。 |
| `doris.R_struct_mean_all` | 原始融合响应全图均值，未乘特征交互 gamma 或 warmup。 |
| `doris.R_struct_mean_struct_classes_valid` | 同一响应在结构类别 2/3/4 且空间有效区域内的均值。 |
| `doris.R_edge_src/tgt_mean_all` | 原始边界融合响应全图均值，未乘 confidence、lambda、warmup。 |
| `doris.R_edge_src/tgt_mean_valid_boundary` | 同一响应在有效边界区域内的均值。 |
| `doris.valid_boundary_ratio_src/tgt` | 有效边界像元 / 全图像元。 |

相应类别或区域为空时区域均值/条件比例为 0，须结合该类预测比例解读。新 DORIS 不再输出含糊的 `teacher_pseudo_label_retain_ratio`；legacy 的历史字段与定义保留。

`R_sem` 是源域概率向量语义一致性先验；它不同于 PBCR 的特征 Prototype。两类原型不混用。

## 8. 本地验证证据与未执行项

| 项目 | 结果与边界 |
|---|---|
| 编译、`git diff --check` | 通过。 |
| 配置静态执行 | 30 个最终配置通过；旧 24 开关核验、新六组参数和目录核验通过。真正 MMCV 解析仍待远端。 |
| 旧配置/脚本保护 | 对修改前快照逐文件 SHA256 比较，既有所有配置和 `.sh` 保持字节一致；Git index 与 HEAD 保持原值。 |
| CPU 单元测试 | 包含原有测试共 36 项，33 通过、3 跳过。现有 Torch CPU 环境为 `D:\Anaconda3\envs\aglldiff\python.exe`，未安装包。 |
| 跳过项 | 两项真正 MMCV 配置测试（原套件一项、新套件一项）；该 CPU 环境缺少 Matplotlib 的一项布局测试。 |
| 独立布局测试 | 在已有默认 Python/Matplotlib 环境单独运行真实 `save_strip` 函数，NumPy 测试夹具渲染两幅横向图通过，并查看排版。仅测试布局，不是训练模型响应结果。 |
| 旧数值兼容 | 旧结构响应/先验状态、边界 Loss 值和梯度与快照一致；旧 DACS 用小型网络 stand-in 完成训练步后，新旧参数及状态逐项精确相同（rtol=0、atol=0）。不等于完整 HRDA/GPU 训练已验收。 |
| 新训练步 | 新 DORIS 的真实 DACS 控制流配合小型 CPU 网络完成前向、所有启用损失的 backward、AdamW 更新；值/参数有限，有效掩膜经 ClassMix 传递。 |
| 融合/损失测试 | 初始化可用性、NaN/Inf 排除、ignore 邻域、R_pair 均值/非交界/不 wrap、Source 单分量、VI 排除、平滑可靠性独立于全局 pseudo weight、负环带有效梯度、空正负区域、权重线性缩放均通过。 |
| 裁剪/导出测试 | HR crop 的掩膜与标签对齐；导出只调用一次 Teacher 预测，恢复先验并保持模型状态不变；缺失/未初始化先验报错。 |
| Bash | Windows Git Bash 无法启动（0xC0000142）；脚本字节检查为 LF，六条命令顺序/路径可核验。`bash -n` 待远端。 |
| GPU 与真实响应 | 未执行 4090 训练、真实 MiT-B5 完整 forward、真实 checkpoint 响应导出或精度比较；未承诺加速、显存下降或类别性能提升。 |

本地快照在被忽略的 `outputs/doris_stage1_before.json`。它不属于新增源码双版本，不会自动同步成训练依赖；远端无快照时相关新旧比较测试会明确跳过。

当前 DACS 的 `local_iter` 仍为既有普通 Python 计数，本轮没有重写恢复训练逻辑。新实验使用独立目录开始；不要把直接续跑旧实验当作完整的恢复一致性验证。

## 9. 同步后先检查，再运行

本轮涉及多个模型入口，请同步完整改动文件集合，不能只同步六个配置或 dacs.py。无需重建环境。

```bash
conda activate hrda
cd /root/HRDA-master
CHECK_ROOT=$(mktemp -d /root/shared-nvme/work_dirs/doris_stage1_check_$(date +%Y%m%d_%H%M%S)_XXXXXX)

python -m compileall -q tools mmseg configs/hrda
bash -n train_doris_mitb5_stage1_20k.sh
python tools/check_doris_stage1_configs.py --json "$CHECK_ROOT/config_audit.json"
python -m unittest discover -s tests -p 'test_*.py' -v
```

配置审计应打印旧 24 组、新 6 组的有效模块开关及融合模式，并输出 PASS。若失败，先核对同步文件，保留错误信息。

### 六个配置各做 5 Iter 冒烟

以下只在手动执行时运行 30 个短训练步，不执行完整 20K：

```bash
for direction in goji30IRRG2goji26IRRG goji26IRRG2goji30RGB goji26RGB2goji30RGB; do
  for suffix in '' '_proto_struct_edge'; do
    name="doris_${direction}_hrda_3vi${suffix}"
    python tools/train.py "configs/hrda/${name}.py" \
      --work-dir "$CHECK_ROOT/smoke/${name}" \
      --options runner.max_iters=5 log_config.interval=1 \
      checkpoint_config.interval=5 evaluation.interval=5 data.workers_per_gpu=0 || break 2
  done
done
```

这些周期/worker 改动只在冒烟命令生效，不能用于评价正式精度或耗时。初期边界/Prototype Loss 可因原有 warmup 和启用阶段为零；五步不足以覆盖 8000/12000/14000 后的所有实际 GPU 分支。

### 正式批次 20K（当前含八组）

```bash
bash train_doris_mitb5_stage1_20k.sh
```

顺序为 T1 HRDA、T1 Full、T2 HRDA、T2 Full、T3 HRDA、T3 Full，再执行新增 MemoryAdaptNet（26IRRG→30RGB）和 DAFormer（26RGB→30RGB）。`set -e` 保持，失败即停止。每组直接保存到 doris_stage1 下按迁移方向和模型命名的独立子目录，正式训练超参数不覆盖；重复执行会复用相同目录。只补新两组时请使用 comparisons 脚本。

单独运行 T2 Full，并按需记录少量过程量：

```bash
FULL_DIR="$CHECK_ROOT/T2_full_20k"
python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py \
  --work-dir "$FULL_DIR" --options uda.doris_process_logging=True
```

不需要过程量时省略最后的 `--options`。默认依然关闭过程图片和源码归档，保留现有日志、Acc、验证、best/latest checkpoint。

## 10. 独立响应导出与正式评价

正式训练完成后，用该实验保存的有效 Full 配置和完整 UDA checkpoint：

```bash
SAVED_CFG="$FULL_DIR/doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py"
CKPT="$FULL_DIR/latest.pth"

python tools/export_doris_response_maps.py "$SAVED_CFG" "$CKPT" \
  --split val --indices 0 1 2 --device cuda:0 \
  --out-dir "$CHECK_ROOT/T2_fixed_responses" --save-npz
```

T1/T3 替换对应有效配置和 checkpoint。索引须在该数据集范围内，可先只选 0；每次最多 10 个固定索引，没有随机选图。

每个样本仅生成 `*_mparc.png`、`*_pbcr.png` 两张横图，可选一个 `*_responses.npz`，另保存本次导出 metadata。两图均使用 `viridis`、[0,1]，响应处于同一 pipeline 图像坐标（常为 resize 后 256×256）。R_adj/R_sem 不可用类别显示灰色，NPZ 附 availability 掩膜，不能将默认响应 1 解释为已观察证据。

导出使用 checkpoint 的 EMA Teacher，eval/FP32，执行训练同入口的 `generate_pseudo_label` 和融合方法；不声称逐位重放训练时 FP16 张量。先验从 checkpoint 恢复，不用目标域 GT 更新。metadata 使用 checkpoint 完成 iteration 决定响应 ramp；若 checkpoint 缺少 iteration，必须明确提供 `--iteration 20000` 等实际值。

缺少 Teacher 权重会拒绝运行。缺少关键先验或全部未初始化会明确报错；只有确实要检查冷启动/不完整先验时，才显式使用 `--allow-missing-priors`，metadata 会记录缺失情况，不伪造已初始化先验。

正式评价仍使用原工具，协议不变。T2 独立 val 示例：

```bash
python tools/test.py "$SAVED_CFG" "$CKPT" --eval mIoU \
  --options data.test.img_dir=rgb_dir/val data.test.ann_dir=ann_dir/val
```

正式 test 与定性预测图：

```bash
python tools/test.py "$SAVED_CFG" "$CKPT" --test-set --eval mIoU
python tools/test.py "$SAVED_CFG" "$CKPT" --test-set \
  --show-dir "$CHECK_ROOT/T2_qualitative" --opacity 1
```

T1 的目标域影像目录与 T2 不同，独立 val 时应按对应配置的 `data.val.img_dir/ann_dir` 指定，不直接照搬 T2 的 rgb_dir。新方法效果须由这些真实实验结果判断。
