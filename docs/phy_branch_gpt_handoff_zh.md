# PSSAL-DA phy 分支诊断与文献驱动重设计交接材料

整理日期为 2026-11-10。目标代码分支为 `doris-02`，整理起点为 `doris` 分支的 `d63695241350844c0fc6a360cef9d3401ede3db0`。本材料以当前源码为事实依据，论文 PDF 与审稿意见由作者另行提供。配套 ZIP 包含实际代码和可用响应数据。

## 1. 作者希望 GPT 解决的问题

研究任务为 Sentinel-2 枸杞种植区跨域语义分割。作者引入 VI 和 phy 分支的主要目标是改善 Goji 与其他植被的区分。现有消融表显示两类 IoU 提升，输入输出与空间对齐检查也已通过，但中间响应存在与预期不一致的现象。正确预测的 Goji 区域经常具有较低的生理响应并被明显减权，两类互相混淆的像素也未同时受到更强减权。

作者希望获得一套具有公开文献依据的方案，回答以下问题。

- 当前行为是否符合代码实际优化目标，与论文功能描述的差距具体在哪里。
- 哪些参数、统计量和图能真实评价 phy 的功能，哪些图只能证明模块已接入计算。
- UDA 和遥感语义分割公开论文如何评价光谱辅助分支、伪标签可靠性、类间可分性与监督权重。
- 怎样调整证据设计，或在必要时调整分支实现，使性能贡献和功能解释一致。
- 怎样以可控制的工作量回应审稿人的中间过程证据诉求。

作者允许后续实现与原论文公式存在差异，但最终代码、公式、框架图、损失说明和实验报告需要对应。具体模型修改方案需要由 GPT 基于文献提出并说明验证步骤。本次代码变更集中于未来训练总迭代和交接材料。

## 2. 实验时长与历史状态

作者观察到模型通常在约 12000 Iter 达到较好性能，因此要求后续命名为 `doris_*.py` 的实验统一训练至 15000 Iter。当前共有 29 个此类配置，包含三个方向的 24 个 HRDA 分支消融配置、2 个三 VI 比较配置和 3 个 R101 无 VI 比较配置。

本轮在这 29 个叶级实验入口设置 `runner.max_iters=15000`，并将 `name_opt` 描述改为 `15k`。父级传统实验配置保留各自原设置。原优化器和 LR 调度公式沿用既有配置，Poly 调度的终止时点随训练总长变为 15000。未来 15K 结果应按新协议独立报告。

历史 7K 和 20K 响应来自原 20K 训练协议。历史 Checkpoint 的保存元数据和历史结果仍按原协议解释。回放构建模型时使用该 Checkpoint 保存的 runner 配置，以保存当时的调度上下文。

Best mIoU 与 Best Goji IoU 的评价依据是正式验证集指标，机制图的案例选择需要记录来源与规则。

## 3. 数据、类别与迁移方向

| reduced label | 类别 |
|---|---|
| 0 | Goji |
| 1 | Vegetation，其他植被 |
| 2 | Building |
| 3 | Water |
| 4 | Road |
| 255 | ignore |

原始标注 1–5 经既有 `LoadAnnotations(reduce_zero_label=True)` 映射为 0–4。参考标注在离线前向结束后用于解释与统计。

| 迁移方向 | 配置声明的源域基础通道 | 目标域基础通道 | 固定目标域样本 |
|---|---|---|---|
| ZN_26 RGB → HN_30 RGB | R,G,B | R,G,B | `3_1024_1280_1280_1536.tif` |
| ZN_26 IRRG → HN_30 RGB | NIR,R,G | R,G,B | `3_1536_640_1792_896.tif` |
| HN_30 IRRG → ZN_26 IRRG | NIR,R,G | NIR,R,G | `3_768_964_1024_1220.tif` |

前两个目标样本位于 `/root/shared-nvme/mydataset/30/rgb_dir/val/`，第三个目标样本位于 `/root/shared-nvme/mydataset/26/img_dir/val/`。原始样本均为 256×256。四周各 30 像素排除伪监督，论文展示区域为 `[30:226,30:226]`，即中心 196×196。

## 4. 当前模型的两种 VI 使用路径

### 4.1 六通道分割输入

三种预计算 VI 为 NDVI、NDRE 和 CIRE。`LoadGojiIndices` 将 VI 与三通道基础影像拼接，产生六通道 `img`，供 Student 和 EMA Teacher 分割网络使用。三 VI 基线本身已经有这一输入。

因此，比较三 VI Baseline 与三 VI + phy，分析的是专门分支的增益。RGB-only R101 对照同时改变了骨干与输入条件，相关比较需要按实际实验设计解释。

### 4.2 独立生理头输入

当 `use_vi=True` 时，同一加载器还返回独立的 `vi_img`。它由真实 VI 文件经过归一化得到，供 `phy_head` 使用。

| VI | 独立 VI 路径的范围归一化 |
|---|---|
| NDVI | clip 到 [-1,1]，再映射到 [0,1] |
| NDRE | clip 到 [-1,1]，再映射到 [0,1] |
| CIRE | clip 到 [0,5]，再映射到 [0,1] |

当前共同配置中 `vi_norm_for_input=False`，六通道图像中的 VI 与独立头所用 VI 的数值处理需要分别核对。`prepare_goji_index` 执行 float32 转换和非有限值处理。VI 指数文件已预计算，加载器负责读取和预处理。

关键源码为 `mmseg/datasets/goji.py`、`mmseg/datasets/pipelines/loading.py`、`mmseg/datasets/pipelines/transforms.py` 和 `mmseg/vi_utils.py`。

## 5. 当前 phy 头实际计算什么

`mmseg/models/uda/phy_reliability.py` 中的 `PhyReliabilityHead` 默认结构如下。

```text
VI [N,3,H,W]
  → Conv 3×3, 3→16, BN, ReLU
  → Conv 3×3, 16→32, BN, ReLU
  → Conv 1×1, 32→1
  → Sigmoid
  → R_phy [N,1,H,W]
```

该头直接处理 VI，代码中没有把它的中间特征拼入分割主干或解码器。生理分支影响目标域分割监督的主要入口是伪监督权重。结构分支具有另外的特征交互路径，应单独分析。

### 5.1 源域监督

`DACS._compute_phy_src_loss` 将源域 GT 的 Goji 和 Vegetation 都设为正类，其余三类为负类，排除 ignore 像素。

```text
y_phy = 1[GT ∈ {0,1}]
L_phy_src_raw = mean_valid BCE(R_phy_src, y_phy)
L_phy_src = lambda_phy_src × L_phy_src_raw
```

源域原始日志 `loss_phy_src` 使用未加权口径，实际反传系数为 `lambda_phy_src`。分析日志时需要区分 raw 和 weighted 数值。

### 5.2 目标域监督

当前 DORIS 的 phy-only、MPARC、Full 配置均采用 `balanced_high_conf`。

```text
high = Teacher confidence ≥ tau_conf_phy
positive = valid & high & Teacher class ∈ {0,1}
negative = valid & high & Teacher class ∈ {2,3,4}
L_pos = mean_positive BCE(R_phy_tgt, 1)
L_neg = mean_negative BCE(R_phy_tgt, 0)
L_phy_tgt = lambda_phy_tgt × 有效正负组的加权平均
```

正负组权重各为 0.5，缺组时按实际有效组计算。代码还兼容旧 `legacy` 模式，它仅使用高置信植被预测作正类监督。历史结果应核对 Checkpoint 保存的模式。

### 5.3 与两类区分目标的关系

Goji→Vegetation 和 Vegetation→Goji 的错误，经过当前生理监督标签映射后，仍属于正类。该 BCE 的目标不会直接区分这两种类间错误与两类正确预测。这是源码可以确定的事实。

`R_phy` 也未被直接监督为伪标签正确概率。需要 GPT 分析怎样的目标、输出定义和证据能与作者希望的两类区分功能对应。

### 5.4 梯度路径和 EMA

- 源域生理 BCE 和目标域生理 BCE 更新独立 `phy_head`。
- 当前 `detach_target_gate=True`，目标分割损失通过已 detach 的生理门控加权。该分割损失不通过权重路径向生理头传递“哪些伪标签正确”的梯度。
- 源域生理监督与目标域生理辅助监督从训练早期参与，`phy_warmup_iters` 控制的是门控作用时序。
- EMA 更新在 `DACS._update_ema` 中对应 `get_model()` 与 `get_ema_model()` 的分割参数。独立 `phy_head` 使用当前 Checkpoint 的 `phy_head.*` 状态。
- 回放中分割预测与 confidence 来自 EMA Teacher，生理响应来自该阶段保存的生理头。
- 生理头包含 BN。正式训练前向使用训练模式下的统计，固定探针和离线回放使用 eval 模式的已保存运行统计。当前材料未提供同权重下两种模式响应差异的专门核对，需要 GPT 判断是否应增加受控的只读比较及其解释范围。

## 6. 实际权重公式

### 6.1 Teacher confidence 和 DACS 基础权重

```text
C(p) = max_c P_teacher(c,p)
pseudo_label(p) = argmax_c P_teacher(c,p)
q = count(C ≥ 0.968) / 全部像素数
W_before(p) = q × valid_mask(p)
```

训练中 q 按当前 Teacher batch 统计。固定单图回放则按该指定图像统计。历史训练 batch 的 q 没有保存为这张固定图像的永久状态，回放记录的是该阶段权重下重新计算的单图响应。

图中的 `w_conf` 是逐像素 C，`w_before_phy` 是 DACS 基础权重。有效区域的后者可能是空间常数。

### 6.2 生理门控

```text
M(p) = 1[pseudo_label(p) ∈ {0,1}]
a(p) = 0.82 + 0.18 × R_phy(p)
R_phy(p) < 0.30 时，soft filter 将 a(p) 乘以 0.5
rho = clip((Iter - 6000) / 4000, 0, 1)
G(p) = (1-M(p)) + M(p) × ((1-rho) + rho × a(p))
W_after(p) = W_before(p) × G(p)
DeltaW(p) = W_after(p) - W_before(p)
```

源码为 `DACS._apply_phy_gate`。0.30 附近的额外 0.5 因子产生明显的分段变化。底权和 soft filter 限定了响应到权重的关系。完整门控下，R 接近 0 时，G 接近 0.41。

Iter 7000 的 rho 为 0.25，Iter 10000 及以后为 1。20K 的绝对减权增大同时受到完整门控与基础权重变化影响，不能仅从颜色深浅比较机制优劣。

`M` 来自 Teacher 类别。图中的“GT 类别”由事后参考标注提供，两者的条件不同，需要保持区分。

### 6.3 与论文的对应

作者提供的 PDF 中，式(5)给出生理响应映射，式(7)给出逐像素 confidence，式(8)描述生理权重。当前实现包含 DACS 高置信比例、乘性门控、底权、soft filter 与渐进调度。

后续修订应以实际采用的实现重新同步公式与图文。作者允许设计演进，要求保留改动来源、对应实验与实际参数。

## 7. 当前 DORIS 参数和配置覆盖

以下值已从三个方向的配置继承结果核对。三个方向的 phy 参数当前相同。

| 参数 | 当前值 |
|---|---|
| positive_classes / apply_to_classes | [0,1] |
| phy_target_supervision_mode | balanced_high_conf |
| lambda_phy_src / lambda_phy_tgt | 0.08 / 0.008 |
| tau_conf_phy / pseudo_threshold | 0.968 / 0.968 |
| phy_pos_weight / phy_neg_weight | 0.5 / 0.5 |
| phy_min_weight | 0.82 |
| phy_filter_threshold / mode | 0.30 / soft |
| phy_warmup_iters / phy_rampup_iters | 6000 / 4000 |
| detach_target_gate | True |
| head in_channels / mid_channels / num_layers | 3 / 32 / 2 |

每个方向中，`_phy_gate.py`、`_phy_struct_gate.py` 和 `_proto_struct_edge.py` 启用 phy。其余分支消融根据各自开关构建。三个方向共 9 个含 phy 的 DORIS 配置。

关键父配置为 `configs/hrda/pssalda_26rgb2_30rgb_ablation_common.py`、`pssalda_26irrg2_30rgb_ablation_common.py`、`pssalda_30irrg2_26irrg_ablation_common.py`。公共模块构造参数在 `configs/hrda/pssalda_26rgb2_30rgb_ablation_utils.py`，各叶级配置继续覆盖正式 DORIS 设计。

源码声明的 MMSegmentation 版本为 0.16.0。作者此前使用的远端环境名为 `hrda`，反馈过 MMCV 1.4.0 和 Torch 1.13.1+cu117，具体运行环境以作者终端为准。

## 8. 已有量化结果与来源

### 8.1 论文分支消融表

以下数字来自作者提供的 PDF 表IV，属于论文报告结果，本次整理未重新执行正式验证。

| 配置 | Goji IoU | Vegetation IoU | mIoU | Veg.mIoU |
|---|---|---|---|---|
| Baseline | 78.61 | 68.10 | 73.67 | 73.36 |
| Physiology | 80.20 | 73.00 | 74.36 | 76.60 |

当前任务接受分支在这些指标上具有增益，需要解释为什么现有过程图与预期功能仍存在距离。比较表、图中 Teacher 身份、各自 Checkpoint 和最优节点选择规则需要逐项对齐。

### 8.2 固定样本 Iter 7000

本地可用真实 `responses.npz` 和回放元数据，配套包提供该节点数据。原始图片和各响应为 256×256，有效区为 38416 像素。

| GT / Teacher 状态 | 像素数 | 平均 R_phy | 平均 DeltaW |
|---|---|---|---|
| Goji 判对 | 13753 | 0.513113 | -0.029889 |
| Goji→其他植被 | 312 | 0.777104 | -0.006141 |
| 其他植被→Goji | 446 | 0.715196 | -0.009654 |
| 其他植被判对 | 9339 | 0.935702 | -0.001835 |

全部有效区 W_before 均值为 0.554825，W_after 均值约 0.540167，平均 DeltaW 约 -0.014658，最大下降约 0.081544，发生变化的有效像素约为 81.6%。

该节点为 RGB→RGB phy-only 的 `best_mIoU_iter_7000.pth`。远端路径为 `/root/shared-nvme/work_dirs/doris_stage1/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_phy_gate/best_mIoU_iter_7000.pth`。

### 8.3 固定样本 Iter 20000

以下数值根据作者提供的回放截图记录。原始 20K NPZ 尚未纳入本地核对，其证据级别与前一小节不同。

| GT / Teacher 状态 | 像素数 | 平均 R_phy | 平均 DeltaW |
|---|---|---|---|
| Goji 判对 | 13625 | 0.553 | -0.13153 |
| Goji→其他植被 | 398 | 0.688 | -0.05327 |
| 其他植被→Goji | 259 | 0.683 | -0.05685 |
| 其他植被判对 | 9618 | 0.918 | -0.01255 |

截图显示 W_before 均值为 0.6405，W_after 均值为 0.5752，平均 DeltaW 为 -0.06530，最大下降为 0.37717，变化有效像素比例为 84.7%。

两节点中，“其他植被→Goji”的错误像素比同一 GT 类的判对像素减权更多，而“Goji→其他植被”的错误像素比 Goji 判对像素减权更少。这个现象提示需要核查响应学习目标与类间可靠性之间的关系。

Goji 的稀疏冠层、行间土壤、源目标光谱差异等可以成为待检验解释。当前材料尚未提供能够确认这些物理原因的额外测量。

## 9. 当前过程图的构造方式及局限

phy 论文版为一行五张空间图，每张使用同一 196×196 区域。

| 面板 | 实际内容 |
|---|---|
| a | 目标域 RGB 展示 |
| b | EMA Teacher 最大类别概率 |
| c | 实际 phy_head 的单通道 R_phy |
| d | W_after-W_before |
| e | GT→Teacher 的 Goji/其他植被四种判别状态 |

第五图颜色编码分别为 Goji→Goji、Goji→Vegetation、Vegetation→Goji、Vegetation→Vegetation。其余类别及 ignore 区域为灰色。该图根据预测和 GT 确定位置，效应大小由权重图和统计表提供。

R_phy 和 DeltaW 存在门控函数所规定的数学依赖，两图之间的一致只能证明计算关系。把单通道头对两类地物的均值差异解释为两类区分能力，需要额外的定义与证据。

作者还把 HRDA 与 HRDA+phy 的混淆图放在一起，手工框出局部改善。该比较使用两个独立训练模型。框内改善应与整张样本、验证集类别指标及其余区域变化共同解释。

默认 confidence 和 R 范围为 [0,1]。回放入口的前后权重散图范围为 [0,1]，DeltaW 默认 [-0.10,0]，超界时按 0.05 向外扩展下界。7000 与 20000 的截图色条分别为 [-0.1,0] 与 [-0.4,0]，跨节点比较应使用共同显示范围或原始数值。

已有统计主要为四组计数、均值、标准差及减权比例。当前缺少这些信息与后续分割改进之间的直接验证，也缺少各组原始 VI、R 的分位数及触发 0.30 soft filter 的比例分析。

## 10. 在线采集与离线回放

核心实现位于 `mmseg/utils/mechanism.py`。

```text
deterministic_probe_config
  → load_probe_sample
  → probe_arrays / branch_probe_payload
  → 保存 responses.npz 和元数据
  → build_doris_mechanism_figures
  → 论文综合图、散图和统计
```

固定探针当前使用原始像素网格、目标域实际通道顺序、VI 处理及归一化。回放采用 eval 模式，通过共享分支方法读取门控响应，恢复 Checkpoint 对应 local_iter。参数、Buffer 和随机状态通过只读上下文及哈希检查验证。

GT 参考在前向后读取。原始 256×256 数据与 196×196 展示结果分目录保存。每张图和 NPZ 记录 Checkpoint、Iter、TIF、输入张量及源文件哈希。

两项已处理的 Baseline 加载兼容问题需要保留。

- 三 VI Baseline 可能只声明 `use_indices=True`，离线接口需额外取得独立 `vi_img`，该返回字段的启用已验证保持六通道 img 逐元素一致。
- RGB-only 历史配置的顶层归一化字典可能继承六通道 channel_order，回放应读取实际目标域训练管线的 Normalize 参数。RGB-only 的辅助 VI 用零通道空张量表示缺席。

### 常用命令

```bash
python tools/replay_doris_phy_7000.py
python tools/replay_doris_phy_7000.py --checkpoint iter_20000.pth
python tools/replay_doris_baseline_18000.py
python tools/replay_doris_daformer_13000.py
python tools/replay_doris_daformer_r101_no3vi_15000.py
```

phy 入口默认读取该实验的 Best mIoU Iter 7000。Baseline 默认路径分别为以下文件。

```text
/root/shared-nvme/work_dirs/doris_stage1/goji26RGB2goji30RGB_hrda_3vi/iter_18000.pth
/root/shared-nvme/work_dirs/doris_stage1/goji26RGB2goji30RGB_daformer_3vi/best_mIoU_iter_13000.pth
/root/shared-nvme/work_dirs/doris_no3vi_r101/goji26RGB2goji30RGB_daformer_r101_no3vi/best_mIoU_iter_15000.pth
```

所有入口在权重所属工作目录的 `analysis/fixed_probe/replayed/` 下创建独立时间戳目录。原始训练快照位于独立的 `snapshots/` 路径。

## 11. 审稿人的相关诉求

原文以作者另附 DOCX 为准，以下仅整理与 phy 直接相关的任务范围。

- 审稿人1第3条要求代表性地物的 VI 响应可视化，并指出 Goji 与其他植被共用掩膜和监督，质疑枸杞特异性解释。该条还讨论单指数与指数组合的验证。
- 审稿人1第7条要求中间响应、Teacher confidence 及监督权重可视化，验证模块是否发挥预期作用。
- 审稿人1第2条要求框架、公式和真实训练流程一致。
- 审稿人2第3条质疑光谱偏移及混合像元条件下，响应能否被解释为典型枸杞生理特征。

作者希望控制验证工作量，优先明确最有解释力的核心证据。论文表述需要覆盖已有局部改善与当前观察到的选择性减权问题。

## 12. 请 GPT 按公开文献开展的工作

### 12.1 文献检索与证据溯源

请先实际阅读公开 UDA、遥感跨域分割和光谱辅助学习论文，确认它们如何展示分支功能。可从 DACS、DAFormer、HRDA、MIC、原型类 UDA 和与光谱先验直接相关的方法检索。

仓库 README 提供 HRDA 原文入口 `https://arxiv.org/pdf/2204.13132` 和 MIC 官方仓库入口 `https://github.com/lhoyer/MIC`。这些是检索起点，具体方法与图表需要逐一阅读验证。

对每条推荐给出论文题目、年份、正式链接或 DOI、相关图表号或章节、实际展示内容，以及它为何适用于当前分支。区分论文直接采用的证据形式与基于原理提出的扩展建议。找不到直接支持时明确记录证据缺口。

### 12.2 先诊断，再设计

请用当前源码核对以下问题。

1. 当前共同正类 BCE 对 Goji/Vegetation 的类间判别提供了什么约束。
2. 高置信 Teacher 产生目标生理监督、门控 detach 及辅助头独立梯度之间有什么影响。
3. 0.30 阈值处的 soft filter、0.82 底权、DACS 常数基础权重与渐进强度各自如何影响当前图形。
4. 性能增益可能对应哪条实际路径，现有证据能够区分到什么程度。
5. 一张最佳节点图、少量固定阶段图、跨样本统计和消融表各自能够回答哪些问题。
6. BN 的训练与回放模式差异是否影响目前的解释，怎样在保持 Checkpoint 状态可追溯的条件下检验。

请区分源码可确认的事实、当前数据可确认的观察和需要新实验的假设。

### 12.3 重新设计最有代表性的评价量和图

请给出优先级清晰、可执行且适合论文正文的方案，并为选择提供文献依据。需要考虑以下候选问题，但由文献和实际计算定义决定最终保留项。

- Goji 与其他植被的光谱或学习表示可分性，应在哪个空间、用什么指标评价。
- 每个 GT 类别内部，正确和错误伪标签得到怎样的门控值和相对权重变化。
- 权重对错误伪监督的排序、覆盖和可靠性评价是否比原始 DeltaW 热图更有解释力。
- 原始 VI 分布、R_phy 分布、阈值触发率与预测混淆能否提供更清楚的过程关系。
- 预测置信度、类别概率差值和最终类间错误变化应如何与辅助分支区分。
- 哪些图需要模型输出，哪些只需要已保存张量，哪些需要重新训练才能成立。

请提出一套尽量紧凑的正文证据组合，说明每张图或表支持的唯一主要结论、采样范围、统计单位、色条或坐标规则、数据导出字段和失败情况的报告方式。

### 12.4 实现与实验方案

请分别给出“基于现有训练结果改进证据组织”和“必要时重新设计 phy 目标及门控”的方案，并比较它们的证据强度、代码修改点、训练成本与适用边界。

涉及未来训练的方案统一采用 15000 Iter，并说明 warm-up、完整生效阶段和最佳节点选择协议。作者希望将性能改善与论文中的两类区分功能对齐，原公式可随经过验证的新实现更新。

需要列明公平的对照条件，区分 RGB-only、RGB+VI 拼接和 RGB+VI+phy 的作用。全局指标、两个类别 IoU、局部错误及过程统计应互相核对。GT 在训练、验证选择与事后解释中的使用范围需要明确。

最终请输出问题诊断、文献对照表、推荐指标与图、最小修改方案、15000 Iter 实验计划、结果可接受与不可支持的判据，以及对应的中文正文描述建议。

## 13. GPT 阅读源码的建议顺序

| 文件 | 要核对的内容 |
|---|---|
| `mmseg/models/uda/phy_reliability.py` | 头结构、两种目标监督模式 |
| `mmseg/models/uda/dacs.py` | 构造、EMA、源 BCE、目标门控、ClassMix 权重、目标 BCE 与反传 |
| `configs/hrda/pssalda_26rgb2_30rgb_ablation_utils.py` | 公共默认参数 |
| `configs/hrda/doris/*/*phy_gate.py` | 三方向正式 phy-only 覆盖 |
| `configs/hrda/doris/*/*phy_struct_gate.py`、`*proto_struct_edge.py` | MPARC/Full 中的相同 phy 设置 |
| 三个 `pssalda_*_ablation_common.py` | 数据路径、通道与 VI 归一化 |
| `mmseg/datasets/goji.py`、`loading.py`、`transforms.py`、`vi_utils.py` | 实际输入与标签映射 |
| `mmseg/utils/mechanism.py` | 固定样本、共享前向、只读采集 |
| `tools/replay_doris_mechanisms.py` | 历史权重恢复与完整性核验 |
| `tools/replay_doris_phy_7000.py` | 单命令入口与显示范围 |
| `tools/check_doris_phy_semantics_7000.py` | 四种判别状态与类内比较 |
| `tools/doris_phy_evidence.py` | GT 参考和类别统计 |
| `tools/build_doris_mechanism_figures.py` | 组图、裁剪、色条、元数据 |
| `mmseg/models/builder.py`、`mmseg/apis/train.py` | max_iters 注入与正式训练设置 |
| `mmseg/utils/checkpoint_policy.py` | 周期、关键节点、最佳权重和历史状态保存 |

源码包中的历史数据和作者截图应按各自来源使用。报告中的候选图和指标属于 GPT 需要基于文献评估的设计问题，尚未被本次整理验证为有效方案。

## 14. 本次核验范围

- 29 个 DORIS 配置与整理起点代码的解析结果逐项比较，差异仅为 runner.max_iters 和 name_opt。
- 24 个 HRDA 消融配置、2 个三 VI 比较配置和 3 个 R101 无 VI 配置的静态检查通过。
- 2 项配置单元检查和 9 项历史回放、Baseline、参考标注检查通过。其中回放检查明确验证旧 20K runner 在新 15K 配置下恢复。
- 本地配置解析使用项目现有静态解析工具及 Config 适配器。真实远端 MMCV 和 GPU 完整模型运行需要作者在相应环境验证。
- 7000 节点原始响应可用，20000 节点当前以作者截图为依据。后者的原始 NPZ 与完整验证日志可作为 GPT 后续要求补充的材料。
- 既有批处理脚本文件名中的 `20k` 保留作命令兼容，运行时读取的 DORIS 配置为 15000 Iter。实验 work-dir 沿用配置中的路径，后续新实验目录由作者按历史 Checkpoint 保存保护规则管理。
