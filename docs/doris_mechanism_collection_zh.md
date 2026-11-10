# DORIS 机制采集工程交付 V1

> 最新输出策略见 [每500步在线机制快照](doris_online_mechanism_snapshots_zh.md)：中间消融每节点一张PNG、Full两张，500步即保存，保留一个NPZ和分支CSV。以下历次策略说明仅供追溯。

> 当前默认输出已收敛为四分支综合证据，以 [四分支输出交付](doris_branch_evidence_convergence_zh.md) 为准。本页每500步PNG、详细CSV和独立PHY-1/PHY-2等描述为历史行为；现默认一个节点一个压缩NPZ、最终每启用分支最多一张综合图，算法定义保留。

> 生理分支后续更新：9组已启用balanced_high_conf辅助监督，生理改为500步NPZ及最终PHY-1/PHY-2两图，不再逐节点生成生理PNG；以[doris_physiology_optimization_zh.md](doris_physiology_optimization_zh.md)为准。其他模块定义保留。

> 后续更新：采集器现已支持按消融模块分工，编号2–8均采集对应机制；本页的“仅Full采集”、固定三张汇总图和固定文件数量是旧版说明。当前开关、字段和图表矩阵以[doris_module_collection_zh.md](doris_module_collection_zh.md)为准。

## 范围与当前结论

本轮以最终整合指令和当前本地代码为准，完善观测工程，不重新设计 PSSAL-DA。没有改变两种融合响应、边界损失、tau_proto、原型时序、学习率、损失强度、类别映射、数据划分或 checkpoint 选择协议。新旧 Full 的有效算法配置已逐项比较。

此前日志方案中默认开启的 doris.* 大量标量和 paper_names 展示改名，在六组正式配置中停用；恢复原始 MMSeg loss 名称，机制证据改写入 CSV 和固定探针文件。旧接口保留以兼容已有脚本。

核查基点：分支 doris，HEAD `8b587700c012d14cbccaddcc94ffd6662ab7468c`；工作区已有未提交修改均保留。论文第6–11页对应响应、原型及损失，第17–19页对应机制分析。审稿人关注的阈值覆盖、四证据及三阶段权重得到采集接口，不代表已用新训练结果证明精度改善或阈值最优。

## 1 修改前核查

| 类型 | 核查结果与处理 |
|---|---|
| 已有可复用 | PrecisionLoggerHook、原始 loss 分组、分组 IoU、源配置/有效配置保存、Git/seed 日志、结构/边界融合函数、ProtoGate 阈值诊断、独立响应导出、Fig.12过程曲线工具 |
| 存在但需修正 | 六组显存默认关闭；Full 展示名替换原 loss key；低频诊断未写 CSV；结构 apply 包含先验更新调用；原型相似度会刷新缓存；验证日志曾混用图像数和训练步数 |
| 本轮新实现 | 500步累计 CSV、单个固定无标签训练样本探针、合并 NPZ、在线两图、结束后三张候选图、来源/模型状态校验、Resume去重和开销记录 |
| 必要的 Resume 恢复修复 | DACS.local_iter 从 runner.iter 恢复，避免续训重新初始化 Teacher/重启时序；空原型缓冲区先按 checkpoint 形状分配再加载 |

只使用现有 PrecisionLoggerHook 调度，未新增第二个 Logger 或 Hook。新 `mmseg/utils/mechanism.py` 是观测工具，新的汇总脚本同时服务在线图和最终候选图，未复制模型源码。

## 2 文件清单

| 文件 | 本轮作用 |
|---|---|
| `mmseg/utils/precision_logger.py` | 复用现有 Hook 的 before_run/after_train_iter/after_run；写入实际 work_dir；排除探针对下一步计时的污染；原始 loss 名称及小损失科学计数法 |
| `mmseg/utils/mechanism.py` | 累计统计、固定样本、只读/RNG保护、CSV去重、状态哈希、合并 NPZ、在线绘图调度、自动汇总 |
| `mmseg/models/uda/dacs.py` | 每100步从实际权重位置采样；复用已经算出的 R_edge，不为训练统计增加模型前向；记录训练 step wall time |
| `mmseg/models/uda/proto_gate.py` | 计数分子、空间有效支持、只读相似度计算、延迟缓存的 checkpoint 恢复 |
| `mmseg/models/uda/struct_prior_gate.py` | `update_priors=False` 明确阻止探针调用更新函数，训练默认仍为 True |
| `mmseg/apis/train.py` | 续训恢复 DACS 计数器；把本次有效 cfg 提供给观测，避免 Resume 后使用旧 metadata 配置 |
| `tools/train.py` | 在 meta 中记录实际输入配置路径，保留现有原始/有效配置保存方式 |
| 六个 `configs/hrda/doris/<迁移方向>/doris_*.py` | 六组均启用 log_memory；三个 Full 开启采集，三个 Baseline 明确关闭；不改原输出路径 |
| `tools/build_doris_mechanism_figures.py` | 只读 CSV/NPZ，统一在线图与三张最终候选图；支持多个独立实验目录，禁止混合不同样本来源 |
| `tools/plot_goji_e1_process_curves.py` | 复用旧曲线工具，加入新 CSV 字段映射；仅绘图时将边界权重乘100 |
| `tools/export_doris_response_maps.py` | 旧独立导出接口也显式禁止结构先验更新 |
| `tools/check_doris_stage1_configs.py` | 核查六组观测开关、显存、固定时序及旧24组 legacy 模式 |
| `tests/test_doris_mechanism.py` | 状态、RNG、CSV、Resume、40统计节点、缓存恢复等专项测试 |
| `tests/test_doris_mechanism_figures.py` | 40个明确标记的模拟节点，验证三张 PDF/PNG 候选图和来源元数据 |
| 既有 Logger/Stage1 测试 | 补显存与原始字段检查；验证采集开关前后参数/梯度完全一致 |

需同步以上运行源码、工具和六个配置。此前 mask 接口及 Goji 分组评价改动也应与本地版本保持一致。测试、文档无需参与模型计算。本轮没有修改批次脚本、论文 PDF/Word、远端环境，没有连接或同步远端。

## 3 配置与基础日志

三组 Full 的配置：

```python
mechanism_logging = True
mechanism_sample_interval = 100
mechanism_summary_interval = 500
fixed_probe_enabled = True
fixed_probe_interval = 500
fixed_probe_max_samples = 1
fixed_probe_index = 0
mechanism_save_raw = True
mechanism_make_figures = True
```

这些是 uda 中的观测配置。fixed_probe_index 是目标域训练影像按文件名排序后的索引；默认0，可在新实验开始前明确指定。改变样本/预处理后不能继续混入同一 analysis 目录。第一幅影像不保证包含全部地物类，代表性需人工检查。

六组 Logger 均为 `PrecisionLoggerHook`、interval=50、by_epoch=False、log_memory=True、paper_names=False。Full 的 doris_process_logging=False；随机 batch 图片仍关闭。Baseline 不创建 analysis 机制目录或伪造模块图。

训练 TXT/JSON 保留 iter/lr/time/data_time/memory/loss、decode.loss_seg、mix.decode.loss_seg、各 hr 项，以及实际启用的 loss_phy_src、loss_phy_tgt、loss_proto_tgt、decode.loss_struct_edge、mix.decode.loss_struct_edge。新增 train_iter_time，不更改训练 loss 的内部 key 或总和。非零且小于1e-4的 loss 使用科学计数法显示，JSON保持原始精度。

`memory` 为原有 torch.cuda.max_memory_allocated() 的生命周期峰值已分配显存，单位 MiB，无 CUDA 安全跳过，不是 reserved/current/nvidia-smi 显存。包含固定探针导致的峰值，因此不能把启用探针的峰值直接作为纯网络复杂度比较。

每1000步仍使用现有评价：五类 IoU、mIoU、Veg.mIoU、NonVeg.mIoU、aAcc、mF1 等；best仍按mIoU，周期/最终checkpoint设置不变。验证日志 iter 对应完成训练步数，eval_batches另记验证batch数。

## 4 机制 CSV 的字段

路径 `analysis/metrics/mechanism_500.csv`，来源固定为 training_batch_actual。完整20K有40条；正常情况下每条采样5个 batch，并非全部500步严格平均。

| 字段 | 定义 |
|---|---|
| iter | 完成的 runner 训练迭代数，500、1000…20000 |
| window_start | 当前统计窗口起点；Resume可能形成不完整窗口 |
| data_source | training_batch_actual |
| n_sampled_batches | 实际采样batch数量，不硬编码为5 |
| sample_interval | 实际配置的训练采样间隔，默认100 |
| valid_count | 各采样batch有效目标图像像元数之和 |
| valid_area_ratio | 累计有效像元数/累计全部图像像元数 |
| teacher_high_conf_ratio | 有效区confidence≥pseudo_threshold的累计数量/valid_count |
| w_conf_mean_valid | Teacher逐像元最大概率之和/valid_count |
| w_t_ps_mean_valid | MPARC后、原型调制前的实际权重之和/valid_count |
| w_t_proto_mean_valid | Prototype后、ClassMix混合前的实际最终权重之和/valid_count |
| r_struct_mean_valid | 实际R_struct之和/valid_count |
| r_edge_mean_valid | 实际目标R_edge之和/valid_count，不是仅边界区域均值 |
| boundary_weight_mean_valid | 实际目标边界监督权重均值，包含域系数、warmup、可靠性，未乘100 |
| tau_proto / tau_conf_proto | 实际筛选阈值，保持0.25 / 0.88 |
| proto_valid_class_count | 最后一个采样batch的融合原型有效类别数，不是窗口均值 |
| proto_update_active | 最后采样时是否达到目标原型更新时序；不意味着一定有像元更新 |
| proto_weight_active | 达到原型加权时序且原型缓存有效类别数足够 |
| proto_target_update_ratio | 实际更新特征像元数/累计有效Teacher特征预测像元数 |

均值按累计分子/分母计算，不直接平均不同有效面积batch的比例。零分母记0，同时保留零计数，不能解释为有观测支持的性能值。有效区域排除ignore=255、valid_pseudo_mask和已配置的裁边。

## 5 原型筛选 CSV 的字段

路径 `analysis/metrics/proto_selection_500.csv`。每节点五类各一行，完整运行理论200行。计数位于实际 ProtoGate 特征网格，不是256×256图像像元网格。

| 字段 | 定义 |
|---|---|
| iter、class_id、class_name | 完成训练步数和0..4五类别 |
| tau_proto、tau_conf_proto | 实际使用的两个阈值 |
| n_sampled_batches、sample_interval | 当前窗口采样数量与频率 |
| data_source | training_batch_actual |
| update_active | 目标原型更新时序已启用，区别于selected_count |
| prototype_initialized | 最后采样时该类目标原型缓存是否已初始化 |
| predicted_count | 有效Teacher预测为该类的特征位置累计数量 |
| confidence_pass_count | 在该类有效候选中通过confidence门槛的数量 |
| weight_pass_count | 在该类有效候选中通过W_t_PS门槛的数量 |
| joint_selected_count | 同时通过两阈值、非ignore的数量，尚未经过min_pixels及时序门槛 |
| updated_count | 达到更新时序且该batch该类达到min_pixels后，实际进入更新的像元数量 |
| selected_ratio | joint_selected_count/predicted_count，同类候选为分母 |
| selected_ratio_all_valid | joint_selected_count/所有类别累计有效特征预测数量 |

空间有效支持按现有W_t_PS的bilinear特征网格对齐处理：完全无效单元不计候选；部分有效支持保留，以反映真实可能通过阈值的单元。该处理只影响诊断分母，没有向训练筛选新增条件。当前tau_proto>0，完全无效区W_t_PS为0，本就不能通过实际筛选。训练筛选、min_pixels、EMA更新规则均保持原实现。

更新前候选、阈值后selected、真正updated严格区分。0.968是DACS基础置信度条件，0.88用于原型Teacher门槛，0.25用于MPARC后W_t_PS；彼此不替代。未做阈值搜索，也不将离线覆盖率当作不同阈值重训后的IoU。

## 6 固定样本探针与只读安全

从目标无标签训练集按文件名稳定排序选取一个样本，单独构建确定性管线：LoadImageFromFile、LoadGojiIndices、固定Resize、原目标Normalize、格式化/Collect。ann_dir=None，不包含LoadAnnotations、随机裁剪/缩放/翻转/颜色增强，也不使用训练DataLoader或Sampler。

样本在before_run只加载一次，保存原始路径、sample_id、IRRG/RGB顺序、确定性管线、尺寸、crop=None（整块输入）和预处理张量SHA。后续节点复用同一CPU缓存，A/B图共享同一节点和空间位置。

每500完成步数在当前模型状态进行eval/no_grad分析：EMA Teacher预测、现有生理门控、结构响应/权重、Student原型特征、既有原型缓存和边界类别对先验。结构调用update_priors=False，原型调用read_only=True，不调用任何先验/原型更新函数。恢复所有子模块原有模式，包括原本单独eval的Dropout，并恢复推理scratch/crop状态和Python/NumPy/Torch CPU/CUDA RNG。

探针采用当前缓存，不更新参数、梯度、BN、EMA、优化器或数据采样。CUDA时使用autocast；NPZ保存float32响应，并记录precision。它不承诺逐位重放FP16训练batch。

节点iter使用完成的runner.iter+1。NPZ另记rule_iter=当前模型计数器，通常等于完成步数；此时模型已经完成该步更新，不能称为该步随机batch的权重重放。DACS原始权重依赖probe自身batch的高置信像元比例，再按同一门控规则重算，因此明确标为fixed_probe_recomputed。

每节点一个合并NPZ：r_conf、r_loc、r_adj、r_prob、r_struct、r_edge、w_conf、w_t_ps、w_t_proto、delta_w_proto、pseudo_label、valid_mask、adjacency_available、semantic_available、proto_active、rule_iter、target_image、metadata。没有保存高维特征。未初始化先验的中性计算值与availability同时保存，图中灰色遮蔽，不能解释为真实高响应。

## 7 在线图与最终三张候选图

在线每节点仅两张PNG：

- `iter_000500_struct_components.png`：Target Image | R_conf | R_loc | R_adj | R_prob | R_struct。
- `iter_000500_weight_stages.png`：Target Image | W_conf | W_t_PS | W_t_proto | Delta W_proto。

响应/权重统一viridis和[0,1]；差值固定[-1,1]、以零为中心，不放大负差异。影像按实际归一化参数还原基础三通道，不对响应单独min-max。IRRG显示为对应假彩色组合。无效区/不可用先验显示灰色，Prototype未生效标inactive。

完整训练after_run自动从保存数据生成三张候选图，不再推理：

| 候选图 | 实际选择规则 |
|---|---|
| fig_struct_components | 优先最后一个有有效区域且结构先验可用的节点；若没有可用先验则如实保留灰色状态 |
| fig_weight_evolution | 从真实节点选择最多6个阶段，包含early/final、原型更新和权重时序附近，优先包含实际proto_active前后；缺节点不补造 |
| fig_proto_selection | 采用已有训练CSV中最近的更新启用节点；展示五类confidence通过比例与联合通过比例，无候选单独标注 |

每张输出PDF与300dpi PNG。figure_selection.json记录所选Iter、节点数、sample_id、来源、源数据SHA及绘图代码SHA。不将阶段性统计上升等同于精度提高。没有重复种子，不生成虚构误差棒或显著性。

中断训练后可单独执行工具。少于40节点时只使用现有节点，短Smoke Test的演化图可能只有2行且均inactive。已有图与新数据/绘图代码不同则写入版本子目录，保留原图；相同来源重复执行不改写已有完整图。

Fig.12旧工具保留。新CSV字段映射：w_conf_mean_valid→Teacher Confidence，w_t_ps_mean_valid→MPARC Pseudo Weight，proto_target_update_ratio→Prototype Target Update Ratio，boundary_weight_mean_valid×100→Boundary Refinement Weight。乘100只在绘图读取阶段进行，原CSV不变，拒绝混入fixed_probe_recomputed曲线。

## 8 输出目录、Resume和空间开销

所有输出以实际runner.work_dir为根，遵循CLI --work-dir，未硬编码历史路径：

```text
work_dir/
  timestamp.log / timestamp.log.json / source_配置.py / 有效配置.py / checkpoints...
  analysis/
    manifest.json
    metrics/
      mechanism_500.csv
      proto_selection_500.csv
      probe_cost.csv
    fixed_probe/
      sample_manifest.json
      snapshots/iter_000500_struct_components.png、iter_000500_weight_stages.png ...
      arrays/iter_000500.npz ... iter_020000.npz
    figures/
      fig_struct_components.pdf/png
      fig_weight_evolution.pdf/png
      fig_proto_selection.pdf/png
      figure_selection.json
    reports/
      metric_definitions.md
      formula_code_mapping.md
```

完整Full理论为40个CSV节点、200条类别记录、80张在线PNG、40个合并NPZ、3个最终PDF和3个PNG，以及少量manifest/说明/成本CSV，总计约134个analysis文件。Baseline没有这些图和数组。仅float32响应/影像及掩膜的未压缩数组约3.5MiB/节点，40节点约140MiB；实际NPZ压缩率和PNG大小由真实空间内容决定，不能以均匀模拟数据估算真实压缩率。不保存模型权重副本或高维特征，无需改输出路径。

CSV按iter或(iter,class_id)唯一，原子写入；重入相同值不重复，冲突值报错保留原文件。快照保存配置SHA、预处理张量SHA、整个模型/先验状态SHA，已有节点先校验metadata/NPZ/PNG完整性，不静默覆盖不同状态。不同配置、样本、Git基点的analysis不混写。修改协议请使用独立work_dir。

Resume使用本次有效配置，不误读checkpoint旧meta。DACS恢复local_iter，原型缓存恢复真实形状和内容。窗口未完成就中断时不虚构丢失的batch统计；恢复后的首个节点可能不足5个采样batch，n_sampled_batches/window_start如实记录。MMCV原有数据采样/RNG checkpoint语义未重写，因此不承诺整段续训逐位重放；探针自身不改变随机序列。中断时未完成的采集节点不补造。

## 9 耗时与显存口径

`train_iter_time`是训练step的wall time，包含训练中的轻量100步统计，不含固定探针、验证或数据加载；未为计时新增每步CUDA强制同步。原time/data_time保留，IterTimer的时间原点扣除本Hook观测时间，避免将500步探针耗时串入下一步data_time。

`probe_cost.csv`记录iter、mechanism_probe_time、memory_before_mib、memory_after_mib、run_peak_increase_mib。probe时间包括推理、必要同步、状态哈希、数组/图片I/O和绘图，不能称为纯GPU kernel时间。峰值增量只表示本次是否抬高进程生命周期峰值；没有reset_peak_memory_stats，不能将增量0解释为探针不占显存。

真实4090耗时尚未测量。云端应报告40个probe时间的中位数/总和，并与相邻500步训练时段作比较，明确是否含验证。不要把固定探针后的总耗时写成原始纯训练速度。rank0写analysis，统计来自rank0的实际训练batch；当前单GPU使用，不宣称多GPU全局汇总。

## 10 符号对应与科学解释边界

| 论文量 | 真实采集位置 | 保存字段 |
|---|---|---|
| W_conf / R_conf | EMA Teacher逐像元最大概率 | w_conf / r_conf；w_conf_mean_valid |
| R_loc | 原R_continuity | r_loc |
| R_adj | 源类别邻接先验响应 | r_adj及adjacency_available |
| R_prob | 源概率语义原型响应，不是PBCR特征原型 | r_prob及semantic_available |
| R_struct | 已确定的结构融合输出 | r_struct；r_struct_mean_valid |
| W_t_PS | 生理/结构门控之后、ProtoGate.compute_weight之前 | w_t_ps；w_t_ps_mean_valid |
| W_t_proto | Prototype乘权之后、ClassMix原地混合之前 | w_t_proto；w_t_proto_mean_valid |
| Delta W_proto | 上述两阶段权重之差 | delta_w_proto，不增强显示 |
| R_edge | 已确定的目标边界融合输出 | r_edge；r_edge_mean_valid |
| tau_proto / tau_conf_proto | update_target实际阈值 | 两CSV阈值及每类分子/分母 |
| L_edge | 原loss_struct_edge及HRDA各尺度/样本分支 | 原始训练loss key保留，不新增可训练loss |

四结构证据、三边界证据、两个响应和边界损失均未因采集改写。此前已列出的旧PDF与原型聚合/margin/DACS权重定义差异仍按冻结方案保留，本轮没有借日志工程重做模型。中间响应和覆盖率只提供机制证据，最终性能仍需要真实验证IoU、后续消融及必要边界评价。

## 11 本地验收与云端手动检查

本地执行真实Config.fromfile核查32个有效配置；六组均恢复显存，三Baseline关闭机制，三Full保持冻结参数。新增小张量测试验证参数/缓冲区/BN/梯度/优化器/RNG不变、异常模式恢复、采集开关前后参数与梯度一致、原型checkpoint round-trip、Resume计数、CSV500/1000与完整40节点、候选分母、min_pixels、空间有效性、状态冲突保护、Baseline无伪图，以及显存MiB进入终端/JSON且无CUDA跳过。

最终本地验收：Torch环境运行43项，41项通过、2项因缺少MMCV/Matplotlib跳过；这2项随后在已有基础Python环境补跑通过，另1项40节点绘图测试通过，合计44项不同测试通过。32个有效配置使用真实MMCV Config.fromfile解析并核查；其中比较配置的custom_imports在静态审计时关闭，因为基础Python没有Torch，此检查不等同于模型导入/初始化验收。3个R101无3VI配置兼容核查通过。

40节点模拟绘图测试生成三张PDF和300dpi PNG，标注SYNTHETIC TEST DATA，并检查色标、阶段选择、元数据、重复生成不覆盖及Figure关闭；模拟图不作为训练结果。Python3.8的 `compileall -q tools mmseg configs/hrda` 通过；本轮源码、配置、工具、测试和文档范围的 `git diff --check -- mmseg configs/hrda tools tests docs` 通过。全工作区diff检查仅报告既有、非本轮修改的 `train_doris_mitb5_stage1_20k.sh:22` 末尾空行，保留用户脚本不作无关改动。未新增依赖，未运行正式训练，未访问远端GPU，未自动commit/push。

以下命令仅供用户手动运行，本轮没有执行：

```bash
conda activate hrda
cd /root/HRDA-master
python tools/check_doris_stage1_configs.py

# 独立Smoke输出目录；仍然只采两个真实节点，不用于精度结论。
SMOKE_ROOT="/root/shared-nvme/work_dirs/doris_mechanism_smoke_$(date +%Y%m%d_%H%M%S)"
python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge.py \
  --work-dir "$SMOKE_ROOT" --options runner.max_iters=1000 checkpoint_config.interval=500

# 检查500、1000节点：同一sample_id/张量SHA，2个合并NPZ，4张在线PNG，CSV为2/10行。
# 提前中断后也可从已有数据单独生成候选图，不推理、不需要GPU。
python tools/build_doris_mechanism_figures.py "$SMOKE_ROOT/analysis"

# 可选：复用旧Fig.12工具读取实际训练统计，避免默认历史输出路径。
python tools/plot_goji_e1_process_curves.py \
  --input "$SMOKE_ROOT/analysis/metrics/mechanism_500.csv" \
  --out-dir "$SMOKE_ROOT/analysis/figures/process"
```

Resume检查：保持同一有效配置和work_dir；在500节点完整写出后手动中断，再加 `--resume-from "$SMOKE_ROOT/iter_500.pth"` 续训。不要把已经有不同1000步模型状态的目录拿来覆盖重跑；冲突检查会阻止覆盖。检查源/目标原型恢复和local_iter没有回到0，CSV无重复iter。

正式20K仍使用原批次/单条命令和原work_dir。需在真实FP16环境确认：deterministic probe实际文件读取、首节点前后模式/RNG、真实耗时/显存，以及T1/T2的8000/12000/14000等阶段。主开关 `--options uda.mechanism_logging=False` 关闭后不创建采集器、不增加probe前向；仅关闭fixed_probe_enabled可保留CSV。自动候选图依赖原始NPZ，make_figures=True时save_raw必须True。

## 最终交付判断

正常完成20K、输入可读且采集无异常的Full运行，将在500…20000产生40个统计节点和40个固定样本节点，随后自动生成三张候选汇总图。该数量已用模拟节点验证，尚不是远端40节点实跑证明。Baseline只保留基础日志/显存/验证。没有改变训练数学或超参数；必要Resume修复恢复的是既定时序与缓存，不重新定义算法。
