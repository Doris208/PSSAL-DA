# DORIS 每500 Iter在线机制快照交付

本页替代此前“默认仅NPZ、训练结束才绘图”的输出策略。保持当前算法、监督、损失系数、阈值、20K协议和正式结果目录，只修改观测工程。

## 原因与修复路径

此前 `MechanismCollector.consume()` 已在500、1000等节点调用快照，但 `snapshot_branches()` 只保存压缩数组，没有渲染PNG；图像构建调用位于 `finish()`，而Logger只在训练达到max_iters后调用它。不是四分支漏启用，也不是500步计数错误。

现在路径为：

```text
after_train_iter
  → MechanismSnapshotHook（ABOVE_NORMAL）
  → 已有PrecisionLoggerHook创建的同一个MechanismCollector.consume()
  → completed_iter = runner.iter + 1
  → 500整除：汇总CSV、执行一次固定探针、保存NPZ、立即保存PNG
  → 验证必需文件并记录completed
  → 原CheckpointHook / EvalHook / IterTimerHook / PrecisionLoggerHook
```

观测Hook在验证之前运行，不等待1000步验证结束。原Logger优先级保持不变；启用新Hook时Logger不再重复consume。只有rank 0的Logger创建采集器，其他rank不写分析文件。探针和绘图开销从IterTimer统计中扣除，CUDA峰值记录不重置。

Full两张图共用一次探针及同一个NPZ，不为MPARC、PBCR分别推理。每节点CPU绘制160dpi PNG，约1728×464像素；不在线生成PDF/SVG。

## 24组开关与七种布局

三个方向各8组，21个非Baseline配置均明确开启：

```python
mechanism_logging=True
mechanism_sample_interval=100
mechanism_summary_interval=500
fixed_probe_enabled=True
fixed_probe_interval=500
mechanism_output_mode='branches'
mechanism_detailed_output=False
mechanism_save_raw=True
mechanism_online_plot=True
mechanism_strict=True
mechanism_make_figures=True
```

`mechanism_online_plot`统一控制在线PNG，不再添加功能重复的`mechanism_save_png`开关；`mechanism_make_figures`保留为训练结束汇总开关。Baseline不启用采集，不生成虚构机制图。所有布局由实际启用分支决定，不根据文件名推断。

|组别|每节点后缀|图中内容|20K在线PNG数|
|---|---|---|---:|
|Physiology|`phys`|Target、R_phy、实际生理调制前后权重、差值|40|
|Structural|`struct`|Target、R_conf、R_loc、R_adj、R_prob、R_struct|40|
|MPARC|`mparc`|Target、R_phy、R_struct、MPARC实际输入/输出权重、差值|40|
|Prototype|`proto`|Target、原型细化前后权重、差值|40|
|Boundary|`edge`|Target、Teacher伪边界、R_edge、连续正边界监督权重|40|
|PBCR|`pbcr`|Target、原型细化前后权重、R_edge、连续正边界监督权重|40|
|Full|`mparc`和`pbcr`|上述两张模块综合图，同一状态、同一固定样本|80|

每个500节点正常保存一个NPZ、对应PNG、训练批次统计行及探针开销行。早期warmup/cache_unready/ramping均照常保存，并显示分支状态；不存在的响应标N/A。原型元数据保留目标更新阶段、有效类别数量、tau_proto/tau_conf_proto、min_pixels和启用时序；新原型图的简短脚注也显示缓存类别数、更新状态及阈值。

所有输入图使用已有归一化的逆变换，RGB/IRRG按实际通道标记。响应/权重固定[0,1]，差值固定[-1,1]；边界监督固定[0,该实验目标域边界系数]，不会逐图min-max放大。灰色表示无效区域或先验不可用。边界图展示连续正边界监督权重，不冒充二值边缘或完整L_edge；负环Loss保持原训练逻辑。

## 目录、来源及完成状态

保留既有 `selected_arrays` 路径以兼容已保存数据，没有为改名复制第二套NPZ。所有路径跟随实际 `runner.work_dir`，包括命令行覆盖。

```text
work_dir/analysis/
  manifest.json
  metrics/branch_mechanism_500.csv
  metrics/probe_cost.csv
  fixed_probe/sample_manifest.json
  fixed_probe/selected_arrays/iter_000500.npz
  fixed_probe/selected_arrays/iter_001000.npz
  fixed_probe/snapshots/iter_000500_mparc.png
  fixed_probe/snapshots/iter_000500_pbcr.png
  fixed_probe/snapshots/iter_001000_mparc.png
  fixed_probe/snapshots/iter_001000_pbcr.png
  figures/fig_mparc_summary.pdf / .png
  figures/fig_pbcr_summary.pdf / .png
  figures/online_figure_selection.json
  reports/branch_evidence_mapping.md
  reports/online_snapshot_policy.md
```

此示例为Full。其他组每节点只有对应后缀的一张PNG。

`branch_mechanism_500.csv`仍为 `training_batch_actual`：每100步对实际训练张量采样，累计有效分子/分母，每500步汇总。未改变既有核心字段、NA规则或统计范围。

恢复旧运行时不覆盖已有说明文件，新增 `reports/online_snapshot_policy.md` 明确当前在线策略，避免旧报告中的“不在线画图”描述造成混淆。

NPZ和PNG为 `fixed_probe_recomputed`：同一个无GT目标训练样本、确定性裁片和预处理。PNG元数据包括迭代数、分支状态、样本/配置/模型状态标识、对应NPZ的SHA以及 `render_timing=after_train_iter`。它们不是该节点随机训练batch的重放，不要求数值与训练CSV相等。

`manifest.json`保留静态配置身份，增加 `snapshot_policy` 和逐节点 `snapshots`。每节点含iter、样本ID和SHA、模型状态SHA、分支、相对文件路径、完成/失败状态、耗时及数组SHA。只有NPZ和全部必需PNG存在、非空，且PNG能打开、来源元数据一致，才标记completed。PNG和NPZ均采用临时文件完成后替换，减少半写文件。

绘图异常记录明确的失败Iter、原因和traceback，manifest标记failed，保留已成功保存的原始数组及部分图片。默认严格模式立即抛错；显式设置 `mechanism_strict=False` 可记录错误后继续。失败不能伪装成completed。若对一个有效历史节点进行冲突操作，原completed记录保留，失败尝试另记，不覆盖有效历史数据。

## Resume与正在运行的T3 Full

本地更新不会改变正在运行的远端Python进程，本轮没有操作远端。用户可以继续旧进程，也可在完整checkpoint保存后自行停止、手动同步新代码，再以原配置、原work_dir和真实checkpoint恢复。

新增的两个输出选项不进入旧观测配置身份计算；旧schema-4 manifest允许增添快照策略和节点记录，不强制换目录。训练/样本相关配置变化仍触发冲突。Resume恢复已有DACS local_iter，下一节点按真实迭代数计算；从1000恢复后下一保存节点为1500，500/1000不重复写、不改有效历史NPZ/PNG。

已经保存的旧NPZ可以只读生成图片，无需checkpoint或GPU；图片必须标记离线从已保存数组绘制。已下载目录仅有真实500步NPZ，因此本轮只导出了500步的两张独立预览，没有制造1000步响应。真实训练目录、原NPZ、日志和manifest均未修改。

预览在 `outputs/doris_t3_full_saved_500_online_preview/`，PNG明确包含 `render_timing=offline_from_saved_arrays`，不能称为当时的在线PNG。只有1000步checkpoint不能恢复500步响应；本地未下载完整checkpoint，不能核验EMA/BN/原型/邻接/边界先验是否齐全，也不宣称已严格完成checkpoint历史重建。本轮没有新增猜测缺失状态的恢复工具。

## 最终候选图与选点

在线策略的Full最终默认输出 `fig_mparc_summary`、`fig_pbcr_summary`，各PDF和300dpi PNG。中间组输出与实际功能对应的一个汇总。工具从已保存NPZ排版，不增加模型前向，也不替代在线保存。

所有40节点均允许保存。2000/6000/10000/15000/20000仅为默认候选近邻；涉及原型时优先保留实际激活前后。用户可用 `--iterations` 指定任意真实已保存节点，例如3500/7500/11500/15500/19500。请求不存在节点会报错，不生成假节点。记录实际选点、来源数组、来源SHA、绘图代码SHA；不同选点或代码版本使用子目录，保留原候选图。

旧“训练后四分支图”读取接口仍可读取旧manifest，避免破坏已有成果。一次性源域VI统计工具保留，不在每500步重算。

## 修改文件

- `mmseg/models/uda/dacs.py`：仅读取在线绘图/严格模式两个输出开关。
- `mmseg/utils/mechanism.py`：在线调用、原型观测元数据、节点状态、错误和Resume保护。
- `mmseg/utils/precision_logger.py`：已有采集器的提前调度Hook，防止重复consume；原Loss/评价日志逻辑不变。
- `mmseg/apis/train.py`：在线模式注册提前调度Hook。
- `tools/build_doris_mechanism_figures.py`：七种布局、在线PNG、两个模块最终汇总、任意选点、旧NPZ只读预览。
- `configs/hrda/doris/<direction>/`：21个非Baseline配置添加两个输出开关，3个Baseline保留。
- `tools/check_doris_stage1_ablation_configs.py`：核验21组在线开关/500间隔及Baseline关闭，不把观测选项当算法差异。
- `tests/test_doris_online_snapshots.py`：真实文件、七种布局、调度、失败、状态安全和Resume测试。
- `tests/test_doris_mechanism.py`：旧schema迁移测试显式关闭在线模式，保持测试目的不变。
- 本说明及旧输出说明页的新版提示。

没有修改生理/结构/原型/边界算法，balanced_high_conf、融合系数、tau_proto、优化器、损失、数据管线、网络和推理逻辑保持原样。24份有效配置排除观测字段后与修改前一致；本轮DACS差异只有两行开关读取。日志源码改动仅涉及观测调度，不修改Loss或IoU计算。

## 本地验证

验收实际生成文件，不只检查函数是否被调用。小Tensor使用已有Torch环境，Agg使用已有Matplotlib环境；本地两套环境分离，端到端保存测试仅在测试桥接处以子进程绘图，生产代码直接调用绘图函数，无子进程。未安装依赖、未运行正式训练或GPU推理。

- 499/500/999/1000/1499/1500模拟计数，仅500/1000/1500触发；观测Hook在验证前执行。
- 七种启用组合均在训练期调用中真实生成500、1000节点的PNG/NPZ/CSV/manifest，Full两张、中间组一张；无在线PDF。
- 原型/生理/结构warmup仍出图；保留真实无变化权重。图像元数据与NPZ迭代、SHA、状态一致。
- 探针前后参数、梯度、优化器状态、EMA、BN及先验缓存不变，Python/NumPy/Torch RNG及异构模块模式不变。
- Full同一1000节点复用不重新推理、不改文件mtime，随后1500正常保存；CSV无重复。
- 严格模式失败抛错、非严格模式告警继续，manifest均failed；空图列表不能标completed。
- 旧schema-4升级输出开关可续用原analysis；改变seed仍拒绝混写。
- 模拟40节点，从3500/7500/11500/15500/19500生成两张最终汇总；七种布局与Baseline零图均验证。
- 保留原有Loss、显存、分组IoU、原型checkpoint恢复及DACS梯度/状态一致性检查。

实际500/1000测试文件位于 `outputs/doris_online_synthetic_*/<group>/`。中间组每组有两个NPZ、两个PNG、两行分支CSV和两条completed记录；Full前两节点为四张PNG，Resume测试另追加1500节点两张PNG。测试数据明确为synthetic，不作为论文性能证据。

本轮最终测试为33项Torch/Hook/日志轻量检查和3项Agg/静态VI绘图检查，共36项通过（包含最后单独补验的历史节点状态冲突保护）；24份有效配置及原24条总脚本命令审计通过。实际保存验收目录为 `outputs/doris_online_synthetic_w0t88jjk/`，40节点候选选点/布局验收目录为 `outputs/doris_online_summary_synthetic_rf0yrcc_/`。Python3.8编译和工作区diff空白检查通过；没有进行真实GPU耗时验收。

## 用户手动验证命令

以下仅提供命令，本轮没有在远端执行。正式实验继续使用原结果目录；1000步Smoke可使用独立测试目录避免混入正式结果。

```bash
conda activate hrda
cd /root/HRDA-master
python tools/check_doris_stage1_ablation_configs.py

SMOKE_ROOT="/root/shared-nvme/work_dirs/doris_online_smoke_$(date +%Y%m%d_%H%M%S)"
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py \
  --work-dir "$SMOKE_ROOT" --options runner.max_iters=1000

# 500步应已有mparc/pbcr两张PNG；1000步累计四张，NPZ两个。
ls "$SMOKE_ROOT/analysis/fixed_probe/snapshots/"

# 正式20K以后，从任意真实采集节点重排两张论文候选图，无需GPU。
python tools/build_doris_mechanism_figures.py /实际work_dir/analysis \
  --iterations 3500 7500 11500 15500 19500
```

正在运行的T3 Full如选择更新后恢复，先由用户确认checkpoint已完整保存并停止旧进程，再手动同步，随后运行：

```bash
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py \
  --work-dir /root/shared-nvme/work_dirs/doris_stage1/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge \
  --resume-from /root/shared-nvme/work_dirs/doris_stage1/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge/iter_1000.pth
```

从已保存原始数组补图的命令，要求独立预览目录，保留原分析成果：

```bash
python tools/build_doris_mechanism_figures.py /实际work_dir/analysis \
  --snapshots-only --iterations 500 --out-dir /独立预览目录
```

云端仍需确认实际GPU/FP16、真实数据读取及500步额外耗时。本地验证不代替4090正式运行验收。本轮没有连接、同步或修改远端，没有commit/push，没有修改论文和历史实验结果。
