# DORIS 四分支机制证据与输出收敛交付

> 在线保存策略已按新指令修订，见 [每500步在线机制快照](doris_online_mechanism_snapshots_zh.md)：21个非Baseline配置每500步立即生成PNG，中间组一张、Full两张，最终汇总为MPARC/PBCR。本页“0张在线PNG、训练后才绘图、Full四张最终图”为历史策略；分支统计定义与算法边界仍适用。

本轮仅调整观测与输出。现有24组配置继续使用原目录；按用户说明，这轮消融尚未运行，不需要另建训练版本或结果目录。9个启用生理的配置维持已有 `balanced_high_conf`，没有再次改动监督。

## 旧输出与当前默认输出

|旧输出项目|当前默认行为|
|---|---|
|分量级机制CSV、每类原型筛选CSV、生理BCE子项过程统计|收敛为 `metrics/branch_mechanism_500.csv`；详细接口保留，默认关闭|
|每500步分模块绘制PNG|关闭在线绘图；每节点只保存一个压缩NPZ|
|单独PHY-1、PHY-2以及可能重复的生理演化图|静态VI证据与生理权重变化合入 `fig_phys_branch`|
|结构分量各自过程曲线|只在一张综合图中保留四个组成响应与最终融合响应|
|完整原型相似度/矩阵/阈值柱状图|只画实际原型细化前、后和差值；保留可选筛选统计接口|
|E_prob、E_img、R_pair、L_pos、L_neg的单独过程图|不默认输出；模型内部计算及基础Loss日志保留|
|散布的分支快照|合并到 `fixed_probe/selected_arrays/iter_XXXXXX.npz`|
|旧分析文件|保留；发现旧schema时新收敛输出写入 `analysis/branches_v4`，不删除历史文件|

默认配置：`mechanism_output_mode='branches'`、`mechanism_detailed_output=False`。需要诊断时，可显式开启 `mechanism_detailed_output=True` 附加旧详细CSV；旧 `detailed` 模式与旧数据绘图读取仍保留，不自动执行。

## 分工及图数

三个方向均使用同一矩阵，单分支组不要求Full的四个模块全部开启。

|序号|实验|采集分支|最终候选综合图数|
|---|---|---|---:|
|1|Baseline|无|0|
|2|Physiology|phy|1|
|3|Structural|struct|1|
|4|MPARC|phy、struct|2|
|5|Prototype|proto|1|
|6|Boundary|edge|1|
|7|PBCR|proto、edge|2|
|8|Full|全部|4|

图数按完整、可读、采集正常的训练输出计算，每张各一份PDF及300dpi PNG。数据缺失不虚构。重复生成相同来源的图不覆盖；数据或绘图代码变化时保留旧图并写入版本子目录。

## 核心CSV字段与实际计算路径

|分支|核心字段|统计范围与含义|
|---|---|---|
|生理|`phy_response_mean_valid`、`phy_weight_before_mean`、`phy_weight_after_mean`、`phy_downweight_ratio`|有效Teacher植被预测像元，当前类别0/1；响应、实际门控前后权重，以及实际降权像元占比|
|结构|`struct_response_mean_valid`、`struct_weight_before_mean`、`struct_weight_after_mean`|有效Teacher结构类别，当前2/3/4；从训练路径取融合响应和实际输入/输出权重，Full的输入为生理处理后的权重|
|原型|`proto_weight_before_mean`、`proto_weight_after_mean`、`proto_target_selected_ratio`、`proto_target_update_ratio`|权重均值使用有效目标图像像元；筛选和更新比例使用实际特征网格上的有效候选数量|
|边界|`edge_response_mean_valid`、`edge_supervision_mean_valid`|按训练相同 `label_to_edge` 规则得到的有效Teacher类别交界像元；后者为实际连续边界权重，含域系数和warmup|

共同字段包含 `iter`、`experiment_name`、`enabled_branches`、`data_source`、`n_sampled_batches`、`sample_interval`、`window_start`。每个分支额外记录 `*_state`、`*_sampled_batches`、`*_valid_count`，用于说明真实有效分母。

原型额外保留 `proto_update_sampled_batches`、`proto_selection_valid_count`、`proto_selected_count`、`proto_updated_count`、`tau_proto`，不再默认写逐类筛选表。实际筛选仍用原配置的置信度条件、权重阈值及最少像元条件；没有改变 `tau_proto=0.25` 或 `tau_conf_proto=0.88`。

每100步从已有训练张量采样，500步汇总一次。均值按累计分子/有效分母计算，不平均各批次均值，也不声称覆盖全部500个训练批次。完整20K正常产生40行；每个完整窗口通常采样5个批次。

关闭、warmup、原型缓存未就绪、没有有效像元的指标写 `NA`，不会用0冒充观测。计数为0可以真实记录。分支处于ramping时纳入实际已开始作用的观测。生理门控warmup不代表生理辅助头没有训练。

原型目标缓存更新时序与权重细化时序分别判断：筛选/更新比例跟随实际 `target_update_start`，权重指标跟随 `weight_start` 和缓存可用性。不能把“缓存更新尚未启动”和“未选中任何像元”混为一谈。

训练统计处理ignore=255和空间有效掩膜。原型候选分母为有效特征网格数，不能与图像像元分母混用。Resume可产生不足500步的部分窗口，`window_start`、实际批次数和分母如实反映；同一已保存节点不重复追加，内容冲突会报错以保护已有数据。

## 四张候选图的数据来源

全部动态图来自同一个确定性目标训练样本的固定探针，标记 `fixed_probe_recomputed`，与CSV的 `training_batch_actual` 分开。探针不读取目标GT；保留样本身份、输入张量SHA、配置SHA、模型状态SHA及实际规则步数，恢复RNG、模块模式和推理临时状态，原型/结构先验只读。

训练批次统计没有增加训练前向；固定探针仍按既定500步频率做独立只读观测前向。

|文件名|图块与来源|
|---|---|
|`fig_phys_branch`|源域训练GT分组的NDVI/NDRE/CIRE统计中位数与IQR；固定目标图、R_phy、实际生理门控前后权重、差值|
|`fig_struct_branch`|固定目标图、R_conf、R_loc、R_adj、R_prob、R_struct，全部复用现有融合函数；选择已采节点中先验可用的较晚节点|
|`fig_proto_branch`|固定目标图、原型细化实际输入权重、输出权重、差值；优先保留首次生效前后节点|
|`fig_edge_branch`|固定目标图、Teacher伪标签生成的真实边界、R_edge、正边界连续监督权重|

边界最后一列准确含义是“正边界连续可靠性监督权重”，不是二值边缘目标，也不是完整L_edge的数值图。负环监督及正负边界Loss仍按原逻辑训练，但不单独成图。连续图使用 `[0, 配置目标域边界系数]` 的固定色标，各节点不分别归一化；普通响应/权重统一0到1，差值统一-1到1。

生理静态证据复用已有源域采样/归一化统计：只额外保存 `physiology/data/phy_vi_statistics.csv` 和 `phy_vi_separability.csv`，默认不生成独立生理图片。像元统计不是独立实验重复，IQR不是置信区间；无法读取真实数据时不得编造统计。

默认代表节点为2000、6000、10000、15000、20000的已采近邻。原型最多5节点，优先保留真实激活前后及最后节点，不将未生效阶段的无变化解释成模块无效。保存的NPZ仍保留warmup时真实前后权重和状态；CSV的有效作用指标为NA。

统一NPZ保留按启用状态需要的二维响应、真实权重/差值、有效掩膜、Teacher伪标签/边界、先验可用标记、输入展示图及溯源元数据。不会保存高维中间特征；E_prob/E_img/R_pair和重复的独立生理NPZ不进入默认保存白名单。

Full图用于解释协同，单分支图用于必要的独立效果补证。性能贡献需在同迁移方向、同训练协议下通过Baseline→单分支、Baseline→MPARC/PBCR、MPARC/PBCR→Full的真实验证结果判断，不能凭热图明显程度下结论。

## 文件修改范围

- `mmseg/utils/mechanism.py`：分支统计、真实权重/边界快照、压缩白名单、按启用状态汇总、报告和最终绘图调度。
- `mmseg/models/uda/dacs.py`：最小输出模式读取、详细诊断开关和边界观测辅助方法；没有改动训练目标或反向传播。
- `tools/build_doris_mechanism_figures.py`：复用现有工具，新增分支模式分发与四张综合图；旧图读取保留。
- `tools/build_doris_physiology_figures.py`：静态分析增加仅保存数据的调用方式，采样及统计规则不变。
- `configs/hrda/doris/<direction>/doris_*`：21个非Baseline配置明确声明分支输出模式、关闭默认详细输出；3个Baseline不启用机制采集。
- `tools/check_doris_stage1_ablation_configs.py`：核查输出模式，避免把输出选项当算法差异。
- `tests/test_doris_branch_outputs.py`：分支统计、NA、快照、去重、40节点最终图覆盖。
- `tests/test_doris_mechanism.py`、`tests/test_doris_physiology.py`：详细输出测试显式选择旧模式，保留兼容验证。
- `tests/test_doris_physiology_figures.py`：补查仅数据模式不会生成独立PNG/PDF。
- 本文及旧机制/生理说明页的更新提示。

本轮没有修改PhyReliabilityHead、结构融合、ProtoGate、R_edge、StructEdgeLoss、HRDA网络或推理。10个受保护算法/日志/数据文件与任务开始快照SHA完全一致；24份有效配置剔除观测选项后SHA全部一致。DACS的最小观测调整另由轻量训练替身测试检查采集开关前后梯度和状态一致。

原有50步基础训练日志、1000步验证、Loss、Time/Data Time、显存MiB、五类IoU、mIoU、Veg/NonVeg mIoU、aAcc及已有评价指标、best/latest checkpoint策略、随机种子和完整配置均保持原协议。没有为本次输出收敛新增评价算法。

## 输出示例

实际位置始终跟随 `runner.work_dir`；`--work-dir`覆盖后全部分析文件也随之迁移。

```text
/root/shared-nvme/work_dirs/doris_stage1/goji26RGB2goji30RGB/
  goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge/
    *.log / *.log.json / *.pth / 有效配置
    analysis/
      manifest.json
      metrics/branch_mechanism_500.csv
      metrics/probe_cost.csv
      fixed_probe/sample_manifest.json
      fixed_probe/selected_arrays/iter_000500.npz ... iter_020000.npz
      physiology/data/phy_vi_statistics.csv
      physiology/data/phy_vi_separability.csv
      figures/fig_phys_branch.pdf / .png
      figures/fig_struct_branch.pdf / .png
      figures/fig_proto_branch.pdf / .png
      figures/fig_edge_branch.pdf / .png
      figures/branch_figure_selection.json
      reports/branch_evidence_mapping.md
```

完整20K每个开启采集的实验预计40个汇总节点、40个合并NPZ、0张在线PNG；最后按启用分支生成最多4张综合图，各PDF+PNG。`probe_cost.csv`是必要的探针开销记录，不是额外分量级机制曲线。

## 本地验证与边界

已通过：24组 `Config.fromfile` 有效配置及24条批次命令审计；28项观测/训练替身/日志/分组IoU轻量测试；2项绘图及源域数据统计测试。绘图使用明确标注SYNTHETIC的模拟40节点，不代表真实训练结果；检查Full四图、原型单分支一图、激活前后选点、300dpi、重复生成、无在线PNG，人工查看排版。

验证含500/1000节点、7种启用组合、Baseline拒绝虚假机制、实际结构输入、ignore和分母、探针模式/RNG/参数/缓存/梯度保护、Resume节点去重、原型checkpoint恢复、Loss/显存日志与既有评价指标兼容。显存日志测试采用CUDA替身，没有测量真实GPU占用；checkpoint恢复为轻量测试，没有在本地正式保存/恢复完整HRDA训练。

本地配置审计关闭自定义模型导入（基础Python缺Torch），使用已有MMCV解析配置；Torch轻量检查在已有环境运行。尚未进行真实数据/FP16/GPU端到端训练，不宣称精度或真实运行耗时已验收。未安装依赖、未连接远端、未同步、未自动训练、未修改论文、未commit/push。

Python3.8针对本轮源码、配置和测试的编译检查通过，源码/配置/工具/测试/文档范围的 `git diff --check` 通过（Git只提示既有文件LF/CRLF转换）。

本地复核命令（PowerShell，使用已有环境，无须安装）：

```powershell
Set-Location D:/PSSAL-DA
$env:PYTHONPATH='D:/PSSAL-DA/tests;D:/PSSAL-DA'
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
& D:/Anaconda3/envs/aglldiff/python.exe -m unittest test_doris_branch_outputs.BranchOutputTests test_doris_mechanism test_doris_physiology test_doris_stage1.DorisCoreTests.test_full_dacs_train_step_with_cpu_standins_keeps_masks_and_legacy_updates test_logger_data_time test_goji_group_iou -v

$env:PYTHONPATH='D:/PSSAL-DA/outputs/no3vi_check_deps;D:/PSSAL-DA/tests;D:/PSSAL-DA'
python -W 'error:constrained_layout not applied:UserWarning' -m unittest test_doris_branch_outputs.BranchFigureTests test_doris_physiology_figures.PhysiologyFigureTests.test_source_sampling_empty_classes_normalization_joint_score_and_cache -v
python tools/check_doris_stage1_ablation_configs.py --skip-custom-imports
```

## 用户按需手动Smoke命令

以下仅供后续手动运行，本轮没有执行。正式实验仍使用原配置和原结果目录；Smoke可使用独立临时目录以免1000步测试混入20K结果。

```bash
conda activate hrda
cd /root/HRDA-master
python tools/check_doris_stage1_ablation_configs.py

SMOKE_ROOT="/root/shared-nvme/work_dirs/doris_branch_smoke_$(date +%Y%m%d_%H%M%S)"
python tools/train.py configs/hrda/doris/goji30IRRG2goji26IRRG/doris_goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge.py \
  --work-dir "$SMOKE_ROOT" --options runner.max_iters=1000 checkpoint_config.interval=500

# 自动汇总已在正常结束时执行；单独重建不会增加模型前向。
python tools/build_doris_mechanism_figures.py "$SMOKE_ROOT/analysis"
```

检查500/1000两行CSV、两个NPZ、没有在线PNG、最多四个综合图。此时生理/原型可能未激活，应显示warmup/NA，不代表分支无效。

Resume可在500节点完整保存后中断，再以同一配置、同一work_dir加 `--resume-from "$SMOKE_ROOT/iter_500.pth"` 继续，检查CSV/NPZ不重复。不要在已经完成1000步的同一目录倒退重跑覆盖不同状态；冲突检查会保留已有证据。

正式20K的真实激活、真实TIFF读取、FP16状态保护与GPU开销留给用户云端验证，不需要为本次输出调整更换正式结果目录。
