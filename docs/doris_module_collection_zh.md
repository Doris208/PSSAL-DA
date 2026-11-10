# DORIS 消融实验按模块分工采集 V2

> 最新输出策略见 [每500步在线机制快照](doris_online_mechanism_snapshots_zh.md)：六个中间组与Full均在线保存，Full每节点MPARC/PBCR两张；不再沿用训练结束才绘图的策略。

> 当前默认输出以 [四分支输出交付](doris_branch_evidence_convergence_zh.md) 为准：按模块采集的分工保留，但不再每500步画PNG，独立PHY-1/PHY-2已合并，详细CSV默认关闭。

> 生理分支后续更新：9组已启用balanced_high_conf辅助监督，生理改为500步NPZ及最终PHY-1/PHY-2两图，不再逐节点生成生理PNG；以[doris_physiology_optimization_zh.md](doris_physiology_optimization_zh.md)为准。其他模块定义保留。

本页替代此前“编号2–7仅基础日志、只有Full进行机制采集”的限制。此次修改只扩展观测，不改变模块实现、训练损失、阈值、更新时序、优化器、数据管线或训练命令。

## 采集矩阵

三组迁移方向均采用同一规则，编号2–8开启采集。每100步从已有训练张量取少量统计，每500步累计分子/分母写CSV，并对一个固定的目标域无标签训练样本做只读探针。完整20K每个启用采集的实验有40个固定样本节点和40个合并NPZ。

|编号/配置后缀|对应机制内容|在线PNG/节点|最终候选图数量（各PDF+PNG）|
|---|---|---:|---:|
|1 Baseline|基础训练、显存、验证日志；不造模块证据|0|0|
|2 `_phy_gate`|R_phy、DACS基础权重、生理后权重及差值|1|2：生理图、权重演化|
|3 `_struct_gate`|四结构分量、R_struct、结构前后权重及差值|2|2：结构图、权重演化|
|4 `_phy_struct_gate`|生理证据、结构证据、MPARC前后权重|3|3：生理图、结构图、权重演化|
|5 `_ps_proto`|原型输入/输出权重及差值、五类筛选/更新计数|1|2：权重演化、原型覆盖|
|6 `_struct_edge`|E_prob、E_img、R_pair、R_edge和实际边界监督权重|1|2：边界图、边界演化|
|7 `_pbcr`|原型权重/覆盖、边界证据|2|3：边界图、权重演化、原型覆盖|
|8 `_proto_struct_edge`|上述所有启用机制，用于组合一致性分析|4|5：生理图、结构图、边界图、权重演化、原型覆盖|

Full不再是唯一的证据来源。原型独立作用应优先使用编号5，边界独立作用使用编号6，结构独立作用使用编号3；组合组用于检查模块组合后的行为。时间演化和中间响应本身不等同于分割性能的因果证明，性能仍看相应消融验证结果。

## 原型独立组的准确含义

`doris_<方向>_hrda_3vi_ps_proto.py`不启用生理/结构模块。因此保存并显示的是：

`Target | W_conf | W_in^proto | W_t^proto | Delta W_proto`

- `W_conf`：Teacher逐像元最大类别概率。
- `W_base`：实际DACS批次高置信像元比例形成的基础权重，含有效区域遮罩；它不等于逐像元W_conf。
- `W_in^proto`：真正进入ProtoGate前的权重。原型独立组/PBCR组等于W_base，不能称作“经过MPARC”。
- 有生理/结构模块时，`W_in^proto`对应已经调制的`W_t^PS`，图中使用该符号。
- `Delta W_proto = W_t^proto - W_in^proto`，采用固定[-1,1]，不增强差值。

原型启动前照常保存真实权重：输出等于输入、差值为0，标记warmup。达到加权时序而缓存类别不足时标记cache_unready，实际开始加权后标记active；不能将cache_unready称为已生效。五类筛选CSV仅在原型启用的编号5/7/8生成，保留候选、联合筛选和真正更新数量及原有分母定义。

每500步从早期持续采集，不只在模块启动后采集。候选演化图优先保留原型实际生效前后相邻节点，以及既定更新/加权时序附近节点。例如26RGB→30RGB的5500/6000，以及其他方向11500/12000或13500/14000；缓存推迟就绪时以实际首次active为准。metadata中的`rule_iter`是探针使用的模型计数，`iter`是完成训练步数；训练批次统计与同节点重算探针仍分开标注。

## 生理、结构和边界

生理保存`r_phy/w_t_phy/delta_w_phy`，差值相对W_base。结构保存`r_conf/r_loc/r_adj/r_prob/r_struct`及先验可用性；生理或结构启用时保存`w_t_ps/delta_w_ps`。只有部分MPARC组件时，该权重是实际启用组件后的结果，不代表完整MPARC。

边界独立组保存`e_prob/e_img/r_pair/r_edge/boundary_weight`，不伪造原型后权重。`boundary_weight`与训练规则一致：R_edge×Teacher可靠性×有效掩膜×目标域系数×边界warmup比例。原图保留真实数值，既有Fig.12工具仅在绘图时将边界权重乘100。类别对先验未初始化时原始R_pair按真实规则为0，图片明确标注prior uninitialized。

生理/结构的warmup指目标权重调制，不能理解为此前head/源先验不学习。边界损失从训练初期按原有比例ramp，并无被本轮新增的“硬启动”；图中记录ramping/active，不虚构边界启动前训练节点。

## 文件、状态与不生效模块

每个实验在实际`runner.work_dir/analysis/`下输出：

```text
manifest.json                     # schema_version=2，modules，时序等
metrics/mechanism_500.csv          # 所有编号2–8；每500步一条
metrics/proto_selection_500.csv    # 仅原型启用时；每节点5类
metrics/probe_cost.csv
fixed_probe/sample_manifest.json
fixed_probe/arrays/iter_000500.npz
fixed_probe/snapshots/iter_000500_physiology.png          # 按启用模块出现
fixed_probe/snapshots/iter_000500_struct_components.png
fixed_probe/snapshots/iter_000500_weight_stages.png
fixed_probe/snapshots/iter_000500_edge_components.png
figures/fig_*.pdf、fig_*.png、figure_selection.json
reports/metric_definitions.md、formula_code_mapping.md
```

CSV增加`phy/struct/proto/edge_enabled`及对应`*_state`。状态包括disabled、warmup、ramping、cache_unready、active。未启用模块的统计值留空，NPZ中不保存其响应键，也不调用其计算方法；0仅用于确有计算、结果为0的量。

可用均值列为：`w_conf/w_base/r_phy/w_t_phy/w_t_ps/w_pre_proto/w_t_proto/w_final/r_struct/e_prob/e_img/r_pair/r_edge/boundary_weight/delta_w_phy/delta_w_ps/delta_w_proto`，均加`_mean_valid`后缀；只有对应量存在时填写。公共有效区域、样本批次数、采样频率及原型计数字段沿用原定义，原型未启用时阈值和原型统计留空。每类计数仍在真实特征网格，不是图像网格。

既有Fig.12过程曲线工具同步支持部分模块：只绘制存在的数据，新增生理后/原型后权重曲线，未启用模块不填零、不绘线；500步节点按0.5、1、1.5等正确标注横坐标。训练批次曲线依然拒绝混入固定探针数据。

历史schema=1的Full-only数据不覆盖：检测到它时，新的V2采集写入同一work_dir的`analysis/module_collection_v2/`。对应自动图表也跟随这个目录。新V2目录仍执行配置、样本和状态冲突保护，Resume不混写不同结果。

## 状态保护与性能

只读no_grad/eval、Teacher/Student参数/BN、先验、原型、优化器和RNG保护继续生效。只有原型启用时才额外提取Student特征；没有结构/原型/边界的配置不访问对应缓存。训练统计复用已有边界分量和各阶段权重，不为统计增加网络前向，不开启旧高频调试图，不保存高维特征。

每500步仍记录probe耗时和显存；PNG数量按上表×40计算，NPZ每节点合并一次。Full比旧版增加生理和边界图，实际开销应由云端probe_cost.csv衡量。既有输出根目录、24条命令和训练参数未变。

## 本地必要验证

20项针对性轻量测试通过：12项采集测试（其中一项覆盖七种模块组合）、2项训练/legacy数值回归、3项Logger测试、3项绘图/曲线测试。验证禁用模块不被计算、原型单独组启动前权重相同、启用模块正确保存、所有组合的在线/最终图数量、实际原型生效前后阶段、空字段曲线不填零、历史schema保护，以及模型状态/梯度/RNG/续训去重。模拟图均为测试数据，不是实验结果。

24个有效配置经真实MMCV Config.fromfile解析并核对模块开关和与对应Full的参数一致性；本地静态解析关闭custom_imports，不等同于完整模型导入/GPU训练验收。Python3.8编译和本轮文件差异检查结果见最终交付说明。

本轮未连接或同步远端，未安装依赖，未运行正式训练或自动commit/push。云端沿用新24条训练命令即可；例如只验证原型采集：

```bash
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_ps_proto.py \
  --work-dir /root/shared-nvme/work_dirs/doris_stage1/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_ps_proto
```

手动从保存数据重绘（无GPU、无模型推理）：

```bash
python tools/build_doris_mechanism_figures.py WORK_DIR/analysis
# 历史schema=1已存在时，改用WORK_DIR/analysis/module_collection_v2
```
