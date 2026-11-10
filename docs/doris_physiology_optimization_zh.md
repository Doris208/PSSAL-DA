# DORIS 生理先验专项优化交付

> 最新观测输出见 [每500步在线机制快照](doris_online_mechanism_snapshots_zh.md)：生理启用组每500步立即保存对应PNG；balanced_high_conf监督与本页算法不变，源域VI静态分析仍只执行一次。

> 后续输出收敛见 [四分支输出交付](doris_branch_evidence_convergence_zh.md)：本页生理监督算法保持不变，独立PHY-1/PHY-2改为一张生理综合图；仍每500步采NPZ。现有DORIS消融尚未运行，可继续使用当前配置与原结果目录。

## 实际变更与边界

已在当前24份配置上原位修改，仅9个生理启用组开启新目标域监督：三个方向各自的`_phy_gate.py`、`_phy_struct_gate.py`、`_proto_struct_edge.py`。另外15个不启用生理的配置字节未变。没有复制新实验矩阵、没有fusion_v2或新增网络参数。

R_phy仍是植被相关生理可靠性，不是Goji专属概率。结构融合、边界融合、边界损失、tau_proto、ProtoGate源码和共享旧配置helper均未修改。除新增的生理模式参数和采集开关外，24份有效配置逐项哈希与修改前一致。

|文件|本轮改动|
|---|---|
|`mmseg/models/uda/phy_reliability.py`|新增纯损失函数target_phy_loss，legacy/均衡监督；头结构不变|
|`mmseg/models/uda/dacs.py`|配置读取、沿现有目标BCE反传路径调用、源分组及实际损失稀疏统计|
|`mmseg/vi_utils.py`|抽取训练已有VI处理公式，训练/数据分析共用|
|`mmseg/datasets/pipelines/loading.py`|现有VI读取/归一化方法调用上述函数，行为不变|
|`mmseg/utils/mechanism.py`|源GT/目标预测分组统计、降权比例、PHY专用NPZ、复用固定样本、结束时一次性制图|
|`tools/build_doris_physiology_figures.py`|PHY-1源域分析和PHY-2过程综合图，两张各PNG/PDF|
|`tools/build_doris_mechanism_figures.py`|compact模式不再生成独立生理strip，其他模块图保留|
|`tools/check_doris_stage1_ablation_configs.py`|完整phy_cfg对齐及9组参数/门控/采集开关核查|
|9份现有doris配置|新增balanced_high_conf参数和physiology_collection=True|
|`tests/test_doris_physiology.py`、`tests/test_doris_physiology_figures.py`|梯度、legacy、两图和数据分析测试|
|`tests/test_doris_mechanism.py`、`tests/test_doris_stage1.py`|实际训练路径、分组分母、只读探针和紧凑输出兼容|

## 1 新旧监督与实际训练路径

旧模式默认`legacy`：只用Teacher预测为{0,1}且confidence>=pseudo_threshold的像元作正类BCE，保持旧数值与梯度，包括旧掩膜作用范围。

新模式`balanced_high_conf`：

```text
valid = 非ignore、类别0..4、valid_pseudo_mask及既有裁边内
positive = valid & Teacher类别∈{0,1} & confidence>=0.968
negative = valid & Teacher类别∈{2,3,4} & confidence>=0.968

L_pos = 正样本BCE之和 / 正样本数
L_neg = 负样本BCE之和 / 负样本数
L_phy_tgt = 0.008 × (0.5 L_pos + 0.5 L_neg)
```

只存在一组时将存在组系数归一化为1；两组都空时返回与预测图相连的零损失。低置信像元不转成负类，目标GT不参与。Teacher类别/置信度和空间掩膜均detach，BCE以float32计算并关闭局部autocast。

不额外调用预测头。复用_apply_phy_gate返回的r_phy_tgt，在原来的目标损失位置只调用一次loss_phy_tgt.backward。日志继续保留loss_phy_src和loss_phy_tgt，内部pos/neg诊断只进入采集数据，不进入MMSeg总loss自动求和。

源监督仍用源GT区分植被{0,1}与非植被，未改公式。终端loss_phy_src沿用原raw口径，实际反传乘0.08；loss_phy_tgt是已经乘0.008的值。

## 2 九份配置的有效参数

以下三个目录中，每个目录的Physiology、MPARC、Full三份配置相同：

```text
configs/hrda/doris/goji30IRRG2goji26IRRG/
configs/hrda/doris/goji26IRRG2goji30RGB/
configs/hrda/doris/goji26RGB2goji30RGB/

每目录：
doris_<方向>_hrda_3vi_phy_gate.py
doris_<方向>_hrda_3vi_phy_struct_gate.py
doris_<方向>_hrda_3vi_proto_struct_edge.py
```

```python
phy_target_supervision_mode = 'balanced_high_conf'
phy_pos_weight = 0.5
phy_neg_weight = 0.5
tau_conf_phy = 0.968
lambda_phy_src = 0.08
lambda_phy_tgt = 0.008
detach_target_gate = True
phy_min_weight = 0.82
phy_filter_threshold = 0.30
phy_filter_mode = 'soft'
phy_warmup_iters = 6000
phy_rampup_iters = 4000
```

只通过各文件phy_cfg.update覆盖，共享旧helper默认值没有改，因此旧实验仍默认legacy。不启用生理的组未开启其他分支。基础训练、结构/原型/边界参数保持原方向设置。

## 3 生理门控没有变化

```text
a = 0.82 + 0.18 R_phy
若R_phy<0.30，a再乘0.5（soft filter）
rho = clip((rule_iter-6000)/4000, 0, 1)
gate = (1-M_phy) + M_phy × ((1-rho)+rho×a)
W_t_phy = W_before_phy × gate
Delta W_phy = W_t_phy - W_before_phy
```

M_phy由Teacher伪标签确定；W_before_phy是已有DACS高置信比例权重及空间掩膜后的值，不是逐像元W_conf。这一实际乘性门控与旧论文式(8)不是完全相同的写法；没有为了制图重新按理想公式计算。warmup指门控，生理头在此之前已经通过辅助监督学习。

运行时`analysis/reports/formula_code_mapping.md`已补充新旧监督、真实门控和字段定义，不修改PDF/Word。

## 4 PHY-1：一次性源域VI证据

训练正常结束后，由现有Collector.finish调用一次源域分析；也可用同一工具单独执行。不在500步节点反复读取数据集。

- 仅cfg.data.train.source的训练影像和源GT；支持--source-root指定本地镜像，拒绝test/val目录。
- 和训练共用PIL单通道读取、float32、非有限值置0、逐指数clip/range归一化。NDVI/NDRE原始clip[-1,1]，CIRE[0,5]，生理头输入均映射[0,1]；图上明确原始范围和归一化口径。
- 原始1..5标签减1，0/255忽略；未知标签报错。严格核对同名影像、VI、GT及二维大小；TIFF地理标签存在时核对，不存在时明确记录“空间配准未证明”，不能仅凭同名同形状宣称地理范围已验收。
- 默认seed=0、最多128个训练tile、每tile每类最多256像元、每原始影像组每类最多2048。先按tile限额抽样，再对同类的非空影像组取相同数量，避免大面积场景主导。
- 默认文件名移除末尾四个坐标识别父影像ID；不能识别时以tile文件名为组，不宣称已完成物理影像级分组。可传--group-regex，第一捕获组指定真实父影像。实际分组、数量和参数写CSV。
- 每类每指数count/mean/sample std/median/Q1/Q3，保留分组行和汇总行；空类为0计数与空数值，图上No data。
- Goji/Vegetation的三单指数、三种双指数及三指数联合都使用同一个Fisher trace ratio：||mean_G-mean_V||² / trace(Cov_G+Cov_V)。一维时等于常见Fisher score；不是把三个一维score相加。不增加J-M和冗余显著性指标。
- 输入文件元信息、配置、采样协议、工具/归一化代码哈希用于缓存；相同输入重复运行不读像元、不重绘。协议/数据变化不静默覆盖现有文件。

仅四份输出：PHY-1的PNG/PDF和两个CSV。箱线图隐藏离群散点，但分位数和统计包含真实抽样值，不丢掉离群值来美化结果。分布与可分性只是光谱证据，不直接证明最终分割收益或独立互补性最优。

本地训练目录为空，目前没有真实源数据统计；仅模拟流程通过。真实图将在数据可读环境运行后产生。

## 5 PHY-2：40份数据，一张过程综合图

沿用现有sample manifest、相同目标无标签训练样本、确定性预处理及只读no_grad/eval/RNG保护。每500步只保存生理NPZ，不输出40张生理PNG。

NPZ字段：r_phy、m_phy、w_before_phy、w_t_phy、delta_w_phy、valid_mask、target_image、rule_iter、metadata。只保存二维响应/掩膜及必要输入显示，不保存特征。

纯生理组不再生成另一份通用NPZ；MPARC/Full的通用NPZ继续服务其他模块，但去掉r_phy/m_phy/w_t_phy/delta_w_phy专属重复字段。其他模块PNG保留，生理PNG strip和独立生理泛化演化图停用。

结束后PHY-2选取接近2000、warmup起点、ramp结束、训练75%和最终的4–5个真实节点，完整20K默认2000/6000/10000/15000/20000；短训练或中断只有现有节点，选取记录写入PNG/PDF元数据。列为Target/R_phy/W_before_phy/W_t_phy/Delta；静态输入只显示一次，M_phy用白色轮廓。响应/权重固定[0,1]，差值固定[-1,1]，无单图min-max或对比度增强。

最终生理候选图只有PHY-1和PHY-2，不添加第三/第四张。Full仍有结构/边界/原型候选图，它们不算生理单独候选图。

## 6 CSV新增字段及分母

复用analysis/metrics/mechanism_500.csv，仍声明sample_interval=100与真实n_sampled_batches，完整窗口通常5个采样batch，不称为全部500步严格平均。

|字段|口径|
|---|---|
|r_phy_source_vegetation_mean / nonvegetation_mean|源GT分组，累计响应和/累计组内像元数|
|r_phy_target_vegetation_pred_mean|Teacher预测为0或1的有效像元响应均值，不是类别概率|
|r_phy_target_nonvegetation_pred_mean|Teacher预测为2、3、4的有效像元响应均值|
|r_phy_target_goji_pred_mean / othervegetation_pred_mean|可选细分0/1预测类，不解释为Goji/Veg分类概率|
|上述对应_count|实际累计分母，零分母均值留空|
|w_before_phy_mean_valid|与已有w_base_mean_valid一致，所有有效目标像元为分母|
|w_t_phy_mean_valid / delta_w_phy_mean_valid|调制后与差值，累计分子/有效像元数|
|phy_downweighted_count|有效Teacher植被预测中W_after<W_before的像元数；W_before=0且未实际变化不计|
|phy_vegetation_count|有效Teacher植被预测数量，不要求达到监督置信门槛|
|phy_downweight_ratio|上述降权数量/phy_vegetation_count，不是占全部有效区比例|
|loss_phy_src / loss_phy_tgt|采样batch的实际标量loss平均；源raw、目标含lambda，与既有日志口径相同|
|loss_phy_pos / loss_phy_neg|按各自组计数加权累计的raw BCE均值，只作CSV诊断|
|phy_pos_count / phy_neg_count|新监督实际通过置信度、空间/类别规则的计数|
|phy_target_supervision_mode / tau_conf_phy / phy_pos_weight / phy_neg_weight / lambda_phy_src / lambda_phy_tgt|末采样实际配置，便于审查|

## 7 输出、Resume与运行

```text
实际work_dir/analysis/
  metrics/mechanism_500.csv
  fixed_probe/sample_manifest.json
  reports/formula_code_mapping.md
  physiology/
    data/phy_vi_statistics.csv
    data/phy_vi_separability.csv
    arrays/iter_000500_phy.npz ... iter_020000_phy.npz
    figures/fig_phy_vi_distributions.png/pdf
    figures/fig_phy_response_evolution.png/pdf
```

CSV/NPZ继续保留去重和模型状态、配置、样本冲突检查。新数据schema=3。既有schema=1保留原目录并沿既有规则使用嵌套采集目录；schema=2/新旧监督配置不一致时不会强行合并。**用户已确认当前doris_消融尚未运行：直接使用已更新配置首次训练，沿用现有work_dir即可，无需另建目录。9组是3个方向各自的Physiology/MPARC/Full，不是新增9种实验。若以后比较已有legacy结果，再单独区分监督版本。**Baseline和其余未改组是否复用仍须训练协议一致。

源分析发生在正常after_run；中断后可单独从已有NPZ制PHY-2，并单独补一次PHY-1。这里只读数据，不重新推理。若旧版采集用了analysis/module_collection_v2，手动工具需指向对应analysis根；已存在不同配置的历史结果时才需独立work_dir；当前尚未运行的doris_实验沿用原定目录。

用户在云端手动执行的最小验证，本轮没有执行：

```bash
conda activate hrda
cd /root/HRDA-master
python tools/check_doris_stage1_ablation_configs.py
CFG=configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py
SMOKE_ROOT="/root/shared-nvme/work_dirs/doris_stage1/goji26RGB2goji30RGB/phy_balanced_smoke_$(date +%Y%m%d_%H%M%S)"
python tools/train.py "$CFG" --work-dir "$SMOKE_ROOT" \
  --options runner.max_iters=1000 checkpoint_config.interval=500

# 检查两个phy NPZ、无逐节点生理PNG、CSV中pos/neg计数及有限loss。
# 1000步时门控仍处于warmup，Delta应为0，不代表新辅助loss没有反传。
python tools/build_doris_physiology_figures.py "$CFG" --work-dir "$SMOKE_ROOT"
# 也可 --only vi / --only response；本地镜像数据用 --source-root，采样使用固定seed。
```

完整20K直接使用24组现有命令和原定输出目录；当前尚未运行，不存在本次需重跑或更换目录的问题。需在真实GPU上确认AMP/FP16、新模式pos/neg计数和loss、6000/10000门控阶段、真实数据文件匹配及耗时。没有真实数据时不宣称图已证明生理先验有效。

## 8 本地验收

23项针对性测试通过：5项生理loss/梯度、14项采集/安全/分母/Resume、1项实际DACS训练路径小网络回归、3项两图/源统计/缓存/精简输出。两张模拟候选图均生成并视觉检查，带SYNTHETIC TEST DATA，不作真实结果。

真实MMCV Config.fromfile解析24份配置（本地静态检查关闭custom_imports）；9份生理配置完整字典一致，参数正确。哈希核对确认另外15份配置、ProtoGate、结构融合、边界loss和共享legacy helper字节未变；扣除本轮新增生理参数/采集开关后，24份有效配置全部与改前一致。Python3.8编译检查通过。

本轮源码、配置、工具、测试和文档范围git diff --check通过；全工作区检查只报告历史train_doris_mitb5_stage1_20k.sh第22行已有末尾空行，本轮未改这个脚本。未安装依赖，未访问/同步远端，未启动正式训练，未修改论文，未commit/push。
