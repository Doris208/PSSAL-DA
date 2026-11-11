# Structural V2 公式与运行入口

RGB26→RGB30 的现有 Structural-only、MPARC、Full 三份配置统一启用 `struct_distinct_v1`，使用相同 Structural 参数及 Early-4K 调度。其余五组沿用各自模块组成。八组训练默认目录与启动脚本统一使用 `doris_struct_distinct_v2`。HRDA、MiT-B5、六通道、15K、优化器、数据和验证协议沿用现有配置。

## 响应与式（9）

生产实现位于 `mmseg/models/uda/struct_prior_gate.py`。

| 响应 | V1 | V2 与实现函数 |
|---|---|---|
| R_conf | Teacher 最大概率，作为融合外乘因子 | Teacher 最大概率，作为四项之一 |
| R_loc | 有效八邻域同类比例 | 保留 `compute_continuity_score` |
| R_adj | 含同类项的源邻接行与邻域分布内积 | `compute_cross_class_adjacency`，异类条件分布的 Bhattacharyya 匹配 |
| R_prob | 当前类别概率原型的余弦相似度 | `compute_probability_margin`，当前类相对最强有效竞争类的余弦差经 sigmoid 映射 |
| R_struct | confidence × 三项可用证据加权平均 | `fuse_distinct_responses`，四项可用证据归一化融合 |

去掉源邻接矩阵对角项后，记其异类质量为 $m_c=\sum_{j\ne c}A^s_{cj}$。仅已初始化且 $m_c\ge0.01$、$m_c>0$ 的行可用。对有效行归一化，再以 $\eta=0.10$ 混合异类均匀分布，得到 $\bar A^\times$。对角项始终为零。

$$
q^\times_p(j)=\frac{n_p(j)\mathbb I[j\ne c_p]}{\max(\sum_{k\ne c_p}n_p(k),\epsilon)},\qquad
R_{adj}(p)=\sum_{j\ne c_p}\sqrt{q^\times_p(j)\bar A^\times_{c_pj}}.
$$

没有异类邻居或源先验不可用时，邻接项 availability=False。数值接口填充值为1，融合、统计和显示均使用真实可用性掩膜排除该值。

$$
R_{prob}(p)=\sigma\left(\frac{\cos(P_t(p),\mu_{c_p})-
\max_{k\ne c_p,\,k\in\mathcal K_{ready}}\cos(P_t(p),\mu_k)}{0.15}\right).
$$

当前类别与至少一个竞争类别的源域概率原型必须可用。源 GT 邻接统计、源概率原型的更新方法沿用原实现，离线回放设置 `update_priors=False`。

式（9）建议对应以下 V2 融合，权重按 confidence/local/adjacency/semantic 顺序为 $(0.15,0.30,0.30,0.25)$。

$$
R_{struct}(p)=\operatorname{Clip}\left(
\frac{\sum_{q\in\{conf,loc,adj,prob\}}\lambda_q a_q(p)R_q(p)}
{\sum_q\lambda_q a_q(p)+10^{-6}},0,1\right).
$$

confidence 要求类别有效且响应有限，local 额外要求存在有效邻居。辅助贡献分解使用同一分母，包含 $C_{conf}+C_{loc}+C_{adj}+C_{prob}$。

## Early-4K、式（10）、式（12）与式（13）

`tools/struct_schedule.py` 的 `struct_gate_strength` 同时供训练门控、特征交互、在线探针和离线数值验证使用。

$$
\rho(t)=0.10+0.90\operatorname{Clip}(t/4000,0,1).
$$

Iter 0/1000/2000/3000/4000/6000/12000 的强度为 0.10/0.325/0.55/0.775/1/1/1。迭代指 DACS 的 `local_iter`，Checkpoint 中另存最后执行训练步与当前计数器对应强度。

式（10）沿用原解码器交互路径，输入响应为 $\rho(t)R_{struct}$，`gamma_struct=0.05`。式（12）沿用 Teacher 类别 $[2,3,4]$ 的 Building/Water/Road 掩膜 $M_s$。

式（13）的实现细节为

$$
g_0(p)=\max(R_{struct}(p),0.65)
\begin{cases}0.5 & R_{struct}(p)<0.32\\1 & \text{otherwise}\end{cases},
\quad
W_{after}=W_{before}\left[1-M_s\rho(t)(1-g_0)\right].
$$

软过滤发生在下限处理之后，因此调度满强度时可能得到0.325的门控值。辅助衰减 $A_{struct}=1-W_{after}/W_{before}$ 仅在有效结构类且 $W_{before}>0$ 上统计，其理论上界为 $0.675\rho(t)$。正式1×7最后一列继续采用 `struct_prediction`，来自选定状态的 Student 测试模式推理。

## 运行与历史兼容

```bash
python tools/train.py configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_struct_gate.py --work-dir "/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_struct_gate"
```

`tools/train.py` 将 CLI work_dir 写入 `cfg.work_dir` 和 `cfg.model.train_cfg.work_dir`。Runner、日志、Checkpoint、`analysis/fixed_samples` 均跟随该目录。原有两个批量训练脚本的对应 Structural 命令也使用此新目录。周期保存策略继续保留，机制边界保存使用实际 Early-4K 结束点。

在仓库根目录执行六样本回放，先设置真实 Baseline Checkpoint 的绝对路径。

```bash
STRUCT_WORK_DIR=/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_struct_gate
python tools/replay_struct_mechanism.py --config-struct "$STRUCT_WORK_DIR/doris_goji26RGB2goji30RGB_hrda_3vi_struct_gate.py" --checkpoint-struct "$STRUCT_WORK_DIR/iter_12000.pth" --checkpoint-baseline "${BASELINE_CHECKPOINT:?请设置真实Baseline权重的绝对路径}"
```

默认六样本列表保留，输出根目录为 `doris_struct_distinct_v2/goji26RGB2goji30RGB/struct_mechanism_26rgb2_30rgb`，每次创建含时间戳的新子目录。`--out-dir` 可显式指定新目录，已有结果拒绝覆盖。1×7正式响应显示有效结构类、真实[0,1]尺度和中心 `[30:226,30:226]` 裁剪。Student 预测完整显示五类，A_struct 只进入辅助诊断。

V1 使用旧实验目录保存的完整解析配置或 Git 历史配置，并显式指定旧 Checkpoint 和新输出目录。`legacy_product`、`doris_confidence_guided`、旧邻接、旧余弦和 `legacy_warmup_linear` 均保留。状态字典未增加参数或 buffer。当前 V2 配置不能作为旧模型的真实训练协议，离线配置一致性检查会拒绝该错配。

生产组件和小网络测试覆盖公式、调度、可用性、状态不变及在线/离线一致性。真实六样本、完整 MiT-B5 GPU 回放与15K性能比较由后续远端运行确认。需记录最佳 mIoU、最佳 NonVeg.mIoU 迭代及选定 Student Checkpoint 哈希，响应去冗余效果和精度提升需由真实结果判断。

## 用户提供的设计动机参考

空间关系、语义上下文与连续性方向可参考用户提供的 Ning 等、Iqbal 等及 Ni 等研究。此处的异类条件分布、Bhattacharyya 匹配、竞争余弦响应和可用性融合为本项目的具体实施方案。本文档沿用用户提供的文献信息。

- https://doi.org/10.1109/TGRS.2025.3590794
- https://doi.org/10.1016/j.isprsjprs.2023.10.020
- https://doi.org/10.1109/TGRS.2024.3379398

## 修改文件与本地验证

| 文件 | 修改用途 |
|---|---|
| `configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_struct_gate.py` | 当前训练入口启用 V2 和独立输出目录 |
| `mmseg/models/uda/struct_prior_gate.py` | 新邻接、语义、可用性融合与旧模式兼容 |
| `tools/struct_schedule.py` | 单一 Early-4K 与旧调度实现 |
| `mmseg/models/uda/dacs.py` | 特征交互和调度日志复用同一强度 |
| `mmseg/utils/mechanism.py` | 生产可用性采集、在线探针与元数据 |
| `mmseg/utils/checkpoint_policy.py` | V2 Checkpoint 保存调度配置与实际强度 |
| `tools/doris_checkpoint_plan.py` | 机制边界节点跟随真实 Early-4K 结束点 |
| `tools/replay_struct_mechanism.py` | V2 回放、权重与 availability 导出、路径和来源记录 |
| `tools/replay_hphy_mechanism.py` | 共用配置检查接纳 Structural 三种融合模式 |
| `tools/doris_struct_evidence.py` | 模式感知公式校验、四项贡献、分类统计与相关性 |
| `tools/build_struct_mechanism_figure.py` | 复用模式感知校验并保留严格权重检查 |
| `tools/struct_mechanism_panels.py` | V2 四项贡献图和分类 CSV，正式图继续1×7 |
| `tools/check_doris_stage1_ablation_configs.py` | 检查 RGB26→RGB30 三组 Structural V2 一致性及八组独立输出目录 |
| `scripts/run_doris_tableiv_26rgb_to_30rgb_hphy15k.sh`、`train_doris_mitb5_stage1_ablations_20k.sh` | 对应 Structural 命令采用新目录 |
| `tests/test_struct_distinct_v1.py`、`tests/test_struct_probe_cpu.py`、`tests/test_struct_mechanism.py`、`tests/test_struct_visualization.py` | 生产公式、调度、梯度、历史兼容、在线离线与绘图测试 |
| 本文件 | 公式、入口和验证范围对齐 |

本地针对性验证命令和结果如下。

```powershell
python -m pytest -q tests/test_struct_distinct_v1.py tests/test_struct_mechanism.py tests/test_struct_visualization.py tests/test_doris_checkpoint_policy.py::PlanTests
# 20 passed, 8 skipped
& 'D:\Anaconda3\envs\aglldiff\python.exe' tests/test_struct_distinct_v1.py -q
# 7 tests OK
& 'D:\Anaconda3\envs\aglldiff\python.exe' tests/test_struct_probe_cpu.py -q
# 2 tests OK
git diff --check
# exit 0
```

默认 Python 缺少 Torch/MMCV，相关测试跳过。现有另一个本地 Torch 环境完成生产组件和小网络探针测试。pytest 的两条 warning 来自字体依赖弃用，以及主动注入 Inf 的异常诊断测试。24 配置静态审计与真实 CLI work_dir 选择语句验证包含在上述测试中。完整 MMCV/MiT-B5 GPU 运行与真实数据效果尚待远端验证。
