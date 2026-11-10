# DORIS 历史 Checkpoint 与双指标 Best 保存策略

## 范围与版本核查

覆盖 `configs/hrda/doris/` 下三个方向、每方向八组，共 24 个配置。只在本地修改和轻量测试，未连接 SSH、未启动训练、未写入远端存储。
本次按用户追加要求，另将两个方向的原型时序提前；这是明确授权的训练时序变更，其余算法参数不变。

依据本地正式训练日志的环境信息、`mmseg/version.py` 和本地 MMCV 源码副本确认：**MMCV 1.4.0、MMSegmentation 0.16.0**。
已用真实 MMCV 1.4.0 `Config.fromfile` 检查 24 个有效保存计划，与轻量配置解析结果逐项一致。

旧实现：

- 三个方向公共配置均为 `checkpoint_config=dict(by_epoch=False, interval=1000, max_keep_ckpts=5)`。
- `train_segmentor` 原来仅通过 `register_training_hooks` 注册一个周期 CheckpointHook；没有第二套自定义权重清理器。机制图 Hook 不保存模型权重。
- MMCV 1.4.0 CheckpointHook 在保存后按 `max_keep_ckpts * interval` 清理更早的 `iter_*.pth`，故 20K 结束只留下 16000–20000 五组。
- 原 MMSeg EvalHook 复用 MMCV `_save_ckpt`，按验证 `mIoU` 严格提升保存 `best_mIoU_iter_N.pth`；Best 不在周期清理的文件名范围内。
- 该 MMCV 版本要求 `save_best` 为字符串或 None，不支持多指标列表。
- IterBasedRunner 的 Linux `latest.pth` 是指向最近编号文件的相对软链接；Windows 是复制。原逻辑均保留。Best 保存使用 `create_symlink=False`，不会将 latest 指向 Best。

## 生效方式和固定保存点

通过 `tools/doris_checkpoint_plan.py::configure_checkpoint_policy` 对现有 DORIS 实验名统一设置有效配置：

```python
checkpoint_config = dict(
    type='DorisCheckpointHook', by_epoch=False, interval=2000,
    max_keep_ckpts=-1, save_optimizer=True, save_last=True,
    key_iterations=[...])  # 由对应配置推导
evaluation['save_best_goji'] = True
```

**公共原始配置文件仍保留原值，DORIS 训练入口会统一覆盖为上述有效设置，并把有效配置写入新实验目录及 Checkpoint。**
这样不需要复制 24 份保存参数，也不会影响同一公共配置的非 DORIS 实验、DAFormer、MemoryAdaptNet 或监督学习组。
直接调用 `train_segmentor` 同样应用此策略。审核保存计划时应使用下方审计命令，而不是只查看公共基类配置。

所有 20K 组固定保留集合 P：

```text
2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000
```

仍只有一个周期／关键节点保存 Hook：用 P 与额外点的集合并集触发保存，`max_keep_ckpts=-1`，不执行历史周期清理。
它在原验证 Hook **之后**保存，使同一节点的周期文件和 latest 带有刚更新的两个 Best 记录；不增加验证次数，不改变每 1000 Iter 的验证频率。

## 时序核查：计数含义

模型使用零基 `local_iter`，训练一步结束后递增。文件 `iter_N.pth` 表示已经完成 N 步、下一步使用 `local_iter=N`。
因此门槛 T 对应的 `iter_T.pth` 正好是执行该门槛步骤前的状态；不再额外保存几乎等价的 `iter_(T-1).pth`。

关键边界 T 的选择规则：在固定点中找边界前／后各 500 Iter 内、且位于正确一侧的点；没有则保存 T 和 T+500。最终与固定点去重。
报告中的“首次可能有效”是由代码确定的下界，不是承诺真实数据在那一步一定产生非零响应。

| 分支 | 当前参数（所有启用该分支的 DORIS 组） | 实际条件与首次可能有效 |
| --- | --- | --- |
| 生理头训练／响应 | `lambda_phy_src=.08`、`lambda_phy_tgt=.008`，目标置信门槛 `.968` | 生理响应前向不等待门控 warm-up；源监督从首步计算，目标监督依赖有效高置信样本 |
| 生理伪标签门控 | `phy_warmup_iters=6000`、`phy_rampup_iters=4000` | local_iter=6000 的 ramp 仍为 0；local_iter=6001（完成 Iter 6002）起可能改变类别 0、1 的权重；local_iter=10000 达到全幅 |
| 结构伪标签门控 | `struct_warmup_iters=6000`、`struct_rampup_iters=4000` | 同样从 local_iter=6001 起可能改变类别 2、3、4 权重；还依赖预测响应、邻接／语义先验有效性 |
| 结构特征交互 | `feature_interaction_warmup_iters=6000`、`feature_interaction_rampup_iters=4000` | `use_feature_interaction=True` 时按同样 ramp 调整交互强度；6000 时仍为零 |
| 源域原型 | `source_update=True`、`min_pixels=8` | 首步即可尝试按真实源标签更新；对应类别必须有足够特征像素，不能假定首次成功更新 Iter |
| 目标域原型 | 各方向见下表；`tau_conf_proto=.88`、`tau_proto=.25`、`min_pixels=8` | 到达 update_start 后仍需同时满足置信度、前置权重、类别像素数；最早可能在完成 Iter `update_start+1` 成功更新 |
| 原型权重／损失 | 各方向见下表；`min_valid_classes_for_weight=3`、`min_valid_classes_for_loss=3` | local_iter >= start，且至少三个有效原型类别；损失还需有效标签及正权重。仅到达门槛不能保证差异或非零损失 |
| DORIS 边界监督 | `warmup_iters=5000` | 是 `min(1,local_iter/5000)` 的线性增幅，不是前 5000 步完全关闭；完成 Iter 2 起可能非零，local_iter=5000 达全幅 |
| 边界类别对先验 | 配置声明 `boundary_pair_warmup_iters=3000` | 当前 DORIS `doris_reliable + valid_boundary_only` 路径在旧 ramp 之前直接返回，因此此参数不产生一个独立的 3000 激活边界；仍依赖 pair prior ready 和有效边界 |
| 邻域 patch 历史 | 全部 24 组均关闭 | 不凭空增加额外节点或状态保存 |

源依据：`dacs.py::_apply_phy_gate`、`forward_train` 的 feature_interaction 调度、`_struct_edge_warmup_ratio`、`_build_doris_boundary_pair_response`；`struct_prior_gate.py::apply`；`proto_gate.py::update_target/compute_weight/loss_target`。

## 用户追加要求后的原型时序

以下变更同时应用于每方向的 `_ps_proto`、`_pbcr`、`_proto_struct_edge` 三组。

| 方向 | target_update_start | weight_start | loss_start | 本次变化 |
| --- | ---: | ---: | ---: | --- |
| A：ZN_26 RGB → HN_30 RGB | 3000 | 6000 | 6000 | 保持 |
| B：ZN_26 IRRG → HN_30 RGB | 4000 | 8000 | 8000 | 从 8000／14000／14000 提前 |
| C：HN_30 IRRG → ZN_26 IRRG | 3000 | 6000 | 6000 | 从 8000／12000／12000 提前 |

没有改变三个方向各自的原型损失系数、门控最小权重、边界损失系数等其他参数。
**旧实验回放必须使用旧 Checkpoint 对应的历史配置。** 新时序不会修改旧权重；离线回放入口的配置一致性检查仍会拒绝以新时序解释旧 Checkpoint。

## 每个实验的最终保存计划

完整配置名为 `doris_<方向>_hrda_3vi<后缀>.py`。A/B/C 对应目录分别为 `goji26RGB2goji30RGB`、`goji26IRRG2goji30RGB`、`goji30IRRG2goji26IRRG`。
每格列出 **额外节点**；每个实验最终保留 **P ∪ 该格节点 + Best mIoU + Best Goji IoU**，latest 按原平台方式管理。

| 配置后缀 | 启用模块 | A 额外节点 | B 额外节点 | C 额外节点 |
| --- | --- | --- | --- | --- |
| 无（baseline） | 无 | 无 | 无 | 无 |
| `_phy_gate` | 生理 | 6500,10500 | 6500,10500 | 6500,10500 |
| `_struct_gate` | 结构 | 6500,10500 | 6500,10500 | 6500,10500 |
| `_phy_struct_gate` | 生理、结构 | 6500,10500 | 6500,10500 | 6500,10500 |
| `_ps_proto` | 原型（实际不启用生理／结构） | 3000,3500,6500 | 4500,8500 | 3000,3500,6500 |
| `_struct_edge` | 边界（实际不启用结构门控） | 5000,5500 | 5000,5500 | 5000,5500 |
| `_pbcr` | 原型、边界 | 3000,3500,5000,5500,6500 | 4500,5000,5500,8500 | 3000,3500,5000,5500,6500 |
| `_proto_struct_edge` | Full 四分支 | 3000,3500,5000,5500,6500,10500 | 4500,5000,5500,6500,8500,10500 | 3000,3500,5000,5500,6500,10500 |

完整逐配置参数、事件、work_dir 和排序后的 `all_iterations` 在 `outputs/checkpoint_policy_20261109/config_audit_mmcv.json`。
审计可在项目环境单独运行，不启动训练：

```bash
python tools/doris_checkpoint_plan.py --out outputs/checkpoint_plan_review.json
```

## Best mIoU、Best Goji 与 latest

`GOJI_CLASSES=('goji','vegetation','building','water','road')`；Goji 是 reduce-zero-label 后类别 0，原始标签 ID 1。指标直接取现有 `CustomDataset.evaluate` 输出的 **`IoU.goji`**，值域 [0,1]。
不改标签、ignore_index、IoU 公式或原有结果的小数精度，不用 `Veg.mIoU` 代替。

- Best mIoU：继续由 MMCV 原 `_save_ckpt` 的比较／命名／序列化逻辑保存。
- Best Goji：同一次验证后独立比较 `IoU.goji`，严格提升才保存 `best_goji_mIoU_iter_N.pth`。0 是有效值；缺失、NaN、Inf 或越界值跳过；相等不更新。
- 两个指标独立维护分数、Iter 和路径。写入任一新 Best 之前，先把本次两个指标的新记录放入 `runner.meta['hook_msgs']`，保证从任一 Best 恢复时不丢掉同次验证的另一项记录。
- 不做第二次验证或额外推理；单 GPU、分布式验证均复用同一结果。只在原验证流程保存，不读测试集指标。
- 两个 Best 即使同 Iter 也保存为独立文件，保持最简单的加载兼容性。容量预估已包含两份；不额外建立易断开的 Best 软链接。
- 启动前已经存在的 Best 文件受保护，不删除；本次运行新产生的旧 Best 在更优文件成功保存后才移除。写入失败不会先删除上一份 Best。
- 周期／关键节点完全不清理。Best 的旧文件更新不影响 `iter_N.pth`。
- latest 始终指向最近周期／关键节点，不表示最优性能；它包含 optimizer 和当时的双指标最佳记录，用于恢复训练。

每次验证生成 `<work_dir>/checkpoint_best.json`，分别报告：

```text
mIoU:     score / iter / path / checkpoint_present
IoU.goji: score / iter / path / checkpoint_present
same_iter: true / false / null
goji_history_start_iter: 独立 Goji Best 记录起点
```

旧 Checkpoint 没有 Goji 最佳记录时，从本次恢复位置开始累计并明确记录起点，不把后续首次观察值冒充此前全部训练的 Goji 最优值，也不会伪造此前缺失权重。
MMCV 原有的零基 `meta.iter` 不改写；新增 `meta.doris_checkpoint.completed_iter`，使 Windows 的普通文件 latest 也能由现有恢复入口正确恢复已完成步数。

## 机制状态完整性与恢复

继续保存整个 UDA 包装模型的 `state_dict`，而非只保存 Student：

- `model.*`：Student；`ema_model.*`：EMA Teacher。
- `phy_head.*`：生理头参数。
- `struct_gate.source_adj_ema / source_adj_initialized`、`semantic_proto_ema / semantic_proto_initialized`：结构历史先验。
- `proto_gate.prototypes / valid_proto / proto_counts`、`source_prototypes / source_valid_proto`、`target_prototypes / target_valid_proto`：融合原型及两个域的历史缓存。
- `struct_edge_pair_prior / struct_edge_pair_prior_ready`：边界类别对历史先验。
- optimizer、配置与 Iter 等 meta 仍通过原 runner 保存。

这些已是参数或注册 buffer，不重复序列化，不添加新模型 buffer，也不改变权重格式。已有 ProtoGate 动态空 bank 的兼容加载逻辑继续使用。
邻域 patch 字典没有纳入 `state_dict`，但当前 24 组均未启用，无须扩展保存。原 batch 特征、随机裁剪／ClassMix、逐 batch 更新诊断不要求长期保存；固定 TIF 在历史状态下重新前向。
因此当前配置具备机制历史状态保存条件；仍不宣称逐位复现原随机训练过程。

测试使用 Full 配置构造的真实分支组件与小 backbone，验证 Teacher、Student、已填充源／目标原型及结构／边界 buffer 全部逐值恢复；不是完整 HRDA GPU 训练验收。

## 保护旧实验与远端存储容量

按用户说明，实际保存到远端挂载存储；本地 D 盘容量不作为运行限制。本次没有访问远端存储。
入口在写配置／日志前拒绝向非空目录启动新训练；续训须显式 `--resume-from`。已有最终 `iter_20000.pth` 的目录直接拒绝复用，应另选新 work-dir。
恢复点之后已经有 Checkpoint 的目录也会报错，避免从旧节点覆盖未来历史。原配置的默认 work-dir 不改，不跨实验放置文件。

本地完整旧实验样本文件为 1,374,291,443 字节，约 **1.28 GiB**，仅用于量级估算，不作为本任务的模型权重替代品。

| 范围 | 预计权重份数（含两个 Best） | 以 1.28 GiB／份估计 |
| --- | ---: | ---: |
| 单个 baseline | 12 | 15.36 GiB |
| 单个 Full | 18 | 23.04 GiB |
| 三个 Full | 54 | 69.12 GiB |
| 全部 24 组 | 352 | 450.53 GiB |

Linux latest 软链接不额外占一份权重；Windows 复制需再加一份／实验。分支结构和 optimizer 会影响实际体积；还应为日志、机制图、保存时内存／临时空间及受保护的旧 Best 预留余量。
保存前检查当前目标文件系统空闲空间，不足时报错；挂载配额造成的写入错误仍会直接抛出，不通过删除历史文件来腾空间。

## 当前真实结果与历史日志限制

本次不训练，因此**没有新产生的正式 Best mIoU 或 Best Goji 权重**，不能给出新时序实验的最优 Iter／数值。
以下仅是本地已有验证日志片段的最高观测，记录在 `outputs/checkpoint_policy_20261109/historical_best_observations.json`，包含日志路径和行号：

| Full 方向 | 本地已观察 Best mIoU | 本地已观察最高 Goji IoU | 两项日志峰值同 Iter |
| --- | --- | --- | --- |
| A | 11000：79.43% | 11000：83.95% | 是 |
| B | 1000：38.50% | 1000：65.81% | 是 |
| C | 1000：58.37% | 1000：63.48% | 是 |

B/C 的本地日志没有覆盖完整 20K，不能将上述数值报告成远端完整实验的最终最高值。本地对应 Best mIoU 文件缺失或不可读；过去尚无独立 Best Goji 文件。
因此无法宣称这三组历史正式 Best 均已在本地验证可加载。新保存路径沿用各配置 work_dir，命名为 `best_mIoU_iter_N.pth` 和 `best_goji_mIoU_iter_N.pth`；后续使用既有离线回放脚本核验实际文件。

## 修改与验证

新增：`tools/doris_checkpoint_plan.py`、`mmseg/utils/checkpoint_policy.py`、`mmseg/core/evaluation/goji_best.py`、`tests/test_doris_checkpoint_policy.py`、本说明。
修改：`tools/train.py`、`mmseg/apis/train.py`、`mmseg/core/evaluation/eval_hooks.py`；两个方向各三份原型配置；`tests/test_doris_stage1.py` 中对应的期望时序。
既有离线回放代码未因本次任务改写。

验证包括：24 组真实 MMCV 配置解析；固定／关键节点去重与全保留；单／分布式验证只调用一次；双 Best 独立提升、零值、相等、NaN、旧元数据和恢复；latest 更新与零基 Iter 兼容；旧文件保护及空间／写入异常；Full 分支状态逐值恢复；原有机制及 Goji 指标回归。
所有测试均使用轻量模型或模拟 runner，没有执行完整 20K 训练。
两个本地解释器合计 **45 项不同测试通过**；Python 3.8／3.11 编译与 `git diff --check` 通过。
与本轮修改前源码哈希清单核对，既有文件仅上述十处发生变化；模型、损失、数据、优化器与学习率实现未变。20 个本地历史 Checkpoint 的 SHA256 均保持不变。
详细记录：`outputs/checkpoint_policy_20261109/scope_and_validation.json`。
