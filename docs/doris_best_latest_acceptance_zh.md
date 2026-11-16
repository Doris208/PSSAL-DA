# DORIS 四指标 Best、滚动 Latest 与三个固定检查点

## A. 修改文件

- `tools/doris_checkpoint_plan.py` 统一配置现有三个方向、二十四组实验，指定三个固定检查点及完整 resume 的提前校验。
- `mmseg/utils/checkpoint_policy.py` 将 DORIS Hook 改为每 2000 Iter 原子滚动更新 latest，额外固定保留 Iter 2000、4000、6000 的完整状态。
- 新增 `mmseg/utils/doris_checkpoint.py`，负责文件锁、事务日志、一次序列化与硬链接、失败回滚、重启恢复、文件所有权、RNG 和 latest 原子替换。
- `mmseg/core/evaluation/goji_best.py` 在既有单卡/分布式验证结果上执行四指标独立择优，同时保留其它配置的历史行为。
- `mmseg/apis/train.py`、`tools/train.py` 在恢复前检查完整状态，恢复 MMCV 1.4 未恢复的 runner.meta，并校正完成迭代计数及 DACS.local_iter。
- `tools/replay_doris_mechanisms.py` 支持通过显式完成迭代元数据识别滚动 latest。四种 Best 文件名继续兼容现有回放接口。
- `tools/check_doris_stage1_ablation_configs.py` 审计二十四组实际保存策略和训练协议的独立性。
- `tools/smoke_doris_hphy.py` 准备完整 HRDA/DACS 模型的四指标评价、Best/latest 保存及加载检查。
- 新增 `tests/test_doris_best_latest.py`，更新 `tests/test_doris_checkpoint_policy.py` 和 `tests/test_doris_mechanism.py`。
- 本报告及历史策略文档中的入口说明。

## B. 保存策略对照

| 项目 | 原策略 | 当前策略 |
|---|---|---|
| 周期保存 | 每 2000 Iter 保留完整历史文件 | 每 2000 Iter 更新同一个 latest.pth |
| 固定节点 | 配置分析派生额外 Checkpoint | 固定保留 iter_2000.pth、iter_4000.pth、iter_6000.pth |
| 训练结束 | 强制保存最终历史文件 | 按 2000 Iter 周期更新 latest |
| 最优指标 | mIoU 与 Goji IoU | 四指标独立择优 |
| Best Optimizer | 包含 | 默认省略 |
| 续训入口 | 历史完整 Checkpoint/latest | 完整 latest 或三个固定检查点，使用原 CLI |
| 成功保存后的托管权重数量 | 随迭代增加 | 至多四个 Best、一个 latest 和三个固定检查点，共 8 个 |

按照每 2000 Iter 的指定更新频率，15K 训练结束时 latest 的完成迭代数为 14000。Best 各自对应实际刷新指标的 Iter。

用户追加的三个固定检查点将此前每实验五文件上限更新为八文件上限。达到对应 Iter 并成功保存后，固定文件持续保留。

## C. 四项监控字段

| 真实评价字段 | 文件名 |
|---|---|
| `IoU.goji` | `best_goji_mIoU_iter_{iter}.pth` |
| `mIoU` | `best_mIoU_iter_{iter}.pth` |
| `Veg.mIoU` | `best_Veg_mIoU_iter_{iter}.pth` |
| `NonVeg.mIoU` | `best_NonVeg_mIoU_iter_{iter}.pth` |

沿用 GojiDataset 的五类与组指标定义、缺失类别处理和每 1000 Iter 的一次验证。每个指标独立接受第一次有效分数和严格提升，0 有效，相等、缺失、NaN、Inf 均保持原记录。新一次评价先清除四项旧日志字段，避免缺失字段误用前一轮值。

## D. Checkpoint 内容与恢复

Best、latest 和三个固定检查点都保存完整模型 state_dict，包括实际启用的 Student、EMA Teacher、Phy、Structural EMA、Prototype 动态缓存和 Edge 状态，以及配置、Git 信息、完成迭代数和必要的机制元数据。

latest 另外保存 Optimizer，包括 param_groups.initial_lr，沿用配置中的 poly/warmup 和完整 15K 调度范围。已有 meta.fp16 和 loss_scaler 状态保留。当前 DACS 使用其原有优化器与 FP16 路径，本轮保持这些实现。Python、NumPy、Torch 和可用 CUDA 的 RNG 状态按 rank 记录。相同 world_size 下恢复保存的流，world_size 改变时以明确日志提示沿用当前初始化的 RNG，模型与 Optimizer 继续恢复。

三个固定检查点与对应 Iter 的 latest 共享完整内容，包含 Optimizer、迭代计数、DACS.local_iter、模块状态、调度及混合精度恢复元数据和 RNG。可通过 `--resume-from PATH/iter_2000.pth` 等方式恢复。

MMCV 1.4 IterBasedRunner.resume 会恢复模型和 Optimizer，但会省略 runner.meta 恢复，并使用保存时的零基迭代计数。本轮恢复全部 meta，使用验证过的 completed_iter 修正 Runner.iter、inner_iter 和 DACS.local_iter。下一步 LR、EMA 和各模块进度按该计数继续。

Best 带有 resumable=False 和 save_optimizer=False。`--resume-from` 指向这类文件时会在训练前明确报错，提示使用 latest 续训或通过 `--load-from` 初始化模型权重。

## E. 存储、安全与去重

同一次验证刷新多个指标时，只调用一次 Runner.save_checkpoint。各指标路径优先使用同文件系统硬链接，分别可由 torch.load 读取。更新某指标时只清理当前管理器拥有且已过时的文件名，其它指标保持有效。

若硬链接创建失败，日志与 checkpoint_best.json 明确记录错误及 single_file_reference 模式。刷新指标通过各自的 JSON 记录引用同一个实际文件，requested_path 记录预期指标名路径，path 记录可实际加载的路径。该模式遵循用户允许的单文件多指标引用方案。回放时应采用 JSON 的实际 path。

每次先在当前目录写完整临时 Checkpoint，校验可读取性、完整 state_dict 键、张量形状/类型、实际 Iter 及 Optimizer 内容，再更新正式路径与 JSON。latest 使用 os.replace，旧 latest 在写入和校验完成前保持有效。Best 更新通过持久事务日志回滚失败操作，并在下次启动恢复未完成事务。同一 Iter 重新评估出现新 Best 时，也能恢复原路径并保护仍被其它指标引用的数据。

固定节点与 latest 共用一次序列化，优先硬链接。文件系统不支持硬链接时，记录明确日志并在空间检查后复制、校验完整文件，再原子发布固定路径。后续 latest 原子更新后，固定路径继续引用原节点状态。固定文件已存在时校验所有权、完成 Iter 和续训状态后保留。固定文件写入失败时，旧 latest 继续有效。

持久策略标识、文件身份、目录限定和文件锁共同限定管理范围。历史实验目录继续受空目录检查保护。恢复旧完整 Checkpoint 时可指定新的空工作目录，历史目录保留。原有二十四份模型、数据、损失和调度配置通过审计确认。

以完整模型张量大小 M、Optimizer 大小 O 表示，八文件的最大托管状态约为 8M+4O。同 Iter 四项 Best 共享时约为 5M+4O。固定节点刚保存时与 latest 硬链接可继续减少占用，后续 latest 更新后两者拥有独立状态。更新过程中还需要新的临时文件空间。

## F. 实际验证

本节的历史 PASS 记录对应增加三个固定检查点之前的版本。本轮追加实现与断言调整进行静态审查，保存和恢复的运行验证由远端完成。

- 保存策略、失败回滚、完整中断恢复、既有机制与回放联合回归通过 77 项，记录在 `outputs/best_latest_acceptance_tests.log`。
- 此前完成的 Checkpoint 专题复测为 14 PASS，见 `outputs/best_latest_final_tests.log`，包括全分支 Best 状态及临时 JSON 恢复检查。
- 真实 MMCV 配置审计通过 24/24，记录在 `outputs/best_latest_config_audit.log`。
- Python 3.8 语法及 Git diff 空白检查通过。

本地使用 PyTorch 2.3.0、MMCV 1.4.0 与仓库已有 outputs/hphy_deps。硬链接测试在允许该文件系统操作的环境运行。

```powershell
$env:TMP='D:\PSSAL-DA\outputs\irrg_test_tmp'
$env:TEMP=$env:TMP
$env:PYTHONPATH='D:\PSSAL-DA\outputs\hphy_deps;D:\PSSAL-DA\tests'
& 'D:\Anaconda3\envs\aglldiff\python.exe' -m pytest tests/test_doris_checkpoint_policy.py tests/test_doris_best_latest.py tests/test_baseline_prediction_cpu.py tests/test_doris_checkpoint_replay.py tests/test_doris_mechanism.py tests/test_irrg_probe_cpu.py tests/test_struct_probe_cpu.py tests/test_hphy.py -q -rs --disable-warnings
& 'D:\Anaconda3\envs\aglldiff\python.exe' -m pytest tests/test_doris_best_latest.py -q -rs --disable-warnings
& 'D:\Anaconda3\envs\aglldiff\python.exe' tools/check_doris_stage1_ablation_configs.py
```

中断恢复测试使用真实 MMCV Runner、AdamW 和 DORIS 分支小网络，在固定批次上保存、重新构建模型与优化器、从生产恢复入口载入并继续实际前后向。模型全部状态、Optimizer 张量、迭代计数、后续 LR 和模块进度与连续运行逐项一致。FP16 元数据恢复单独核对，CUDA 数值路径待真实设备验收。

2000、4000、6000 同时是当前固定保存位置和中断恢复检查位置。2000 对应第一次 latest 更新和零基计数检查，4000 对应 H-Phy/Structural Early-4K 满强度，6000 对应本轮 RGB26/IRRG26 Prototype 权重与目标辅助损失的启用进度。已有测试通过预设迭代计数后运行少量真实训练步骤覆盖这些边界。

## G. 真实设备 Smoke

当前本地 CUDA 不可用，4090/真实数据验收保留为待执行项。以下命令使用独立输出目录，执行完整 MiT-B5/HRDA/DACS 合成短运行。四个评价字段由模型的合成样本预测和合成标签经过原 GojiDataset 评价产生，报告标记 synthetic_validation。

```bash
python tools/smoke_doris_hphy.py --direction goji26IRRG2goji30RGB --experiments 1 2 3 4 5 6 7 8 --output /root/shared-nvme/doris_best_latest_smoke_20261113 --check-fixed-probes
```

## H. 历史兼容与在线机制

既有 Goji/mIoU Best 名称继续受支持，新增 Veg/NonVeg 名称使用相同迭代解析。最新策略的滚动 latest 通过保存的完成迭代元数据解析。旧配置回放继续使用其原模型与输入协议，历史文件保持原内容。

在线机制配置继续使用 sample_interval=100、fixed_probe_interval=500、summary_interval=500，以及原有 NPZ、CSV、PNG 等输出。各历史节点已采集的 NPZ 可以独立渲染。离线模型回放可选择实际保留的 Best、latest 或三个固定节点。

## I. 15K 结束后的示意目录

以下 Iter 仅用于展示四项指标可分别在不同时间择优。

```text
experiment/
  best_goji_mIoU_iter_8000.pth
  best_mIoU_iter_12000.pth
  best_Veg_mIoU_iter_10000.pth
  best_NonVeg_mIoU_iter_14000.pth
  latest.pth
  iter_2000.pth
  iter_4000.pth
  iter_6000.pth
  checkpoint_best.json
  .doris_checkpoint_policy.json
  .doris_checkpoint.lock
  analysis/
```

JSON 保留 mIoU、IoU.goji、same_iter 等兼容字段，并包含四项指标、配置哈希、Git、验证来源、存储共享关系和真实文件标识。成功完成事务后，临时模型和事务日志已清理。

## J. 运行入口

原训练与恢复选项继续使用。为新实验提供独立的空工作目录。

```bash
python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py --work-dir /root/shared-nvme/work_dirs/doris_four_best_latest/goji26IRRG2goji30RGB/full

python tools/train.py configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py --work-dir /root/shared-nvme/work_dirs/doris_four_best_latest/goji26IRRG2goji30RGB/full --resume-from /root/shared-nvme/work_dirs/doris_four_best_latest/goji26IRRG2goji30RGB/full/latest.pth
```

## 静态复查补充

按最新要求，后续运行验收交由远端完成。本轮静态复查补充修正了两处问题。

- Best 序列化时同步本次刷新指标的兼容 hook_msgs，包含真实 score、Iter 和对应模型路径。
- 旧权重的文件指纹变化时，通过原记录的 SHA256 再确认归属，避免磁盘重新挂载后漏清理已过时 Best。有效文件名始终保留，删除仍限定在当前管理范围。

上述两处最后修正已完成静态复查，其运行结果待远端验证。前述 PASS 数量对应修正前已完成的测试记录。

## 配置覆盖

`tools/train.py` 在记录有效配置前调用 `configure_checkpoint_policy`，`mmseg/apis/train.py` 的训练 API 入口再次应用同一策略并注册唯一保存 Hook。原始公共配置的周期设置经入口覆盖后，运行配置使用上述八文件策略。

26IRRG→30RGB 的 Baseline、Physiology、Structural、MPARC、Prototype、Edge、PBCR、Full 八组均匹配，包含名称带 `_hphy_early4k` 的三组。现有逐组和批量脚本都经过 `tools/train.py`。同一规则同时覆盖另外两个方向，共二十四组 DORIS HRDA 消融配置。

增加固定节点后的 MMCV 配置解析审计已完成 24/24，使用 `tools/check_doris_stage1_ablation_configs.py --skip-custom-imports`。该检查加载并合并配置、应用生产入口的策略函数，核对四项 Best、2000 Iter latest、三个固定节点、1000 Iter 验证及批量脚本命令。此结果属于配置静态检查，实际保存与续训仍待远端运行验证。
