# PSSAL-DA 默认最小输出：修改与验证记录

日期：2026-11-05。工作目录：`D:\PSSAL-DA`。工作分支：`codex/minimal-output`。
起始 HEAD：`74bd5f283ab85ce100f92ad583cb8fab56222d21`。

已实际修改本地代码，未提交、未推送，未连接远端、安装依赖或删除历史实验。原有暂存内容保持不变。此前修复的验证后 `KeyError: data_time` 逻辑保留，并加入回归测试。

## 1. 默认行为

正式实验使用 `uda.output_mode='minimal'`；`paper_process_logging=False`；`archive_source_code=False`。

- 保留训练日志、Iter/Max Iter、LR、总 Loss、所有实际 Loss、已有各分支 acc_seg、Time、Data Time，以及原有验证指标与 checkpoint。
- 默认不保存 ClassMix/HRDA、Teacher、phy、struct、proto、joint 调试图片，不收集相应 HRDA debug_output。
- 跳过关闭的调试函数中的额外 Student 前向、绘图、转 CPU、图片写盘，以及纯诊断的统计归约。
- 默认不生成完整源码压缩包；保留有效配置、`source_原配置名.py` 输入副本、Git SHA、环境与初始化信息。
- `tools/test.py --eval mIoU` 与 `test.sh RUN_DIR` 默认只评价；`--show-dir` 才保存正式彩色预测；`--save-confidence-map` 另行开启置信图。

三个迁移场景的 24 个正式消融配置统一设置输出开关。另检查了 `configs/hrda/pssal/` 下两个监督场景的 16 个配置，并精简其公共日志配置，避免监督实验仍保留旧的诊断字段白名单。其他旧 DACS 配置也受模型端默认 minimal 总开关约束；未逐一执行所有历史配置的 MMCV 解析。

## 2. 修改文件与位置

下表路径均相对于 `D:\PSSAL-DA`，位置用函数或配置项定位。

| 文件 | 修改位置及原因 |
|---|---|
| `tools/train.py` | `main()`：向 PrecisionLogger 传递输出模式/过程统计开关；复制输入配置；源码归档改为显式开启；记录 Git SHA。 |
| `tools/test.py` | `parse_args()/main()`：新增 `--save-confidence-map`，要求同时提供 `--show-dir`；旧参数保留。 |
| `test.sh` | 默认删除自动 `--show-dir`；透传额外测试参数；保留旧 JSON 配置查找，补充有效 Python 配置查找并排除 `source_*` 副本；引用路径，文件使用 LF。 |
| `mmseg/apis/test.py` | `single_gpu_test()`：置信图第二次推理改为显式开启；仅保存分割图时跳过输入反归一化/缩放；原预测返回值和调色板保存函数保留。 |
| `mmseg/models/uda/dacs.py` | `__init__/update_debug_state/forward_train`：图片总开关、debug_output 收集和保存条件；phy/struct/proto/joint/boundary 诊断统计按需计算；过程统计单独开启；`train_step` 补全总 Loss 的报告值。 |
| `mmseg/models/uda/proto_gate.py` | 更新日志及 `_weight_debug()`：可跳过逐类统计、计数报告、margin 调试插值；仍保留训练会使用的 `proto_weight`，所有原型更新和损失计算保留。独立调用的新增参数默认保留旧诊断能力。 |
| `mmseg/models/uda/struct_prior_gate.py` | `apply()`：关闭诊断时在训练权重/响应和状态更新完成后提前返回；跳过调试矩阵、熵、逐类统计；保留 `R_struct/R_continuity/R_adj/R_sem`。 |
| `mmseg/models/uda/pssal_supervised.py` | 构造、结构/生理/边界日志处：监督分支也默认跳过纯诊断统计。 |
| `mmseg/utils/precision_logger.py` | 构造、`_keep_train_key/_dump_log/_log_info/log`：训练 JSON 与终端同时过滤；所有 loss 和已有 acc_seg 字段保留；minimal 终端按总 Loss/Acc、source、mix、target、aux 分组换行；验证不经过训练过滤；GPU memory 采集显式开启。保留旧参数位置与此前验证模式判断修复。 |
| `configs/hrda/pssalda_26rgb2_30rgb_ablation_utils.py` | 公共输出开关、`make_uda_override/make_log_config`：统一 minimal，默认关闭图片和逐类统计，日志间隔仍为 50。 |
| `configs/hrda/pssalda_26rgb2_30rgb_ablation_common.py` | `archive_source_code/uda`：统一默认输出值，不改变模块 enable 或训练设置。 |
| `configs/hrda/pssalda_26irrg2_30rgb_ablation_common.py` | 同上。 |
| `configs/hrda/pssalda_30irrg2_26irrg_ablation_common.py` | 同上。 |
| `configs/hrda/pssal/pssal_supervised_ablation_utils.py` | 监督实验公共 helper：minimal、逐类统计默认关闭、移除冗余日志白名单；所有实际 Loss 由 logger 保留。 |
| `tests/test_minimal_output.py`（新增） | 输出行为 Mock、入口兼容性、配置保护、CPU 新旧数值/梯度/状态/更新对比。 |
| `docs/minimal_output_report_zh.md`（新增） | 本报告与远端执行指令。 |

已审查但未改动：`phy_reliability.py`、HRDA segmentor/head、`collect_env.py`、边界 Loss、数据集及标签映射、三个离线论文分析脚本、`tools/export_pbcr_edge_debug.py`。独立导出脚本未加入正常训练/验证/推理调用链。

## 3. 输出瘦身对照

| 输出类型 | 原产生位置 | 原触发条件 | 新处理方式 | 是否影响训练 |
|---|---|---|---|---|
| ClassMix 与 HRDA 中间图 | `work_dir/class_mix_debug/` | `local_iter % debug_img_interval == 0`，包括初始 iter 0 | diagnostic 且 `save_class_mix_debug=True` 才开启；默认不绘图、不收集 decoder debug | ClassMix 与 HRDA 训练计算保留 |
| Teacher 标签/置信图 | `work_dir/teacher_pseudo_debug/teacher_raw_pseudo_label/`、`teacher_target_confidence_map/` | Teacher 保存开关及间隔 | 默认关闭，保存函数立即返回 | EMA、伪标签、训练置信度/权重计算保留 |
| 生理调试图 | `work_dir/phy_debug/` | phy 保存开关及间隔 | 默认关闭；跳过其 Student 前向与保存 | 生理响应、门控、实际 Loss 保留；旧调试前向副作用见后文 |
| 结构图与邻接矩阵 | `work_dir/struct_gate_debug/` | struct 保存开关及间隔 | 默认关闭；跳过其 Student 前向、调试矩阵与绘图 | 训练响应、源域先验更新、特征交互保留；副作用见后文 |
| 原型权重、margin、最终权重图 | `work_dir/proto_gate_debug/` | proto 保存开关及间隔 | 默认关闭；跳过仅用于图片的 margin 插值 | Prototype EMA、相似度、监督权重及 Loss 保留 |
| 联合门控调试图 | `work_dir/phy_struct_gate_debug/` | joint 启用且保存开关及间隔 | 默认关闭；跳过额外 Student 前向 | 启用联合门控时仍正常融合；副作用见后文 |
| 逐类/门控/边界冗余标量 | 模型 `log_vars`，`.log/.log.json` | 每次模型训练及日志 interval | 模型端跳过纯诊断归约；终端与 JSON 均过滤 | 不删除用于阈值、原型更新或 Loss 的统计 |
| Figure 12 原始过程统计 | `log_vars`，`.log.json` | 原先部分字段自动记录 | `paper_process_logging=True` 才记录四类原始量，不自动生成 CSV/曲线 | 开关只控制记录 |
| 完整源码归档 | `work_dir/code.tar.gz` | 每次启动训练 | `archive_source_code=True` 才归档 | 不影响模型 |
| 正式预测彩色图 | `--show-dir/target_prediction/原文件名` | 原 `test.sh` 自动保存；测试 API 同时计算置信图 | 默认只评价；明确 `--show-dir` 时保存五类调色板结果 | 预测及评价协议保持 |
| 推理置信热图 | `--show-dir/target_confidence_map/原文件名` | 原 `--show-dir` 附带第二次概率推理 | 同时指定 `--save-confidence-map` 才进行第二次推理 | 默认少一次用于导出的前向；正式预测不变 |
| LR/HR/Attention | `--hrda-out LR/HR/ATT` | 原显式参数 | 保留显式参数，默认无此输出 | 原显式分析功能保留 |
| Checkpoint、有效配置、环境及初始化日志 | 原 work_dir | 原策略 | 保留；多保存一份小型输入配置副本 | 正式实验仍 interval=1000、max_keep_ckpts=5、save_best=mIoU |

目录名也可由原配置自定义；未改动已有目录设置。

正式彩色图继续遵循 `ori_filename`，该数据集通常为 `.tif`，不会为了统一成 PNG 而改名或改映射。`--out` 保留原始 0..4 类别预测数组；彩色图使用 `GOJI_PALETTE`。GT 的 `reduce_zero_label=True`、ignore=255 和评价协议未改。

## 4. 数值边界与日志语义

### 保留的训练计算

保留 ClassMix、Teacher EMA 前向/更新、原伪标签筛选、MPARC 门控与特征交互、Source/Target Prototype 更新及 EMA、Prototype loss、PBCR 边界响应和 Loss。尤其保留 `use_semantic_proto=True` 时源域 Student 的 `generate_pseudo_label()`：该前向更新结构语义先验，属于训练算法。

没有将所有 `.item()` 或 CPU 操作一律删除。例如原型更新中的最小像素数判断、原伪标签权重比例等算法控制继续使用原实现。

### 总 Loss 仅修改报告

原 DACS 多次 `_parse_losses()` 会让后面的 `loss` 覆盖前面的总值。现在在 `optimizer.step()` 之后，将已经取得的各项 Loss 标量汇总为报告用的 `loss`。`loss_phy_src` 历史字段记录未乘 lambda 的 BCE，因此汇总时乘 `lambda_phy_src`；该字段本身保持原语义。其他实际记录项按原加权值汇总。backward 次序、梯度、Loss 公式和优化器步骤均未修改。新旧 `loss` 总字段不宜直接当作相同口径的历史曲线比较。

### 关闭旧调试前向的副作用

旧 phy/struct/joint 图片函数虽使用 `torch.no_grad()`，Student 仍可能处于训练模式；BN running state 可能更新，Dropout/DropPath 可能消耗随机数。CPU stand-in 测试验证了这种行为：开启三类调试前向时 BN 计数增加且 RNG 前进，关闭时两者不变。

因此，不能承诺新 minimal 运行与历史“调试开启”的训练逐位一致。本次依照要求移除诊断副作用，不添加 RNG 保存恢复、不改 BN 模式或算法来掩盖差异。组件级数学输出、状态与梯度对比通过，不等于完整 4090 训练已经验收。

### 四类过程量

明确开启 `uda.paper_process_logging=True` 后，训练日志保留以下原始量，仍按日志 interval 汇总：

| 字段 | 精确定义 |
|---|---|
| `teacher_target_confidence_mean` | Teacher 每像素最大 softmax 概率的全图均值，在 MPARC/Prototype 加权前计算。 |
| `mparc_pseudo_weight_mean` | 生理/结构门控及可选 joint 融合之后、Prototype 加权之前的 pseudo weight 全图均值。不是最终权重。 |
| `proto_tgt_update_ratio` | Target Prototype 本轮实际用于更新的像素数 / 特征图全部像素数；保留原置信度、权重、ignore 和 min_pixels 筛选。 |
| `struct_edge_w_mix_mean` | ClassMix 后实际混合边界监督权重的均值，包含原 warmup 与源/目标 Loss 权重。 |

模块未启用或相应阶段未执行时，相关字段可能缺失，不伪造为 0。论文原绘图脚本读取已有 curated CSV，其列如 `boundary_weight_x100` 包含后处理含义；本次不重新解释其历史聚合公式，不把最终 Prototype 权重冒充 MPARC，也不把新 JSON 当作历史 CSV 的直接替代。已有 CSV 和三个离线分析脚本保持原样。

## 5. 本地验证结果与限制

本地不安装项目环境。使用已有 `D:\Anaconda3\envs\aglldiff\python.exe` 的 PyTorch 2.3.0 CPU 张量执行轻量测试；本地缺少 MMCV。

| 检查 | 结果 |
|---|---|
| `python -m compileall -q tools mmseg configs/hrda` | 通过。 |
| `git diff --check` | 通过；Git 提示部分文件未来 checkout 可能转换 CRLF，非 whitespace 错误。本次 `test.sh` 字节确认是 LF。 |
| `python -m unittest discover -s tests -p test_minimal_output.py -v`，上述 CPU 环境 | 16 项中 15 项通过，1 项跳过。 |
| 40 个正式配置的 Python 执行/继承静态对比 | 去除明确列出的输出项后，模型、损失、路径、数据、标签、模块 enable、优化器、调度、批量、迭代、评价和 checkpoint 参数与修改前一致。此检查不是 MMCV 的替代实现验收。 |
| DACS 构造及输出 Mock | 24 个迁移配置的真实 DACS 构造代码配合小型模型 stand-in；关闭模块不被开启，默认图片开关有效；结构开启/joint 关闭场景不会访问缺失调试属性。 |
| 输出行为 | 关闭的保存 helper 不访问文件/Student；显式开启后到达图片写出调用。默认评价无图片写出；单独 `--show-dir` 无第二次推理；明确置信图参数后才有第二次推理；正式调色板逐像素一致。 |
| CPU 新旧数值对比 | 生理、joint、PBCR 权重及边界先验、EMA 参数一致；Prototype 状态/权重/Loss/梯度/AdamW 更新一致；结构响应/状态/梯度一致；还原诊断模式的原型 debug 字典一致。对已比较的张量使用 rtol=0、atol=0。 |
| 日志与入口 | 训练 JSON/终端过滤、所有 Loss 保留、验证精度/字段保留、验证缺少 data_time 回归、旧入口参数解析通过。 |
| 完整 `Config.fromfile` | 本地因 MMCV 缺失而跳过，已在测试中保留；请在现有远端 hrda 环境执行下文命令。 |
| Bash 语法/运行 | 尝试本机 Git Bash；进程启动失败（Windows 状态 0xC0000142），未获得 shell 运行验收；下文提供远端 `bash -n`。 |
| GPU/完整网络 | 未执行 4090 五步或 20K 训练、完整推理、真实数据集评价；不报告加速百分比或显存降幅。 |
| 保护现有修改 | Git index 与起始快照逐项一致，HEAD 未变；404 个既有 Python 文件中，本任务清单以外文件哈希未变。 |

修改前快照和核验中间结果保存在被忽略的 `outputs/minimal_output_*.json`，未纳入源码；远端若没有这些本地快照，新旧基线对比测试会明确跳过，不会宣称重新做过新旧对比。输出行为及真正的 MMCV 配置测试仍可执行。

Checkpoint 策略未改。MiT-B5 + EMA/优化器及多个中间恢复 checkpoint 仍可能占较大空间，本次不调整周期或删除历史文件。另有既存限制：DACS `local_iter` 是普通 Python 计数，恢复 runner 不代表其所有内部 warmup/EMA 进度都能完整恢复；本任务未修改恢复机制，下面均使用新目录启动。

Git SHA 用于标识提交基线，不包含未提交差异；本次未提交，所以不要只凭 SHA 判断远端已同步全部修改。请同步本任务源码及测试/报告。

## 6. 同步后远端命令

以下命令在同一个 GPU 终端中依次执行；只使用已有 `hrda` 环境。项目、数据、预训练路径保持原值。`mktemp -d` 为本轮创建唯一输出根目录。

### 环境进入、检查与新目录

```bash
conda activate hrda
cd /root/HRDA-master

python -m compileall -q tools mmseg configs/hrda
bash -n test.sh
python -m unittest discover -s tests -p test_minimal_output.py -v

CFG=/root/HRDA-master/configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py
RUN_ROOT=$(mktemp -d /root/shared-nvme/work_dirs/minimal_output_$(date +%Y%m%d_%H%M%S)_XXXXXX)
printf '本轮输出目录: %s\n' "$RUN_ROOT"
```

该配置保留当前 `_safe` 实验设置，包括 Prototype 的 8000/14000 阶段；没有切回旧的强辅助权重版本。预训练仍为 `/root/shared-nvme/pretrained/mit_b5.pth`，源域 `/root/shared-nvme/mydataset/26`，目标域 `/root/shared-nvme/mydataset/30`。

### 1）5 Iter 冒烟训练（含第 5 步验证与保存）

```bash
python tools/train.py "$CFG" --work-dir "$RUN_ROOT/smoke5" \
  --options runner.max_iters=5 log_config.interval=1 \
  checkpoint_config.interval=5 evaluation.interval=5 data.workers_per_gpu=0
```

这里只为冒烟测试临时缩短训练、验证/保存周期与 worker；不能用它评价正式精度或耗时。检查存在日志、有效配置、`source_*` 配置、`iter_5.pth`/latest 及验证 best；检查无自动调试目录或 code.tar.gz：

```bash
find "$RUN_ROOT/smoke5" -type d \( -name class_mix_debug -o -name teacher_pseudo_debug -o -name phy_debug -o -name struct_gate_debug -o -name proto_gate_debug -o -name phy_struct_gate_debug \) -print
find "$RUN_ROOT/smoke5" -name code.tar.gz -print
```

两条 find 默认应无输出。5 步不能覆盖 8000/14000 后的所有真实训练分支；本地组件测试覆盖部分后期权重计算，正式训练仍需观察。

### 2）正式 20K 训练

```bash
FORMAL_DIR="$RUN_ROOT/train20k"
python tools/train.py "$CFG" --work-dir "$FORMAL_DIR"
```

保持配置原有 20000 iter、batch=2、lr=6e-5、warmup/模块预热、滑窗参数、验证/保存周期。无需加 `--resume-from` 或 `--load-from`。

### 3）独立验证（val 集）

正式训练完成后使用它保存的有效配置和 checkpoint：

```bash
EVAL_CFG="$FORMAL_DIR/$(basename "$CFG")"
CKPT="$FORMAL_DIR/latest.pth"
python tools/test.py "$EVAL_CFG" "$CKPT" --eval mIoU \
  --options data.test.img_dir=rgb_dir/val data.test.ann_dir=ann_dir/val
```

该项目 `data.test` 本来指向 test 集，因此独立验证显式替换这两个目录以对应训练时 val 集。不要在验证命令上加 `--test-set`。如仅想先验证冒烟 checkpoint，将 `EVAL_CFG` 改为 `$RUN_ROOT/smoke5/$(basename "$CFG")`，`CKPT` 改为 `$RUN_ROOT/smoke5/iter_5.pth`。

### 4）独立正式推理及测试集定量评价（不保存图片）

```bash
mkdir -p "$RUN_ROOT/test"
python tools/test.py "$EVAL_CFG" "$CKPT" --test-set \
  --eval mIoU --out "$RUN_ROOT/test/predictions.pkl"
```

`--out` 是这条命令明确要求保存的原始预测数组，供离线分析；如果只要指标，去掉 `--out` 即可。原有 mF1/aAcc、逐类 IoU 及支持的 Veg/NonVeg 指标仍由原评价实现输出。

### 5）需要论文定性图时显式保存

```bash
python tools/test.py "$EVAL_CFG" "$CKPT" --test-set \
  --show-dir "$RUN_ROOT/qualitative" --opacity 1
```

正式五类结果保存至 `$RUN_ROOT/qualitative/target_prediction/`，不附带输入图、置信图、LR/HR/attention 图。

需要置信图时单独执行：

```bash
python tools/test.py "$EVAL_CFG" "$CKPT" --test-set \
  --show-dir "$RUN_ROOT/qualitative_with_confidence" --opacity 1 \
  --save-confidence-map
```

该功能会额外执行概率推理。HRDA 的 LR/HR/ATT 分析仍通过显式 `--hrda-out LR`、`HR` 或 `ATT` 调用，不自动开启。

`test.sh` 也可继续使用：

```bash
bash test.sh "$FORMAL_DIR"
bash test.sh "$FORMAL_DIR" --show-dir "$RUN_ROOT/qualitative_from_sh" --opacity 1
```

这两条分别是评价，以及评价并保存正式预测图；配置选择有歧义时脚本会提示改用显式 `tools/test.py CONFIG CHECKPOINT`。

## 7. 重新开启诊断与过程量

只记录四类过程量、仍关闭图片：

```bash
python tools/train.py "$CFG" --work-dir "$RUN_ROOT/process20k" \
  --options uda.paper_process_logging=True
```

模型与 logger 的对应开关由训练入口同步。此选项不生成论文 CSV 和曲线。

只需要常规诊断标量时，使用 `--options uda.output_mode=diagnostic`。逐类原型统计还需 `uda.proto_cfg.log_class_stats=True`。GPU memory 统计通过对应 PrecisionLoggerHook 的 `log_memory=True` 单独启用，默认不采集。

显式开启图片的五步示例（仅对配置中原本启用的模块起效，不会打开任何训练模块）：

```bash
python tools/train.py "$CFG" --work-dir "$RUN_ROOT/debug5" \
  --options runner.max_iters=5 log_config.interval=1 \
  checkpoint_config.interval=5 evaluation.interval=5 data.workers_per_gpu=0 \
  uda.output_mode=diagnostic uda.save_class_mix_debug=True \
  uda.save_teacher_pseudo_debug=True \
  uda.phy_cfg.save_phy_debug=True uda.struct_cfg.save_struct_debug=True \
  uda.proto_cfg.save_proto_debug=True uda.proto_cfg.log_class_stats=True
```

不把间隔设为 0；保留原间隔即可在 local_iter=0 导出。联合门控本来启用的配置可额外设置 `uda.joint_gate_cfg.save_joint_debug=True`；不要为了图片而改变其 `enable`。需要完整源码归档时单独加 `archive_source_code=True`。

## 8. 上游核查依据

[HRDA DACS](https://github.com/lhoyer/HRDA/blob/master/mmseg/models/uda/dacs.py)、[HRDA segmentor](https://github.com/lhoyer/HRDA/blob/master/mmseg/models/segmentors/hrda_encoder_decoder.py)、[HRDA 训练入口](https://github.com/lhoyer/HRDA/blob/master/tools/train.py)。

ClassMix/HRDA 调试图与启动时源码归档来自上游框架；Teacher 标签/置信图、phy/struct/proto/joint 调试和大量逐类标量属于本仓库新增功能。修改针对这些输出入口及其专用计算，没有改写上游网络结构。

## 9. 日志可读性补充调整

根据实际运行反馈，恢复上一版被过滤掉的已有 `acc_seg`，同时保留 minimal 输出模式。只需重新同步 `mmseg/utils/precision_logger.py`；训练入口与配置不需要改参数。

- 首行：Iter、LR、ETA、Time、Data Time、总 Loss，以及已有的各分支 Acc。
- 后续缩进行：`source` 源域分支 Loss、`mix` 混合分支 Loss、适用时的 `target` 分支 Loss、`aux` 其他实际 Loss。显式过程统计另列 `stats`。
- Acc 的原始值已经是 0..100，终端仅加百分号并保留两位小数，不再乘 100；JSON 保留原始字段名及数值精度。
- 所有实际 Loss 继续完整输出，没有因默认值为零而隐藏字段。diagnostic 模式保留原有单行明细形式；验证输出不变。

已有准确率的解释：

| 字段 | 含义 |
|---|---|
| `decode.acc_seg` | 当前源域 batch 上融合预测相对源域 GT 的像素准确率。 |
| `mix.decode.acc_seg` | 当前混合 batch 上融合预测相对混合监督标签的像素准确率；监督标签含源域 GT 和目标域伪标签。 |
| `decode.hr.acc_seg` / `mix.decode.hr.acc_seg` | 各自 HR 分支的像素准确率，按原实现使用对应 crop 或图像标签。 |
| `tgt.decode.acc_seg` 等 | 关闭混合等既有训练模式下实际返回的其他 acc_seg，若存在则保留。 |

这些是训练阶段的 batch 拟合指标，不等同于目标域验证集准确率。尤其 Mix Acc 高，不代表目标域真实分割精度同样高。沿用旧版 accuracy 的原始分母口径（`target.numel()`，未重新排除 ignore 像素）；不修改指标算法。判断实验效果仍以每 1000 步验证阶段的逐类 IoU、mIoU、aAcc/mF1 等为准。

格式示例（`xx.xx` 仅作占位，非实测准确率）：

```text
Iter [200/20000] lr: ..., eta: ..., time: ..., data_time: ..., loss: 0.6672, decode.acc_seg: xx.xx%, decode.hr.acc_seg: xx.xx%, mix.decode.acc_seg: xx.xx%, mix.decode.hr.acc_seg: xx.xx%
    source: decode.loss_seg: 0.3446, decode.loss_struct_edge: 0.0000, decode.hr.loss_seg: 0.0382, decode.hr.loss_struct_edge: 0.0000
    mix: mix.decode.loss_seg: 0.2119, mix.decode.loss_struct_edge: 0.0000, mix.decode.hr.loss_seg: 0.0236, mix.decode.hr.loss_struct_edge: 0.0000
    aux: loss_phy_src: 0.6118, loss_proto_tgt: 0.0000, loss_phy_tgt: 0.0000
```

已运行的 Python 训练进程不会热加载新版 logger；同步后新启动的进程才使用新格式。无需为日志显示立即中断当前实验，当前进程的原定验证指标仍会正常输出。
