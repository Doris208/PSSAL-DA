# DORIS 第一阶段新增两组 MiT-B5 比较实验

本次范围按用户确认，只新增两组，不扩展到三个方向各两组：

| 方法 | 方向 | 新配置（相对项目根目录） |
|---|---|---|
| MemoryAdaptNet＋MiT-B5＋3VI | 26IRRG→30RGB | `configs/memoryadaptnet/doris_goji26IRRG2goji30RGB_memoryadaptnet_3vi.py` |
| DAFormer＋MiT-B5＋3VI | 26RGB→30RGB | `configs/daformer/doris_goji26RGB2goji30RGB_daformer_3vi.py` |

新配置与各自被复制/继承的旧配置放在同一目录。旧 R101 配置和原六个 DORIS 配置不修改。两组新配置统一使用 doris_ 前缀及方法名，不在实验文件名和输出目录中额外标注骨干；模型内部均明确配置为 mit_b5。

## 1. 参数与输出位置

两组均为 MiT-B5、六通道、五类别、256×256 裁剪、batch=2、workers=4、seed=0、AdamW、lr=6e-5、Poly、1500 Iter 线性 LR warmup、20K 训练、每 1000 Iter 验证/保存、max_keep_ckpts=5、以 mIoU 保存 best。

预训练统一为 `/root/shared-nvme/pretrained/mit_b5.pth`。新两组不加载 ResNet 权重。输入通道适配复用仓库既有 MiT 预训练加载代码。

输出直接位于：

```text
/root/shared-nvme/work_dirs/doris_stage1/
├── goji26IRRG2goji30RGB_memoryadaptnet_3vi/
└── goji26RGB2goji30RGB_daformer_3vi/
```

没有 batch 时间戳层，子目录不带 doris_ 前缀，与原六组脚本风格一致。重复执行会复用相同目录，不自动删除、移动或恢复实验。

## 2. 为什么需要适配代码

### DAFormer

旧 `goji26RGB2goji30RGB_daformer_3vi.py` 虽继承名称含 mitb5 的模型 base，最终却用 `_delete_=True` 替换成 ResNet-101。本次新文件明确替换为 `mit_b5`、`in_chans=6`，解码器输入通道改为 `[64,128,320,512]`，删除继承的 ResNet 专用参数。

保留原单尺度 `EncoderDecoder + DAFormerHead + DACS`、ClassMix、Teacher EMA=0.999、pseudo_threshold=0.968、目标区域四边 30 像素裁剪、原数据流水线及优化器参数组规则。没有开启 HRDA、多尺度分支、MPARC、PBCR 或 DORIS 融合。

### MemoryAdaptNet

复用仓库已有 MemoryAdaptNet 实现，做 MiT 输入适配：

- MiT 末级通道为 512，Stage-1 head 和 memory bottleneck 的输入由 2048 改为 512。
- 保留原 head 的 channels=256、memory_dim=1024、transform_channels=512、融合输出/Stage-2 输入=2048，以及原记忆注意力与更新公式。
- MiT 末级特征为 stride=32，256×256 输入对应 8×8；现有五层 stride-2 判别器不能接收 8×8。新增 `feature_upsample_scale`，仅新配置设为 4，在 head/bottleneck 前用双线性插值恢复原方法的 stride-8（32×32）网格；训练和预测共用这个路径。
- 原 R101 配置不设置该参数，默认 1，返回原特征，不做插值。
- 记忆模块仍从第 3000 Iter 启用，Target 熵阈值=0.3，lambda_stage1=1、lambda_stage2=1、lambda_adv=0.001，判别器 Adam lr=1e-4；原记忆更新和方法特定的数据策略保留，未套用 DACS/ClassMix 或额外 Teacher。
- 总分割优化器会遍历 UDA wrapper 的全部参数，而判别器已有独立优化器。因此新配置将分割优化器内 discriminator 参数组的 lr_mult/decay_mult 设为 0，判别器只由其独立 Adam 实际更新；不改旧 R101 优化器配置。
- 新配置开启 `log_train_accuracy=True`，输出 `stage1.acc_seg`，记忆阶段再输出 `stage2.acc_seg`。这两项为源域 batch Acc，沿用现有 MMSeg 的百分数和分母口径，不参与反向传播；旧配置默认 False。
- 使用已有 `MemoryAdaptNetOptimizerStateHook` 每 1000 Iter 保存判别器优化器与 local_iter sidecar。恢复时应使用对应的 `iter_N.pth` 和同名 `.memoryadaptnet_optimizers.pth`；不承诺仅有 latest 链接就能恢复 sidecar。

两个方法均保留各自旧配置的数据流水线与评价 split。MemoryAdaptNet 的训练策略与 DAFormer/DACS 不同，不能把它的记忆更新解释为 DORIS PBCR。

## 3. 修改清单

本轮新增：

- 上述两个配置。
- `train_doris_mitb5_stage1_comparisons_20k.sh`：只跑新增两组。
- `tests/test_doris_comparisons.py`：配置、维度、训练/预测及脚本检查。
- 本说明。

本轮更新：

- `mmseg/models/segmentors/memory_adapt_encoder_decoder.py`：追加默认=1 的可选特征插值参数。
- `mmseg/models/uda/memory_adaptnet.py`：追加默认关闭的源域 Acc 统计，不更改 Loss/记忆更新。
- `train_doris_mitb5_stage1_20k.sh`：保留原六条顺序，在最后追加 MemoryAdaptNet 和 DAFormer 两条，现在共八组。
- `tools/check_doris_stage1_configs.py`：保留原 30 配置检查，并增加新两组的方向、MiT 通道、记忆输入、判别器参数组、20K、输出路径等检查；现在检查 32 个配置。
- `docs/doris_stage1_report_zh.md`：补充当前八组及只补两组的入口。

既有所有配置保持字节一致；不修改 DACS/HRDA 或之前 DORIS 的结构与边界代码。本轮不连接远端、不安装依赖、不运行正式训练、不 commit/push。

## 4. 本地验证

- 编译 `tools mmseg configs tests` 和 `git diff --check` 通过。
- 32 配置的 Python 执行/继承静态检查通过；真正 `Config.fromfile` 仍需在远端现有 MMCV 环境执行。
- CPU 测试使用已有 Torch 环境，包含原套件共 44 项，40 项通过、4 项因依赖缺失跳过（三项 MMCV、一项 Matplotlib）。新测试涵盖 512 通道/32×32 网格、真实 Memory head/attention/判别器、记忆模块启用前后训练、反向传播、判别器单独更新、预测输出为 256×256 五类、可选 Acc 不改变 Loss/梯度，以及默认 scale=1 的原路径。
- CPU 训练测试使用 backbone stand-in，并缩小记忆模块隐藏宽度；正式配置仍保持 1024/2048 原宽度。它不替代完整 MiT-B5 在 4090 上的验证，不声称真实训练已通过。
- Bash 文件使用 LF，八组总脚本及两组补跑脚本的命令路径对应。当前 Windows Git Bash 不能启动，真正 shell 检查需在远端执行。

## 5. 同步后先核验

同步本轮所有改动，尤其 MemoryAdaptNet 的两个模型文件，不能只上传配置。

```bash
conda activate hrda
cd /root/HRDA-master
python tools/check_doris_stage1_configs.py
bash -n train_doris_mitb5_stage1_comparisons_20k.sh
```

成功时末尾显示：

```text
PASS: 24 legacy and 8 new effective configs (6 HRDA/Full + 2 comparisons); output directories are distinct.
```

可先执行独立 5 Iter 检查，写入临时新目录。Memory 的临时 memory_start_iter=0 用于尽早覆盖记忆分支，不改变正式配置的 3000：

```bash
CHECK_ROOT=$(mktemp -d /root/shared-nvme/work_dirs/doris_comparisons_check_XXXXXX)

python tools/train.py configs/memoryadaptnet/doris_goji26IRRG2goji30RGB_memoryadaptnet_3vi.py \
  --work-dir "$CHECK_ROOT/memory5" \
  --options runner.max_iters=5 uda.memory_start_iter=0 \
  log_config.interval=1 evaluation.interval=5 checkpoint_config.interval=5 data.workers_per_gpu=0

python tools/train.py configs/daformer/doris_goji26RGB2goji30RGB_daformer_3vi.py \
  --work-dir "$CHECK_ROOT/daformer5" \
  --options runner.max_iters=5 log_config.interval=1 \
  evaluation.interval=5 checkpoint_config.interval=5 data.workers_per_gpu=0
```

Memory 的 sidecar 保存周期仍为 1000，因此 5 Iter 不生成该文件是正常的。

## 6. 只补跑新增两组

原六组已经运行/完成时，使用这个入口，不需要重新跑原六组：

```bash
python tools/check_doris_stage1_configs.py && bash train_doris_mitb5_stage1_comparisons_20k.sh
```

顺序为 MemoryAdaptNet（26IRRG→30RGB），然后 DAFormer（26RGB→30RGB），每组 20K，`set -e` 保持。

如要从头执行全部八组，才使用：

```bash
bash train_doris_mitb5_stage1_20k.sh
```

## 7. 单独训练与评价

```bash
python tools/train.py configs/memoryadaptnet/doris_goji26IRRG2goji30RGB_memoryadaptnet_3vi.py
python tools/train.py configs/daformer/doris_goji26RGB2goji30RGB_daformer_3vi.py
```

它们的默认输出已经与补跑脚本相同。正式测试保持原入口；例如 MemoryAdaptNet：

```bash
MEM_DIR=/root/shared-nvme/work_dirs/doris_stage1/goji26IRRG2goji30RGB_memoryadaptnet_3vi
python tools/test.py "$MEM_DIR/doris_goji26IRRG2goji30RGB_memoryadaptnet_3vi.py" \
  "$MEM_DIR/latest.pth" --eval mIoU
```

DAFormer 将目录和有效配置名换成 `goji26RGB2goji30RGB_daformer_3vi` 对应名称即可。以上默认评价 test split；独立 val 需显式设置 `data.test.img_dir=rgb_dir/val data.test.ann_dir=ann_dir/val`。
