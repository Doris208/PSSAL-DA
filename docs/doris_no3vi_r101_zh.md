# 三组 ResNet-101 无 3VI 输入对照

以当前仓库为准，保留三个方法各自的模型与 UDA 实现，不把 HRDA 配置用于替代其他方法。HN 对应目录 30，ZN 对应目录 26。

| 表 | 方向 | 新配置 |
|---|---|---|
| I | HN-IRRG → ZN-IRRG | `configs/hrda/baseline_resnet/doris_goji30IRRG2goji26IRRG_hrda_r101_no3vi.py` |
| II | ZN-IRRG → HN-RGB | `configs/memoryadaptnet/doris_goji26IRRG2goji30RGB_memoryadaptnet_r101_no3vi.py` |
| III | ZN-RGB → HN-RGB | `configs/daformer/doris_goji26RGB2goji30RGB_daformer_r101_no3vi.py` |

每个配置直接继承同目录的原 R101 六通道配置，仅用现有三通道 dataset 配置完整替换 `data`，将 backbone 输入改为 3，并设置独立名称和目录。原配置、旧批次脚本均不修改。

## 输入、初始化与训练协议

训练、验证、测试均不使用 `LoadGojiIndices`，不读取 NDVI、NDRE、CIRE，不收集 `vi_img`；完整替换 `data` 会同时移除继承的 VI 开关。HRDA/DAFormer 的目标四边 30 像素伪标签 mask 保留，MemoryAdaptNet 原配置没有该 mask，仍然不添加。

归一化复用 `configs/_base_/datasets/` 下对应无 `_3vi` 的三通道配置。下表顺序指 Normalize 后的实际输入，加载器仍为 OpenCV；IRRG 使用 `channel_order=[2,1,0]`，RGB 使用等价的 `to_rgb=True`。两者与各自六通道配置前三通道的变换一致。

| 影像 | 输入顺序 | mean | std |
|---|---|---|---|
| HN-IRRG | NIR, R, G | 140.700, 144.074, 151.917 | 65.998, 63.564, 53.397 |
| ZN-IRRG | NIR, R, G | 111.366, 122.724, 135.261 | 61.302, 62.369, 55.989 |
| ZN-RGB | R, G, B | 122.724, 111.366, 101.588 | 62.369, 61.302, 59.396 |
| HN-RGB | R, G, B | 144.074, 140.700, 133.597 | 63.564, 65.998, 65.842 |

三个骨干均为 `ResNetV1c(depth=101, in_channels=3)`，保持原 strides/dilations、BN 和解码头。使用匹配其三层 3×3 deep stem 的 `open-mmlab://resnet101_v1c`，MMCV 1.4.0 将其解析为官方 `resnet101_v1c-e67eebb6.pth`。云端首次初始化需要已有缓存或能够下载此权重；本地没有权重文件，未执行实际权重加载。

共同协议：seed=0，单 GPU batch=2（源域和目标域各取对应 batch），workers=4，AdamW，基础 lr=6e-5，weight_decay=0.01，Poly power=1，1500 步线性 warmup，warmup_ratio=1e-6。HRDA/DAFormer 保留 head 的 10 倍学习率和原参数分组；MemoryAdaptNet 保留主优化器与判别器 Adam 的超参数，同时按用户补充要求隔离判别器更新。增强仍是 256×256、resize 比例 0.8–1.2、cat_max_ratio=0.75 的随机裁剪、水平/垂直翻转各 0.5。

- 三组都显式设为 **20,000 步**。仓库旧 R101 配置的有效值为 15,000；用户已确认历史六通道实际实验统一为 20k，本次据此覆盖，不修改旧文件。
- HRDA 保留双尺度、128×128 HR crop、HR loss=0.1 和 DACS；DAFormer 保留单尺度 EncoderDecoder/DAFormerHead 和 DACS，EMA=0.999；MemoryAdaptNet 保留其两阶段解码器、原记忆模块、3000 步启用点、熵阈值 0.3 和对抗损失，未套用 DACS。
- MPARC、PBCR、生理门控、结构门控、额外原型门控与边界损失均未启用。MemoryAdaptNet 自带的 prototype memory 属于原方法，保留。
- 源域/目标域的 train、目标域 val/test 目录不变；`GojiDataset` 保持原标签 1..5 → 0..4、0 → 255 的映射。训练 UDA wrapper 不接收目标域 GT。
- 每 1000 步验证/保存，原滑窗 256×256、stride=128、无测试翻转设置不变；best 仍按五类 mIoU 选取。

## 可比性边界

配置检查已逐项比较当前原 R101 配置的网络、UDA、优化器、LR schedule、增强、split、checkpoint 和评估设置；允许的差异为 VI 输入、V1c 权重入口、用户确认的 20k、MemoryAdaptNet 判别器更新隔离，以及输出名称/输入说明等非训练元数据。

本地仅有 MiT-B5 相关运行配置/日志，未找到论文这三组 R101 六通道实验的运行导出配置，因此历史 batch、命令行覆盖、权重内容等不能独立核实。20k 的依据是用户确认，不代表已读到对应运行记录。

旧配置的权重路径是 `/root/HRDA-master/pretrained/resnet101-5d3b4d8f.pth`，本地没有该文件。该文件名通常对应 torchvision 的 7×7 stem ResNet，而配置使用 deep-stem ResNetV1c；无法排除云端文件已被重命名。新配置使用明确匹配的 V1c 权重入口。若历史实际用了不同的预训练内容，两组之间就不只有输入差异，不能宣称严格公平；应核对历史初始化日志/权重，必要时统一权重重新比较。

原 R101 MemoryAdaptNet 使用主 AdamW 遍历整个 UDA wrapper，同时又有判别器独立 Adam，导致判别器被两个优化器更新。按用户补充要求，新 R101 无 3VI 配置现已与 DORIS MiT-B5 MemoryAdaptNet 对比配置保持一致：主 AdamW 中 `discriminator` 参数组的 `lr_mult=0.0`、`decay_mult=0.0`，判别器仅由独立 Adam（lr=1e-4）实际更新。其他参数组、损失、记忆更新和网络不变，核查工具同时断言新旧 DORIS 对比配置的有效 optimizer 完全一致。原始六通道 R101 文件保留；与该历史结果做纯输入消融时，六通道一侧也应采用同样的隔离设置重跑。MiT-B5 与 R101 的骨干差异仍然存在，不能把二者当作仅相差 3VI 的一对实验。

## 输出与分组指标

根目录为 `/root/shared-nvme/work_dirs/doris_no3vi_r101/`，三个实验分别写入各自配置名去掉 `doris_` 后的独立子目录。每组保留 `.log`、`.log.json`、有效配置、逐类 IoU、mIoU、best checkpoint、`iter_20000.pth` 和 `latest.pth`；沿用最多保留 5 个周期 checkpoint 的设置，不改原输出目录。

本次仅在 `GojiDataset.evaluate` 增加两个统计，不改变原评估结果或 checkpoint 选取：

- `Veg.mIoU` = mean(IoU.goji, IoU.vegetation)，对应类别 0、1。
- `NonVeg.mIoU` = mean(IoU.building, IoU.water, IoU.road)，对应类别 2、3、4。

分组值基于原评估器返回的逐类 IoU（已按百分数保留两位小数）取宏平均，沿用 NaN 类别不参与平均的口径；控制台按百分数展示，日志结果字典为 0..1。该统计对使用 GojiDataset 的六通道运行也会生效，便于按同口径比较。只请求非 IoU 指标时不附加分组 IoU。

## 运行

```bash
conda activate hrda
cd /root/HRDA-master

# 配置解析检查，不构建模型，不加载数据或执行训练。
python tools/check_doris_no3vi_r101_configs.py

# 三条可以分别执行，各自配置已经设置 work_dir。
python tools/train.py configs/hrda/baseline_resnet/doris_goji30IRRG2goji26IRRG_hrda_r101_no3vi.py
python tools/train.py configs/memoryadaptnet/doris_goji26IRRG2goji30RGB_memoryadaptnet_r101_no3vi.py
python tools/train.py configs/daformer/doris_goji26RGB2goji30RGB_daformer_r101_no3vi.py

# 或按顺序运行这三组；与上面三条不要重复执行。
bash train_doris_no3vi_r101_20k.sh
```

本地验收：真实 MMCV 1.4.0 `Config.fromfile(import_custom_modules=False)`，只跳过需 PyTorch 的 runtime custom_imports；核查了有效模型、输入、网络/UDA及优化协议、数据管线和批次脚本命令。新增分组 IoU 两项轻量测试，不导入训练栈。不运行本地/远端正式训练，不自动 commit/push。真实数据读取、V1c 权重初始化及 GPU 训练仍由云端环境验证。
