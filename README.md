# 高效混合CNN-Transformer模型有序压缩研究项目
##  项目概述
本项目旨在探索和实现一种结合了卷积神经网络（CNN）和Transformer的高效混合模型，用于图像识别任务。项目的核心研究内容包括：

1. 设计可配置的轻量级CNN骨干网络和混合CNN-Transformer架构，支持串行和并行等多种融合策略。
2. 系统性地应用“有序模型压缩”技术（如剪枝、量化）来优化模型的效率（大小、速度）和性能（准确率）。
3. 提供一个灵活的实验管理框架，方便定义、运行和比较不同的模型架构、训练策略及压缩流程。
本项目代码使用 Python 和 PyTorch 构建。
##  主要特性
- **混合模型架构**: 实现了一个 `HybridCNNTransformer` 模型，包含：
    - 可配置的 `LightweightCNN` 骨干，允许自定义卷积块（通道数、池化等）。
    - 标准的 Transformer Encoder 模块。
    - 支持“串行”（CNN特征展平后送入Transformer）和“并行”（简化版，CNN全局特征与MLP路径结合）融合策略。
    - 可学习的或固定的位置编码 (`PositionalEncoding`)。
  
- **有序模型压缩**:
    - **剪枝**: 实现了基于Ln范数 (n=1，即L1范数) 的结构化通道/特征剪枝 (`apply_channel_pruning`)，作用于Conv2d和Linear层。**注意**：当前实现是将权重置零，本身不改变层定义的输出维度或FLOPs，如需减少FLOPs需要进一步的模型结构重建。
    - **量化**: 支持训练后静态量化 (PTQ) 和量化感知训练 (QAT)，使用 PyTorch 的原生量化工具。
    - 模块融合辅助函数 (`_fuse_modules_for_quantization`)。
  
- **灵活的实验管理**:
    - 通过 `config.py` 文件集中管理所有配置（全局设置、数据集、模型、训练、压缩参数）。
    - 支持定义多个实验分支 (`EXPERIMENT_CONFIGS`)，每个分支可以有独立的参数配置。
    - 自动处理基线模型的训练、保存和加载，以便在统一的预训练模型上进行压缩实验。

- **详细的训练与评估**:
    - 包含标准的训练 (`train_one_epoch`) 和评估 (`evaluate`) 循环。
    - 支持多种优化器和学习率调度器。
    - 自动记录每个实验阶段的性能指标（准确率、损失、模型大小、FLOPs估算、参数量、推理时间）到JSON文件。

- **模块化代码结构**:
    - `config.py`: 所有配置。
    - `dataset.py`: 数据加载和预处理。
    - `model.py`: 模型定义。
    - `utils.py`: 辅助函数（统计、剪枝、量化）。
    - `train_eval.py`: 训练和评估逻辑。
    - `main.py`: 实验流程主控制脚本。

##  项目结构
```
your_project_directory/
├── config.py         # 存储配置、超参数、路径
├── dataset.py        # 处理数据集加载和预处理
├── model.py          # 定义混合CNN-Transformer模型架构
├── utils.py          # 包含辅助函数 (统计, 剪枝, 量化助手)
├── train_eval.py     # 包含训练和评估循环
├── main.py           # 运行实验的主脚本
├── data/             # (自动创建) 存放下载的数据集
├── saved_models/     # (自动创建) 存放训练好的模型检查点
└── results/          # (自动创建) 存放实验结果日志 (JSON文件)
```

## 🚀 安装与设置

### 1. 环境要求
* Python 3.8 或更高版本
* PyTorch (建议 1.9+ 或更高，因为使用了较新的量化和Transformer API；您当前环境为 2.4.1)
* Torchvision (与 PyTorch 版本匹配)
* Tqdm (用于进度条)
* Thop (可选, 用于FLOPs计算)

### 2. 安装依赖
建议在虚拟环境中安装：
```bash
# 创建并激活虚拟环境 (例如使用 conda)
# conda create -n cnn_transformer_env python=3.9
# conda activate cnn_transformer_env

# 安装 PyTorch 和 Torchvision (请根据您的系统和CUDA版本从 PyTorch 官网获取命令)
# 例如 (CUDA 11.8):
# pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
# 或 CPU 版本:
# pip install torch torchvision torchaudio

# 安装其他库
pip install tqdm

# 安装 thop (可选)
pip install thop
```

### 3. CUDA 设置 (如果使用GPU)
确保您的 NVIDIA 显卡驱动和 CUDA Toolkit 版本与您安装的 PyTorch 版本兼容。代码会自动检测并使用可用的 CUDA 设备。

##  配置 (`config.py`)

`config.py` 文件是项目的核心配置文件。在运行任何实验之前，请仔细检查并根据您的需求进行调整：

* **`DEVICE`**: 自动检测CUDA。如果想强制使用CPU，可以改为 `torch.device("cpu")`。
* **`DATA_DIR`, `MODEL_SAVE_DIR`, `RESULTS_DIR`**: 数据集、模型和结果的存储路径。这些目录如果不存在会被自动创建。
* **`DEFAULT_DATASET_CONFIG`**: 默认的数据集设置（名称、图像大小、num_workers）。
* **`DEFAULT_MODEL_PARAMS`**: 默认的模型架构参数（CNN块配置、Transformer层数/头数等）。
* **`DEFAULT_TRAINING_PARAMS`**: 默认的训练参数（批大小、学习率、优化器类型、调度器类型、训练轮数等）。
* **`DEFAULT_COMPRESSION_PARAMS`**: 默认的压缩参数（剪枝比例、PTQ校准批次数、量化后端）。
* **`EXPERIMENT_CONFIGS`**: **这是最重要的部分**。它是一个列表，每个元素（字典）定义了一个完整的实验流程。
    * `name`: 实验的唯一名称，用于保存模型和结果。
    * `is_fp32_baseline_generation: True`: 标记此实验用于生成FP32基线模型，其权重会被后续实验加载。
    * `load_fp32_baseline_from_experiment: 'BaselineExperimentName'`: 指定此压缩实验依赖哪个已生成的FP32基线模型。
    * `dataset_args`, `model_args`, `training_args`, `compression_args`: 分别用于覆盖对应类别的默认参数。
    * **重要**: 对于依赖预训练基线的压缩实验，请确保其 `training_args` 中的 `'initial_train_epochs'` 设置为 `0`，以避免从头重新训练。

## 运行实验 (`main.py`)

1.  **检查配置**: 再次确认 `config.py` 中的设置符合您的预期，特别是 `EXPERIMENT_CONFIGS` 列表。您可以注释掉不想立即运行的实验配置。
2.  **打开终端**: 导航到项目根目录。
3.  **执行脚本**:
    ```bash
    python main.py
    ```

脚本将按顺序执行 `config.py` 中 `EXPERIMENT_CONFIGS` 列表定义的每个实验：
* **基线模型准备**: 脚本会首先识别并运行所有标记为 `is_fp32_baseline_generation: True` 的实验（或名称包含 "Baseline_FP32"）。如果对应的模型文件已存在，它会询问您是使用现有模型还是重新训练。训练好的基线模型状态会被缓存，供后续依赖它们的实验使用。
* **压缩实验**: 对于其他实验，如果配置了 `load_fp32_baseline_from_experiment`，脚本会尝试加载相应的基线模型权重，然后按顺序执行配置的剪枝、微调和量化步骤。

##  输出与产物

* **控制台**:
    * 显示全局配置信息（设备、路径）。
    * 每个实验开始时打印其名称和关键配置。
    * 训练和评估过程会显示 `tqdm` 进度条，包含损失、准确率、当前学习率等信息。
    * 每个阶段（如初始训练、剪枝后、微调后、量化后）的模型统计数据和评估结果会被打印。
* **`data/` 目录**:
    * 如果本地没有，会自动下载并存储 CIFAR10/CIFAR100 等数据集。
* **`saved_models/` 目录**:
    * 存放训练好的模型权重 (`.pth` 文件)。
    * 文件名会包含实验名称和阶段信息（例如, `Baseline_FP32_Serial_CIFAR10_fp32_baseline.pth`, `Serial_CIFAR10_P40_F_QPTQ_pruned_40_finetuned.pth`, `Serial_CIFAR10_P40_F_QPTQ_p40_f_quantized_ptq_int8.pth`）。
* **`results/` 目录**:
    * 为每个实验生成一个JSON日志文件（例如, `Baseline_FP32_Serial_CIFAR10_results.json`），包含该实验的配置摘要和所有记录的性能指标。
    * 所有实验完成后，会生成一个 `_all_experiments_summary.json` 文件，汇总所有实验的结果日志，方便进行横向比较。

##  代码模块概览

* **`config.py`**: 实验的“大脑”，定义所有可调参数和实验流程。
* **`dataset.py`**: `get_dataloaders` 函数负责加载和预处理数据集，支持CIFAR10/100，并可配置数据增强策略。
* **`model.py`**:
    * `LightweightCNN`: 可配置的CNN骨干网络。
    * `PositionalEncoding`: 可学习或固定的位置编码。
    * `HybridCNNTransformer`: 核心的混合模型，接收 `model_args` 配置其结构、融合策略等。
    * 包含 `_fuse_modules_for_quantization` 辅助方法以支持量化前的模块融合。
* **`utils.py`**:
    * `get_model_stats`: 计算模型大小、FLOPs（如果thop可用）、参数量。
    * `apply_channel_pruning`: 实现基于Ln范数(n=1)的结构化剪枝（权重置零）。
    * `prepare_model_for_quantization`, `calibrate_model_ptq`, `convert_to_quantized_model`: PyTorch量化流程的辅助函数。
* **`train_eval.py`**:
    * `train_one_epoch`: 单个epoch的训练逻辑。
    * `evaluate`: 模型评估逻辑，计算损失、准确率和平均推理时间。
* **`main.py`**:
    * `get_optimizer`, `get_scheduler`: 根据配置创建优化器和学习率调度器。
    * `run_experiment`: 执行单个实验流程的核心函数。
    * `main`: 主函数，负责管理基线模型的准备和所有实验的顺序执行。

##  自定义与扩展

* **添加新数据集**: 修改 `dataset.py` 中的 `DATASET_REGISTRY` 字典，并可能需要为新数据集实现自定义的 `Dataset` 类（如果不是Torchvision原生支持的）。
* **修改模型架构**:
    * 调整 `config.py` 中 `DEFAULT_MODEL_PARAMS` 或特定实验的 `model_args` 来改变 `LightweightCNN` 的块配置、Transformer的参数等。
    * 直接修改 `model.py` 中的类定义以实现更复杂的CNN骨干或新的融合机制。
* **定义新实验**: 在 `config.py` 的 `EXPERIMENT_CONFIGS` 列表中添加新的字典条目，详细定义新实验的参数和依赖关系。例如，尝试不同的剪枝比例、新的压缩顺序（如先量化再剪枝）、知识蒸馏等。
* **高级剪枝**: `utils.py` 中的 `apply_channel_pruning` 目前是将权重置零。要实现真正的FLOPs减少，您需要研究并实现能够修改网络结构（移除层/通道并调整后续层输入）的剪枝技术，或者使用支持此类操作的第三方库。


