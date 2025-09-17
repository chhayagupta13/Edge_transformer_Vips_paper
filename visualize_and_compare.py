import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 这行代码确保 OpenMP 库的冲突问题能尽早被处理
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'd:\anaconda3\envs\pytorch\library\plugins\designer\pyqt5.dll'
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
import matplotlib as mpl # <-- 新增：导入 matplotlib 模块
import numpy as np
import seaborn as sns
from torchvision.utils import make_grid
from tqdm import tqdm
import argparse
from typing import Dict, Any, Optional, List, Tuple
import dataset
# --- 设置 matplotlib 字体以支持中文 ---
# 尝试使用 'SimHei' (黑体)。如果您的系统没有，可以尝试 'Microsoft YaHei' (微软雅黑) 或其他中文字体。
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 这行代码解决负号 '-' 显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False

# 导入项目文件
import config as cfg
from dataset import get_dataloaders
from model import HybridCNNTransformer
from utils import get_model_stats, prepare_model_for_quantization, convert_to_quantized_model

# ... (后续所有函数和主函数代码保持不变)

# --- 定义图片保存目录 ---
# 这个目录会创建在项目结果文件夹下
PLOT_SAVE_DIR = os.path.join(cfg.RESULTS_DIR, 'plots')
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)


def load_model_for_visualization(exp_name: str, stage: str, num_classes: int, model_args: Dict[str, Any],
                                 image_size: int, device: torch.device) -> nn.Module:
    """
    根据实验名称和阶段加载特定模型。
    阶段可以是 'fp32_baseline', 'Finetuned', 'PTQ_INT8', 'QAT_INT8'。
    """
    model_path = os.path.join(cfg.MODEL_SAVE_DIR, f"{exp_name}_{stage}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到，实验名称: {exp_name}, 阶段: {stage}: {model_path}")

    model = HybridCNNTransformer(num_classes=num_classes, model_args=model_args, image_size=image_size)

    # 处理量化模型的加载
    if stage in ['PTQ_INT8', 'QAT_INT8']:
        temp_model = HybridCNNTransformer(num_classes=num_classes, model_args=model_args, image_size=image_size)
        temp_model_prepared = temp_model

        if stage == 'QAT_INT8':
            temp_model_prepared = prepare_model_for_quantization(temp_model, is_qat=True,
                                                                 backend=cfg.DEFAULT_COMPRESSION_PARAMS[
                                                                     'quantization_backend'])
        elif stage == 'PTQ_INT8':
            temp_model_prepared = prepare_model_for_quantization(temp_model, is_qat=False,
                                                                 backend=cfg.DEFAULT_COMPRESSION_PARAMS[
                                                                     'quantization_backend'])
            # 对于加载已保存的 PTQ 模型，我们需要应用 qconfig 和 prepare，但不需要用数据校准
            temp_model_prepared.qconfig = torch.ao.quantization.get_default_qconfig(
                cfg.DEFAULT_COMPRESSION_PARAMS['quantization_backend'])
            torch.ao.quantization.prepare(temp_model_prepared, inplace=True)

        # 转换准备好的模型为量化格式 (例如，将 nn.Conv2d 替换为 nnq.Conv2d)
        model = convert_to_quantized_model(temp_model_prepared)  # 这也会将模型移动到 CPU
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"已加载 {stage} 模型: {model_path}")
        model.eval()  # 确保模型处于评估模式以便推理

    else:  # FP32 或微调模型 (也是 FP32)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"已加载 FP32 模型 {stage}: {model_path}")
        model.to(device)
        model.eval()

    return model


def visualize_feature_maps(model: nn.Module, dataset_args: Dict[str, Any], training_args: Dict[str, Any],
                           device: torch.device, num_images: int = 3, layer_indices: List[int] = [0, 4, 8],
                           save_path: Optional[str] = None, model_label: str = "模型"):
    """
    可视化 CNN 主干网络的特征图。
    包含图片保存、较小的图片尺寸和增大的间距。
    """
    print("\n--- 可视化特征图 ---")
    _, testloader, _ = get_dataloaders(dataset_args, training_args)

    # 获取一批图片
    images, _ = next(iter(testloader))
    images = images[:num_images].to(device)

    # 确保模型处于评估模式
    model.eval()

    # 调整 figsize 以适应较小图片、更多行和更好的间距
    # (宽度用于特征图，高度用于图片). 增大了基础高度以提供更多间距。
    fig, axes = plt.subplots(num_images, len(layer_indices) + 1,
                             figsize=(2.5 * (len(layer_indices) + 1), 2.0 * num_images))
    if num_images == 1:  # 处理只有一张图片的情况
        axes = axes[np.newaxis, :]  # 确保是二维数组以便一致的索引

    # 调整子图间距 (hspace 用于垂直，wspace 用于水平)
    fig.subplots_adjust(hspace=0.5, wspace=0.1, top=0.9)  # 增大 hspace 用于垂直间距，调整 top 以适应主标题

    for img_idx in range(num_images):
        # 原始图片
        ax_orig = axes[img_idx, 0]
        # 去归一化并显示图片
        # 错误行已修复：从 dataset 模块直接获取 DATASET_REGISTRY
        mean = dataset.DATASET_REGISTRY[dataset_args['name']]['mean']
        std = dataset.DATASET_REGISTRY[dataset_args['name']]['std']
        img_display = images[img_idx].cpu().clone()
        for c in range(3):
            img_display[c] = img_display[c] * std[c] + mean[c]
        img_display = np.clip(img_display.permute(1, 2, 0).numpy(), 0, 1)
        ax_orig.imshow(img_display)
        ax_orig.set_title(f"输入 {img_idx + 1}", fontsize=8)  # 较小的标题字体
        ax_orig.axis('off')

        # 特征图提取
        features_to_visualize = []
        with torch.no_grad():
            temp_x = images[img_idx:img_idx + 1]
            # 在评估时，确保 BatchNorm 处于评估模式以保持行为一致
            model.eval()

            # 手动遍历 CNN 主干网络以捕获中间特征
            current_layer_idx_in_sequential = 0
            for j, (name, module) in enumerate(model.cnn_backbone.features.named_children()):
                temp_x = module(temp_x)
                if current_layer_idx_in_sequential in layer_indices:  # 检查是否需要此层的输出
                    features_to_visualize.append(temp_x)
                current_layer_idx_in_sequential += 1

        if not features_to_visualize:
            print(f"  警告：未找到指定层索引的特征。跳过。")
            continue

        for i, feature_map_tensor in enumerate(features_to_visualize):
            # 确保索引查找安全，以防实际收集到的特征少于请求的特征
            if i >= len(layer_indices):
                break

            feature_map = feature_map_tensor.squeeze(0).cpu().numpy()  # [C, H, W] 格式

            # 选择一些特征图进行可视化 (例如，最多 8 个通道)
            num_channels_to_show = min(feature_map.shape[0], 8)
            grid = make_grid(torch.from_numpy(feature_map[:num_channels_to_show, :, :]).unsqueeze(1),
                             nrow=min(num_channels_to_show, 4), normalize=True, scale_each=True, padding=1)

            ax = axes[img_idx, i + 1]
            ax.imshow(grid.permute(1, 2, 0))
            ax.set_title(f"层 {layer_indices[i]} 特征", fontsize=8)  # 较小的标题字体
            ax.axis('off')

    plt.suptitle(f"特征图: {model_label}", y=0.98, fontsize=10)  # 调整主标题 y 位置和字体大小

    if save_path:
        filename = f"特征图_{model_label.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', dpi=300)  # 保存图片，使用紧密边框和高 DPI
        print(f"  特征图已保存到: {os.path.join(save_path, filename)}")
    plt.show()
    plt.close(fig)  # 关闭图表以释放内存


def visualize_weight_distribution(model: nn.Module, model_label: str, save_path: Optional[str] = None):
    """
    可视化卷积层和线性层的权重分布。
    对于量化模型，它会尝试可视化实际的量化权重。
    包含图片保存、较小的图片尺寸和增大的间距。
    """
    print(f"\n--- 可视化权重分布 ---")

    # 过滤模块，只包含有权重可绘制的模块
    plottable_modules = []
    for name, module in model.named_modules():
        if isinstance(module, (
        nn.Conv2d, nn.Linear, torch.ao.nn.quantized.modules.conv.Conv2d, torch.ao.nn.quantized.modules.linear.Linear)):
            # 检查是否有 'weight' 属性，直接获取或通过量化方法获取
            if (hasattr(module, 'weight') and module.weight is not None) or \
                    (isinstance(module, (torch.ao.nn.quantized.modules.conv.Conv2d,
                                         torch.ao.nn.quantized.modules.linear.Linear)) and hasattr(module,
                                                                                                   'weight') and callable(
                        module.weight)):
                plottable_modules.append((name, module))

    if not plottable_modules:
        print(f"  未找到可绘制的 Conv2d/Linear/量化模块: {model_label}。")
        return

    # 调整 figsize 以适应较小图片和更好的间距
    fig, axes = plt.subplots(len(plottable_modules), 1, figsize=(7, 3.0 * len(plottable_modules)))  # 调整高度乘数
    if len(plottable_modules) == 1:
        axes = [axes]  # 确保是可迭代的以便一致的索引

    fig.subplots_adjust(hspace=0.6, top=0.92)  # 增大 hspace 用于垂直间距，调整 top 以适应主标题

    is_quantized_model = any(
        isinstance(m, (torch.ao.nn.quantized.modules.conv.Conv2d, torch.ao.nn.quantized.modules.linear.Linear)) for m in
        model.modules())

    for idx, (name, module) in enumerate(plottable_modules):
        if is_quantized_model and isinstance(module, (
        torch.ao.nn.quantized.modules.conv.Conv2d, torch.ao.nn.quantized.modules.linear.Linear)):
            weights = module.weight().detach().cpu().numpy().flatten()  # 对于量化模块，调用 .weight()
            title = f"{name} (INT8 权重)"
            color = 'lightcoral'
        else:  # 假设为 FP32
            weights = module.weight.detach().cpu().numpy().flatten()
            title = f"{name} (FP32 权重)"
            color = 'skyblue'

        ax = axes[idx]
        sns.histplot(weights, bins=50, kde=True, ax=ax, color=color, stat='density')  # 使用 stat='density' 以保持一致的比例
        ax.set_title(title, fontsize=9)  # 较小的标题字体
        ax.set_xlabel("权重值", fontsize=8)  # 较小的轴标签字体
        ax.set_ylabel("密度", fontsize=8)  # 较小的轴标签字体
        ax.tick_params(axis='x', labelsize=7)  # 较小的刻度标签字体
        ax.tick_params(axis='y', labelsize=7)

    plt.suptitle(f"权重分布: {model_label}", y=0.98, fontsize=11)  # 调整主标题 y 位置和字体大小

    if save_path:
        filename = f"权重分布_{model_label.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', dpi=300)  # 保存图片，使用紧密边框和高 DPI
        print(f"  权重分布已保存到: {os.path.join(save_path, filename)}")
    plt.show()
    plt.close(fig)  # 关闭图表以释放内存


def plot_performance_metrics(all_results: Dict[str, Any], save_path: Optional[str] = None):
    """
    绘制所有实验的性能指标图。
    确保 X 轴标签水平显示，字体较小，图表更宽，并保存图片。
    """
    print("\n--- 绘制性能指标图 ---")

    metrics_to_plot = ['acc', 'size_mb', 'gflops', 'inf_time_ms']
    titles = {
        'acc': '准确率 (%)',
        'size_mb': '模型大小 (MB)',
        'gflops': 'FLOPs (GFLOPs)',
        'inf_time_ms': '推理时间 (ms/图片)'
    }
    y_labels = {
        'acc': '准确率 (%)',
        'size_mb': '大小 (MB)',
        'gflops': 'FLOPs',
        'inf_time_ms': '时间 (ms)'
    }

    experiment_labels = []
    baseline_accs = {}  # 用于存储基线准确率以便比较

    # 收集数据，重点关注每个实验的最终阶段
    plot_data = {metric: [] for metric in metrics_to_plot}

    # 对实验进行排序以保持绘图顺序一致
    sorted_exp_names = sorted(all_results.keys())

    for exp_name in sorted_exp_names:
        exp_results = all_results[exp_name]

        final_stage_metrics = None
        stage_label = ""

        if exp_results['config'].get('is_fp32_baseline_generation'):
            if 'FP32_Initial_Train' in exp_results['metrics']:
                final_stage_metrics = exp_results['metrics']['FP32_Initial_Train']
                stage_label = "FP32 基线"
            elif 'FP32_Loaded_Baseline' in exp_results['metrics']:
                final_stage_metrics = exp_results['metrics']['FP32_Loaded_Baseline']
                stage_label = "FP32 基线"
        else:
            if 'QAT_INT8' in exp_results['metrics']:
                final_stage_metrics = exp_results['metrics']['QAT_INT8']
                stage_label = "QAT"
            elif 'PTQ_INT8' in exp_results['metrics']:
                final_stage_metrics = exp_results['metrics']['PTQ_INT8']
                stage_label = "PTQ"
            elif 'Finetuned' in exp_results['metrics']:
                final_stage_metrics = exp_results['metrics']['Finetuned']
                stage_label = "微调剪枝"
            elif 'Pruned' in exp_results['metrics']:
                final_stage_metrics = exp_results['metrics']['Pruned']
                stage_label = "剪枝 (未微调)"
            elif 'FP32_Loaded_Baseline' in exp_results['metrics']:  # 对于未微调的 P0QAT/P0PTQ 实验
                final_stage_metrics = exp_results['metrics']['FP32_Loaded_Baseline']
                stage_label = "FP32 (未压缩)"

        if final_stage_metrics:
            # 缩短标签，只保留最重要的信息
            short_name = exp_name.replace('Baseline_', '').replace('_CIFAR10', '_C10').replace('_SVHN',
                                                                                               '_SVHN').replace(
                'Serial_', 'S_').replace('Parallel_', 'P_').replace('_FP32', '')
            experiment_labels.append(f"{short_name} ({stage_label})")
            for metric in metrics_to_plot:
                plot_data[metric].append(final_stage_metrics.get(metric, 0))

            if 'Baseline_FP32' in exp_name:
                baseline_accs[exp_name.replace('Baseline_FP32_', '')] = final_stage_metrics['acc']

    num_plots = len(metrics_to_plot)
    # 调整 figsize 以使图表更宽，并更好地垂直间距
    fig, axes = plt.subplots(num_plots, 1, figsize=(18, 4.0 * num_plots))

    if num_plots == 1:
        axes = [axes]  # 确保 axes 是可迭代的，即使只有一个子图

    x = np.arange(len(experiment_labels))

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        bars = ax.bar(x, plot_data[metric], label=titles[metric], color=plt.cm.Paired(x / len(x)))
        ax.set_ylabel(y_labels[metric], fontsize=9)  # 较小的 Y 轴标签字体
        ax.set_title(titles[metric], fontsize=10)  # 较小的标题字体
        ax.set_xticks(x)
        # 标签保持水平，字体大小调整以保持紧凑
        ax.set_xticklabels(experiment_labels, rotation=0, ha='center', fontsize=6)
        ax.tick_params(axis='x', labelsize=6)  # 较小的 X 轴刻度标签字体
        ax.tick_params(axis='y', labelsize=7)  # 较小的 Y 轴刻度标签字体
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # 在柱状图上方添加数值标签
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom', ha='center',
                    fontsize=5)  # 较小的值标签字体

        # 为相关图表添加基线准确率的水平线
        if metric == 'acc':
            for base_key, base_acc in baseline_accs.items():
                if "S_C10" in experiment_labels[i] and "CIFAR10" in base_key and "Serial" in base_key:
                    ax.axhline(y=base_acc, color='r', linestyle='--', label=f'串行C10 基线准确率')
                elif "P_C10" in experiment_labels[i] and "CIFAR10" in base_key and "Parallel" in base_key:
                    ax.axhline(y=base_acc, color='r', linestyle='--', label=f'并行C10 基线准确率')
                elif "S_SVHN" in experiment_labels[i] and "SVHN" in base_key and "Serial" in base_key:
                    ax.axhline(y=base_acc, color='r', linestyle='--', label=f'串行SVHN 基线准确率')
                elif "P_SVHN" in experiment_labels[i] and "SVHN" in base_key and "Parallel" in base_key:
                    ax.axhline(y=base_acc, color='r', linestyle='--', label=f'并行SVHN 基线准确率')
                ax.legend(fontsize=7, loc='best')  # 较小图例字体，最优位置

    # 调整整体布局以防止标签与下一个子图的标题重叠。
    # `h_pad` 增加了子图之间的垂直填充。
    # `rect` 调整了子图参数以实现整体紧凑布局，并为主标题留出空间。
    plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=2.0)

    if save_path:
        filename = "性能指标总结.png"
        plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', dpi=300)  # 保存图片
        print(f"  性能指标总结已保存到: {os.path.join(save_path, filename)}")
    plt.show()
    plt.close(fig)  # 关闭图表以释放内存


def main():
    parser = argparse.ArgumentParser(description="可视化并比较模型性能和内部状态。")
    parser.add_argument('--experiment_name', type=str, required=False,
                        help="要可视化的特定实验 (例如：'Serial_CIFAR10_P40_QAT')。")
    parser.add_argument('--stage', type=str, default='all',
                        choices=['fp32_baseline', 'Finetuned', 'PTQ_INT8', 'QAT_INT8', 'all','pruned_40_finetuned'],  # <-- 修正了这里
                        help="要可视化的特定模型阶段。'all' 表示绘制性能图。")
    parser.add_argument('--visualize_feature_maps', action='store_true',
                        help="可视化选定模型的特征图。")
    parser.add_argument('--visualize_weight_distribution', action='store_true',
                        help="可视化选定模型的权重分布。")
    parser.add_argument('--plot_performance_metrics', action='store_true',
                        help="绘制所有实验的整体性能指标图。")
    parser.add_argument('--save_plots', action='store_true',
                        help="将生成的图表保存到 results/plots 目录下的文件中。")

    args = parser.parse_args()

    device = cfg.DEVICE
    print(f"可视化设备: {device}")

    # 如果启用了保存，设置保存路径
    save_plot_dir = PLOT_SAVE_DIR if args.save_plots else None
    if save_plot_dir and not os.path.exists(save_plot_dir):
        os.makedirs(save_plot_dir)
        print(f"已创建图表保存目录: {save_plot_dir}")

    # --- 选项 1: 从 JSON 结果绘制整体性能图 ---
    if args.plot_performance_metrics:
        results_summary_path = os.path.join(cfg.RESULTS_DIR, "_ALL_EXPERIMENTS_SUMMARY.json")
        if not os.path.exists(results_summary_path):
            print(f"错误：总结结果文件未找到: {results_summary_path}。请先运行 main.py。")
            return

        with open(results_summary_path, 'r', encoding='utf-8') as f:  # 指定编码以处理可能的中文
            all_results = json.load(f)
        plot_performance_metrics(all_results, save_path=save_plot_dir)

    # --- 选项 2: 可视化特定模型的详细信息 (特征图, 权重) ---
    elif args.visualize_feature_maps or args.visualize_weight_distribution:
        if not args.experiment_name or not args.stage:
            print("错误：要可视化特征图或权重分布，必须提供 --experiment_name 和 --stage 参数。")
            return

        try:
            exp_params = cfg.get_experiment_params(args.experiment_name)
            dataset_args = exp_params['dataset_args']
            training_args = exp_params['training_args']
            model_args = exp_params['model_args']

            # 通过 dataset.get_dataloaders 获取 num_classes 和 image_size
            # get_dataloaders 会返回 num_classes，image_size 从 dataset_args 获取
            _, _, num_classes = get_dataloaders(dataset_args=dataset_args, training_args=training_args)
            image_size = dataset_args.get('image_size', cfg.DEFAULT_DATASET_CONFIG['image_size'])

            model_to_visualize = load_model_for_visualization(
                args.experiment_name, args.stage, num_classes, model_args, image_size, device
            )

            # 为保存的文件创建标签
            model_label_for_file = f"{args.experiment_name}_{args.stage}"

            if args.visualize_feature_maps:
                # layer_indices 可以是动态的或用户定义的，
                # 这里使用固定集合，根据你的 CNN 主干网络结构进行调整
                visualize_feature_maps(model_to_visualize, dataset_args, exp_params['training_args'], device,
                                       num_images=3, layer_indices=[0, 4, 8],  # 示例索引，根据需要调整
                                       save_path=save_plot_dir, model_label=model_label_for_file)

            if args.visualize_weight_distribution:
                visualize_weight_distribution(model_to_visualize, model_label_for_file, save_path=save_plot_dir)

        except FileNotFoundError as e:
            print(f"错误：{e}")
        except ValueError as e:
            print(f"配置错误：{e}")
        except Exception as e:
            print(f"可视化过程中发生意外错误：{e}")
            import traceback
            traceback.print_exc()
    else:
        print("未选择可视化选项。使用 --help 查看可用选项。")


if __name__ == '__main__':
    main()