# config.py
import torch
import os
import platform
import copy  # 确保导入 copy
import json  # 导入 json 用于打印完整配置

# --- ⚙️ Global Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'saved_models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# --- 🖼️ Dataset Configuration ---
DEFAULT_DATASET_CONFIG = {
    'name': 'CIFAR10',
    'image_size': 32,
    'num_workers': 2,
    'num_classes': 10,  # 添加 num_classes 便于模型初始化时获取
}

# --- 🧠 Model Hyperparameters ---
DEFAULT_MODEL_PARAMS = {
    'cnn_output_channels': 128,
    'transformer_layers': 2,
    'transformer_heads': 4,
    'transformer_dim_feedforward': 256,
    'use_pos_encoding': True,
    'transformer_dropout': 0.1,
    'pos_encoding_learnable': True,
    'cls_token_init_std': 0.02,
    'parallel_mlp_hidden_channels': 64,
    'parallel_mlp_output_channels': 32,
}

# --- 🏋️ Training Hyperparameters ---
DEFAULT_TRAINING_PARAMS = {
    'batch_size': 128,
    'base_lr': 0.001,
    'weight_decay': 1e-4,
    'initial_train_epochs': 20,
    'finetune_epochs': 10,
    'lr_finetune': 0.0001,
    'optimizer_type_finetune': 'adamw',
    'qat_epochs': 10,
    'lr_qat': 0.00001,
    'optimizer_type_qat': 'adamw',
    'lr_scheduler_type': 'cosine',
    'cosine_t_max_factor': 1.0,
    'step_lr_step_size': 7,
    'step_lr_gamma': 0.1,
    'optimizer_type': 'adamw',
    'global_seed': 42,
}

# --- ✂️ Compression Hyperparameters ---
DEFAULT_COMPRESSION_PARAMS = {
    'pruning_amount': 0.4,
    'quantization_type': None,
    'ptq_calib_batches': 20,
    'quantization_backend': (
        'fbgemm' if platform.system() == "Windows"
        else ('qnnpack' if hasattr(os, 'uname') and 'arm' in os.uname().machine.lower() else 'fbgemm')
    ),
}

# --- 🧪 Experiment Definitions ---
EXPERIMENT_CONFIGS = [
    # --- CIFAR10 Experiments ---
    {
        'name': 'Baseline_FP32_Serial_CIFAR10',
        'is_fp32_baseline_generation': True,
        'dataset_args': {'name': 'CIFAR10', 'image_size': 32},
        'model_args': {'fusion_strategy': 'serial', **DEFAULT_MODEL_PARAMS},
        'training_args': {
            'initial_train_epochs': DEFAULT_TRAINING_PARAMS['initial_train_epochs'],
            'base_lr': DEFAULT_TRAINING_PARAMS['base_lr'],
            'optimizer_type': DEFAULT_TRAINING_PARAMS['optimizer_type'],
            'lr_scheduler_type': DEFAULT_TRAINING_PARAMS['lr_scheduler_type'],
        },
        'compression_args': {'pruning_amount': 0, 'quantization_type': None},
    },
    {
        'name': 'Baseline_FP32_Parallel_CIFAR10',
        'is_fp32_baseline_generation': True,
        'dataset_args': {'name': 'CIFAR10', 'image_size': 32},
        'model_args': {'fusion_strategy': 'parallel', **DEFAULT_MODEL_PARAMS, 'use_pos_encoding': False},
        'training_args': {
            'initial_train_epochs': DEFAULT_TRAINING_PARAMS['initial_train_epochs'],
            'base_lr': DEFAULT_TRAINING_PARAMS['base_lr'],
            'optimizer_type': DEFAULT_TRAINING_PARAMS['optimizer_type'],
            'lr_scheduler_type': DEFAULT_TRAINING_PARAMS['lr_scheduler_type'],
        },
        'compression_args': {'pruning_amount': 0, 'quantization_type': None},
    },
    {
        'name': 'Serial_CIFAR10_P40_F_QPTQ',
        'load_fp32_baseline_from_experiment': 'Baseline_FP32_Serial_CIFAR10',
        'dataset_args': {'name': 'CIFAR10', 'image_size': 32},
        'model_args': {'fusion_strategy': 'serial', **DEFAULT_MODEL_PARAMS},
        'training_args': {
            'initial_train_epochs': 0,
            'finetune_epochs': DEFAULT_TRAINING_PARAMS['finetune_epochs'],
            'lr_finetune': DEFAULT_TRAINING_PARAMS['lr_finetune'],
            'optimizer_type_finetune': DEFAULT_TRAINING_PARAMS['optimizer_type_finetune'],
            'lr_scheduler_type': DEFAULT_TRAINING_PARAMS['lr_scheduler_type'],
        },
        'compression_args': {
            'pruning_amount': DEFAULT_COMPRESSION_PARAMS['pruning_amount'],
            'quantization_type': 'ptq',
            'ptq_calib_batches': DEFAULT_COMPRESSION_PARAMS['ptq_calib_batches'],
        },
    },
    {
        'name': 'Serial_CIFAR10_P40_QAT',
        'load_fp32_baseline_from_experiment': 'Baseline_FP32_Serial_CIFAR10',
        'dataset_args': {'name': 'CIFAR10', 'image_size': 32},
        'model_args': {'fusion_strategy': 'serial', **DEFAULT_MODEL_PARAMS},
        'training_args': {
            'initial_train_epochs': 0,
            'finetune_epochs': DEFAULT_TRAINING_PARAMS['finetune_epochs'],
            'lr_finetune': DEFAULT_TRAINING_PARAMS['lr_finetune'],
            'optimizer_type_finetune': DEFAULT_TRAINING_PARAMS['optimizer_type_finetune'],
            'qat_epochs': DEFAULT_TRAINING_PARAMS['qat_epochs'],
            'lr_qat': DEFAULT_TRAINING_PARAMS['lr_qat'],
            'optimizer_type_qat': DEFAULT_TRAINING_PARAMS['optimizer_type_qat'],
            'lr_scheduler_type': DEFAULT_TRAINING_PARAMS['lr_scheduler_type'],
        },
        'compression_args': {
            'pruning_amount': DEFAULT_COMPRESSION_PARAMS['pruning_amount'],
            'quantization_type': 'qat',
        },
    },
    {
        'name': 'Parallel_CIFAR10_P40_QAT',
        'load_fp32_baseline_from_experiment': 'Baseline_FP32_Parallel_CIFAR10',
        'dataset_args': {'name': 'CIFAR10', 'image_size': 32},
        'model_args': {'fusion_strategy': 'parallel', **DEFAULT_MODEL_PARAMS, 'use_pos_encoding': False},
        'training_args': {
            'initial_train_epochs': 0,
            'finetune_epochs': DEFAULT_TRAINING_PARAMS['finetune_epochs'],
            'lr_finetune': DEFAULT_TRAINING_PARAMS['lr_finetune'],
            'optimizer_type_finetune': DEFAULT_TRAINING_PARAMS['optimizer_type_finetune'],
            'qat_epochs': DEFAULT_TRAINING_PARAMS['qat_epochs'],
            'lr_qat': DEFAULT_TRAINING_PARAMS['lr_qat'],
            'optimizer_type_qat': DEFAULT_TRAINING_PARAMS['optimizer_type_qat'],
            'lr_scheduler_type': DEFAULT_TRAINING_PARAMS['lr_scheduler_type'],
        },
        'compression_args': {
            'pruning_amount': DEFAULT_COMPRESSION_PARAMS['pruning_amount'],
            'quantization_type': 'qat',
        },
    },
    {
        'name': 'Parallel_CIFAR10_P40_F_QPTQ',
        'load_fp32_baseline_from_experiment': 'Baseline_FP32_Parallel_CIFAR10',
        'dataset_args': {'name': 'CIFAR10', 'image_size': 32},
        'model_args': {'fusion_strategy': 'parallel', **DEFAULT_MODEL_PARAMS, 'use_pos_encoding': False},
        'training_args': {
            'initial_train_epochs': 0,
            'finetune_epochs': DEFAULT_TRAINING_PARAMS['finetune_epochs'],
            'lr_finetune': DEFAULT_TRAINING_PARAMS['lr_finetune'],
            'optimizer_type_finetune': DEFAULT_TRAINING_PARAMS['optimizer_type_finetune'],
            'lr_scheduler_type': DEFAULT_TRAINING_PARAMS['lr_scheduler_type'],
        },
        'compression_args': {
            'pruning_amount': DEFAULT_COMPRESSION_PARAMS['pruning_amount'],
            'quantization_type': 'ptq',
            'ptq_calib_batches': DEFAULT_COMPRESSION_PARAMS['ptq_calib_batches'],
        },
    },
    {
        'name': 'Serial_CIFAR10_P0_QPTQ',
        'load_fp32_baseline_from_experiment': 'Baseline_FP32_Serial_CIFAR10',
        'dataset_args': {'name': 'CIFAR10', 'image_size': 32},
        'model_args': {'fusion_strategy': 'serial', **DEFAULT_MODEL_PARAMS},
        'training_args': {
            'initial_train_epochs': 0,
            'finetune_epochs': 0,
        },
        'compression_args': {
            'pruning_amount': 0,
            'quantization_type': 'ptq',
            'ptq_calib_batches': DEFAULT_COMPRESSION_PARAMS['ptq_calib_batches'],
        },
    },
    {
        'name': 'Serial_CIFAR10_P0_QAT',
        'load_fp32_baseline_from_experiment': 'Baseline_FP32_Serial_CIFAR10',
        'dataset_args': {'name': 'CIFAR10', 'image_size': 32},
        'model_args': {'fusion_strategy': 'serial', **DEFAULT_MODEL_PARAMS},
        'training_args': {
            'initial_train_epochs': 0,
            'finetune_epochs': 0,
            'qat_epochs': DEFAULT_TRAINING_PARAMS['qat_epochs'],
            'lr_qat': DEFAULT_TRAINING_PARAMS['lr_qat'],
            'optimizer_type_qat': DEFAULT_TRAINING_PARAMS['optimizer_type_qat'],
            'lr_scheduler_type': DEFAULT_TRAINING_PARAMS['lr_scheduler_type'],
        },
        'compression_args': {
            'pruning_amount': 0,
            'quantization_type': 'qat',
        },
    },

    # <--- 新增开始: SVHN Experiments ---
    {
        'name': 'Baseline_FP32_Serial_SVHN',
        'is_fp32_baseline_generation': True,
        'dataset_args': {'name': 'SVHN', 'image_size': 32},
        'model_args': {'fusion_strategy': 'serial', **DEFAULT_MODEL_PARAMS},
        'training_args': {
            'initial_train_epochs': DEFAULT_TRAINING_PARAMS['initial_train_epochs'],
            'base_lr': DEFAULT_TRAINING_PARAMS['base_lr'],
            'optimizer_type': DEFAULT_TRAINING_PARAMS['optimizer_type'],
            'lr_scheduler_type': DEFAULT_TRAINING_PARAMS['lr_scheduler_type'],
        },
        'compression_args': {'pruning_amount': 0, 'quantization_type': None},
    },
    {
        'name': 'Baseline_FP32_Parallel_SVHN',
        'is_fp32_baseline_generation': True,
        'dataset_args': {'name': 'SVHN', 'image_size': 32},
        'model_args': {'fusion_strategy': 'parallel', **DEFAULT_MODEL_PARAMS, 'use_pos_encoding': False},
        'training_args': {
            'initial_train_epochs': DEFAULT_TRAINING_PARAMS['initial_train_epochs'],
            'base_lr': DEFAULT_TRAINING_PARAMS['base_lr'],
            'optimizer_type': DEFAULT_TRAINING_PARAMS['optimizer_type'],
            'lr_scheduler_type': DEFAULT_TRAINING_PARAMS['lr_scheduler_type'],
        },
        'compression_args': {'pruning_amount': 0, 'quantization_type': None},
    },
    {
        'name': 'Serial_SVHN_P40_F_QPTQ',
        'load_fp32_baseline_from_experiment': 'Baseline_FP32_Serial_SVHN',
        'dataset_args': {'name': 'SVHN', 'image_size': 32},
        'model_args': {'fusion_strategy': 'serial', **DEFAULT_MODEL_PARAMS},
        'training_args': {
            'initial_train_epochs': 0,
            'finetune_epochs': DEFAULT_TRAINING_PARAMS['finetune_epochs'],
            'lr_finetune': DEFAULT_TRAINING_PARAMS['lr_finetune'],
            'optimizer_type_finetune': DEFAULT_TRAINING_PARAMS['optimizer_type_finetune'],
            'lr_scheduler_type': DEFAULT_TRAINING_PARAMS['lr_scheduler_type'],
        },
        'compression_args': {
            'pruning_amount': DEFAULT_COMPRESSION_PARAMS['pruning_amount'],
            'quantization_type': 'ptq',
            'ptq_calib_batches': DEFAULT_COMPRESSION_PARAMS['ptq_calib_batches'],
        },
    },
    {
        'name': 'Serial_SVHN_P40_QAT',
        'load_fp32_baseline_from_experiment': 'Baseline_FP32_Serial_SVHN',
        'dataset_args': {'name': 'SVHN', 'image_size': 32},
        'model_args': {'fusion_strategy': 'serial', **DEFAULT_MODEL_PARAMS},
        'training_args': {
            'initial_train_epochs': 0,
            'finetune_epochs': DEFAULT_TRAINING_PARAMS['finetune_epochs'],
            'lr_finetune': DEFAULT_TRAINING_PARAMS['lr_finetune'],
            'optimizer_type_finetune': DEFAULT_TRAINING_PARAMS['optimizer_type_finetune'],
            'qat_epochs': DEFAULT_TRAINING_PARAMS['qat_epochs'],
            'lr_qat': DEFAULT_TRAINING_PARAMS['lr_qat'],
            'optimizer_type_qat': DEFAULT_TRAINING_PARAMS['optimizer_type_qat'],
            'lr_scheduler_type': DEFAULT_TRAINING_PARAMS['lr_scheduler_type'],
        },
        'compression_args': {
            'pruning_amount': DEFAULT_COMPRESSION_PARAMS['pruning_amount'],
            'quantization_type': 'qat',
        },
    },
    # <--- 新增结束 ---
]


# --- 确保目录存在 ---
def ensure_dirs():
    for dir_path in [DATA_DIR, MODEL_SAVE_DIR, RESULTS_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


ensure_dirs()


# --- Helper function to merge configs ---
def get_experiment_params(exp_config_name: str) -> dict:
    """
    根据给定的实验名称，获取并合并完整的实验参数字典。
    """
    exp_conf = next((c for c in EXPERIMENT_CONFIGS if c['name'] == exp_config_name), None)
    if not exp_conf:
        raise ValueError(f"Experiment configuration '{exp_config_name}' not found.")

    # 深度复制默认配置，避免修改全局默认值
    params = {
        'dataset_args': copy.deepcopy(DEFAULT_DATASET_CONFIG),
        'model_args': copy.deepcopy(DEFAULT_MODEL_PARAMS),
        'training_args': copy.deepcopy(DEFAULT_TRAINING_PARAMS),
        'compression_args': copy.deepcopy(DEFAULT_COMPRESSION_PARAMS),
        'name': exp_conf['name']
    }

    # 合并特定实验的配置
    for key_group in ['dataset_args', 'model_args', 'training_args', 'compression_args']:
        if key_group in exp_conf:
            params[key_group].update(exp_conf[key_group])

    # 额外处理实验中可能有的顶层参数
    for key, value in exp_conf.items():
        if key not in ['dataset_args', 'model_args', 'training_args', 'compression_args', 'name']:
            params[key] = value

    # 添加全局路径和设备信息
    params['device'] = DEVICE
    params['model_save_dir'] = MODEL_SAVE_DIR
    params['results_dir'] = RESULTS_DIR
    params['data_dir'] = DATA_DIR
    return params


if __name__ == '__main__':
    print(f"Global device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    ensure_dirs()
    print("\n--- All Defined Experiment Names ---")
    for i, conf in enumerate(EXPERIMENT_CONFIGS):
        print(f"{i + 1}. {conf['name']}")
    print("\n--- Testing parameter resolution for one new experiment ---")
    try:
        svhn_exp_name = 'Serial_SVHN_P40_QAT'
        print(f"Resolving parameters for '{svhn_exp_name}':")
        svhn_params = get_experiment_params(svhn_exp_name)
        print(json.dumps(svhn_params, indent=2, default=lambda o: str(o) if isinstance(o, torch.device) else o))
    except ValueError as e:
        print(f"Error loading params for {svhn_exp_name}: {e}")

    print("\nConfig file loaded and test print complete.")