# dataset.py
import torch
import torchvision
import torchvision.transforms as transforms
import os
from typing import Tuple, Dict, Any, Optional

# 从 config.py 导入全局配置
try:
    import config as cfg
except ImportError:
    print("Warning: config.py not found directly. Using local defaults for DATA_DIR if not provided.")


    class cfg:
        DATA_DIR = './data/'
        # 模拟 config.py 中需要的一些默认值，以便独立运行 dataset.py 时不会报错
        DEFAULT_TRAINING_PARAMS = {'batch_size': 128}
        DEFAULT_DATASET_CONFIG = {'num_workers': 2}

# 数据集特定信息注册表
DATASET_REGISTRY = {
    'CIFAR10': {
        'loader': torchvision.datasets.CIFAR10,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010),
        'num_classes': 10,
        'default_image_size': 32,
    },
    'CIFAR100': {
        'loader': torchvision.datasets.CIFAR100,
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
        'num_classes': 100,
        'default_image_size': 32,
    },
    # <--- 新增开始 ---
    'SVHN': {
        'loader': torchvision.datasets.SVHN,
        'mean': (0.4377, 0.4438, 0.4728),
        'std': (0.1980, 0.2010, 0.1970),
        'num_classes': 10,
        'default_image_size': 32,
    }
    # <--- 新增结束 ---
}


def get_dataloaders(
        dataset_args: Dict[str, Any],
        training_args: Dict[str, Any],
        data_dir_override: Optional[str] = None,
        for_calibration: bool = False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """
    获取指定数据集的训练和测试 DataLoader。

    参数:
        dataset_args (dict): 包含数据集特定参数的字典，例如:
            'name': 数据集名称 (e.g., 'CIFAR10', 'SVHN')
            'image_size': 图像大小 (可选, 否则使用数据集默认值)
            'num_workers': DataLoader的 num_workers (可选, 否则使用config.py默认)
            'augmentation_policy': (str, 可选) 'standard', 'randaugment', 'autoaugment_cifar'
            'num_classes': (int, 可选) 数据集类别数，如果不在DATASET_REGISTRY中，可以从这里获取
        training_args (dict): 包含训练相关参数，主要用于获取 'batch_size'
        data_dir_override (str, 可选): 覆盖全局 DATA_DIR 的数据集路径。
        for_calibration (bool): 如果为 True，则训练集加载器将只应用基本变换 (Resize, ToTensor, Normalize)，
                                 不包含随机数据增强。这适用于 PTQ 校准。

    返回:
        trainloader (DataLoader): 训练数据加载器
        testloader (DataLoader): 测试数据加载器
        num_classes (int): 数据集的类别数量
    """
    dataset_name = dataset_args.get('name', 'CIFAR10')
    dataset_info = DATASET_REGISTRY.get(dataset_name)

    if dataset_info:
        DatasetClass = dataset_info['loader']
        mean = dataset_info['mean']
        std = dataset_info['std']
        num_classes = dataset_info['num_classes']
        image_size = dataset_args.get('image_size', dataset_info['default_image_size'])
    else:
        print(f"Warning: Dataset '{dataset_name}' not found in registry. "
              f"Attempting to proceed with parameters from dataset_args.")
        DatasetClass = None
        mean = dataset_args.get('mean')
        std = dataset_args.get('std')
        num_classes = dataset_args.get('num_classes')
        image_size = dataset_args.get('image_size')

        if not all([mean, std, num_classes, image_size]):
            raise ValueError(
                f"For unregistered dataset '{dataset_name}', 'mean', 'std', 'num_classes', 'image_size' must be provided in dataset_args.")
        if DatasetClass is None:
            raise ValueError(
                f"For unregistered dataset '{dataset_name}', a custom 'loader' (DatasetClass) must be provided in dataset_args or you need to extend DATASET_REGISTRY.")

    batch_size = training_args.get('batch_size', cfg.DEFAULT_TRAINING_PARAMS.get('batch_size', 128))
    num_workers = dataset_args.get('num_workers', cfg.DEFAULT_DATASET_CONFIG.get('num_workers', 2))
    data_dir = data_dir_override if data_dir_override else cfg.DATA_DIR
    augmentation_policy = dataset_args.get('augmentation_policy', 'standard').lower()

    transform_train_list = [transforms.Resize((image_size, image_size))]

    if for_calibration:
        print(f"Using basic transformations for calibration (no random augmentations).")
    else:
        # <--- 修改开始：调整数据增强逻辑，使其更通用 ---
        if augmentation_policy == 'standard':
            if image_size == 32 and dataset_name in ['CIFAR10', 'CIFAR100', 'SVHN']: # 添加 SVHN
                transform_train_list.append(transforms.RandomCrop(image_size, padding=4))
            else:
                transform_train_list.append(transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)))
            transform_train_list.append(transforms.RandomHorizontalFlip())
        # (后续的 augmentation policy 判断逻辑保持不变，因为它们是基于 torchvision 的通用实现)
        # ... (省略未更改的 elif 代码块) ...
        elif augmentation_policy == 'randaugment':
            try:
                transform_train_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
                print(f"Using RandAugment for {dataset_name} training.")
            except AttributeError:
                print(
                    "Warning: transforms.RandAugment not found (requires torchvision >= 0.11). Falling back to standard augmentation.")
                if image_size == 32:
                    transform_train_list.append(transforms.RandomCrop(image_size, padding=4))
                else:
                    transform_train_list.append(transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)))
                transform_train_list.append(transforms.RandomHorizontalFlip())
        elif augmentation_policy == 'autoaugment_cifar':
            try:
                if dataset_name in ['CIFAR10', 'CIFAR100']:
                    transform_train_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
                    print(f"Using AutoAugment (CIFAR10 policy) for {dataset_name} training.")
                else:
                    print(
                        f"Warning: AutoAugment (CIFAR10 policy) is not typically used for {dataset_name}. Falling back to standard augmentation.")
                    if image_size == 32:
                        transform_train_list.append(transforms.RandomCrop(image_size, padding=4))
                    else:
                        transform_train_list.append(transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)))
                    transform_train_list.append(transforms.RandomHorizontalFlip())

            except AttributeError:
                print(
                    "Warning: transforms.AutoAugment not found (requires torchvision >= 0.8). Falling back to standard augmentation.")
                if image_size == 32:
                    transform_train_list.append(transforms.RandomCrop(image_size, padding=4))
                else:
                    transform_train_list.append(transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)))
                transform_train_list.append(transforms.RandomHorizontalFlip())
        else:
            if augmentation_policy != 'standard':
                print(
                    f"Warning: Augmentation policy '{augmentation_policy}' not recognized or not suitable for {dataset_name}. Using standard augmentation.")
            if image_size == 32 and dataset_name in ['CIFAR10', 'CIFAR100', 'SVHN']: # 添加 SVHN
                transform_train_list.append(transforms.RandomCrop(image_size, padding=4))
            else:
                transform_train_list.append(transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)))
            transform_train_list.append(transforms.RandomHorizontalFlip())
        # <--- 修改结束 ---

    transform_train_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transforms.Compose(transform_train_list)

    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Creating it.")
        os.makedirs(data_dir)

    try:
        # <--- 修改开始：使数据集加载更灵活 ---
        # SVHN 使用 'split' 参数而不是 'train'
        if dataset_name == 'SVHN':
            trainset = DatasetClass(root=data_dir, split='train', download=True, transform=transform_train)
            testset = DatasetClass(root=data_dir, split='test', download=True, transform=transform_test)
        elif DatasetClass:
            trainset = DatasetClass(root=data_dir, train=True, download=True, transform=transform_train)
            testset = DatasetClass(root=data_dir, train=False, download=True, transform=transform_test)
        # <--- 修改结束 ---
        else:
            raise NotImplementedError(f"Custom DatasetClass for '{dataset_name}' not implemented in this function.")
    except Exception as e:
        print(f"Error loading dataset {dataset_name} from {data_dir}. Make sure it's accessible or can be downloaded.")
        print(f"If you previously downloaded to a different path, ensure 'data_dir' in config.py points there.")
        raise e

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True if not for_calibration else False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )

    print(
        f"Loaded {dataset_name}: train_samples={len(trainset)}, test_samples={len(testset)}, num_classes={num_classes}, image_size={image_size}x{image_size}, for_calibration={for_calibration}")
    return trainloader, testloader, num_classes


if __name__ == '__main__':
    print("Testing dataset.py...")

    try:
        data_directory_to_use = cfg.DATA_DIR
    except NameError:
        data_directory_to_use = './test_data/'
        print(f"cfg.DATA_DIR not found, using local test directory: {data_directory_to_use}")
        os.makedirs(data_directory_to_use, exist_ok=True)

    try:
        # --- 测试 CIFAR10 ---
        mock_dataset_args_cifar10 = {'name': 'CIFAR10', 'augmentation_policy': 'randaugment'}
        mock_training_args_cifar10 = {'batch_size': 64}
        print(f"\n--- Loading CIFAR10 with RandAugment ---")
        train_loader_c10, test_loader_c10, num_classes_c10 = get_dataloaders(
            dataset_args=mock_dataset_args_cifar10,
            training_args=mock_training_args_cifar10,
            data_dir_override=data_directory_to_use
        )
        print(
            f"CIFAR10: Num Classes = {num_classes_c10}, Train Batches = {len(train_loader_c10)}, Test Batches = {len(test_loader_c10)}")

        # --- 测试 CIFAR100 ---
        mock_dataset_args_cifar100 = {'name': 'CIFAR100', 'augmentation_policy': 'autoaugment_cifar'}
        mock_training_args_cifar100 = {'batch_size': 32}
        print(f"\n--- Loading CIFAR100 with AutoAugment ---")
        train_loader_c100, test_loader_c100, num_classes_c100 = get_dataloaders(
            dataset_args=mock_dataset_args_cifar100,
            training_args=mock_training_args_cifar100,
            data_dir_override=data_directory_to_use
        )
        print(
            f"CIFAR100: Num Classes = {num_classes_c100}, Train Batches = {len(train_loader_c100)}, Test Batches = {len(test_loader_c100)}")

        # <--- 新增开始 ---
        # --- 测试 SVHN ---
        mock_dataset_args_svhn = {'name': 'SVHN', 'augmentation_policy': 'standard'}
        mock_training_args_svhn = {'batch_size': 128}
        print(f"\n--- Loading SVHN with Standard Augmentation ---")
        train_loader_svhn, test_loader_svhn, num_classes_svhn = get_dataloaders(
            dataset_args=mock_dataset_args_svhn,
            training_args=mock_training_args_svhn,
            data_dir_override=data_directory_to_use
        )
        print(
            f"SVHN: Num Classes = {num_classes_svhn}, Train Batches = {len(train_loader_svhn)}, Test Batches = {len(test_loader_svhn)}")
        # <--- 新增结束 ---

        # --- 测试用于校准的加载器 ---
        print(f"\n--- Loading CIFAR10 for PTQ Calibration (no augmentations) ---")
        calib_train_loader_c10, _, _ = get_dataloaders(
            dataset_args={'name': 'CIFAR10'},
            training_args={'batch_size': 128},
            data_dir_override=data_directory_to_use,
            for_calibration=True
        )
        print(f"CIFAR10 (Calibration): Train Batches = {len(calib_train_loader_c10)}")


    except Exception as e:
        print(f"\nAn error occurred during dataset testing: {e}")
        import traceback
        traceback.print_exc()

    print("\ndataset.py tests finished.")