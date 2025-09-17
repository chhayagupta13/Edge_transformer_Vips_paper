# train_eval.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import time
from typing import Tuple, Optional


def train_one_epoch(
        model: nn.Module,
        trainloader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        epoch_desc: str = "Training",
        scheduler: Optional[_LRScheduler] = None,
        grad_clip_value: Optional[float] = None
) -> Tuple[float, float]:
    """
    Trains the model for one epoch. Compatible with float and QAT models.
    """
    model.train()
    running_loss: float = 0.0
    correct_predictions: int = 0
    total_samples: int = 0
    progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=epoch_desc, leave=False)

    for i, (inputs, labels) in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if grad_clip_value is not None:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip_value)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        lr_to_display = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({
            'loss': f'{(running_loss / total_samples):.4f}',
            'acc': f'{(100. * correct_predictions / total_samples):.2f}%',
            'lr': f'{lr_to_display:.1e}'
        })

    if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()

    avg_epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    avg_epoch_acc = 100. * correct_predictions / total_samples if total_samples > 0 else 0
    return avg_epoch_loss, avg_epoch_acc


def evaluate(
        model: nn.Module,
        testloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        epoch_desc: str = "Evaluating"
) -> Tuple[float, float, float]:
    """
    Evaluates the model on the test set. Compatible with float, QAT, and converted quantized models.
    """
    model.eval()
    running_loss: float = 0.0
    correct_predictions: int = 0
    total_samples: int = 0
    total_inference_time_seconds: float = 0.0
    progress_bar = tqdm(enumerate(testloader), total=len(testloader), desc=epoch_desc, leave=False)

    with torch.no_grad():
        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            start_time = time.perf_counter()
            outputs = model(inputs)
            end_time = time.perf_counter()
            total_inference_time_seconds += (end_time - start_time)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'loss': f'{(running_loss / total_samples):.4f}',
                'acc': f'{(100. * correct_predictions / total_samples):.2f}%'
            })

    avg_epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    avg_epoch_acc = 100. * correct_predictions / total_samples if total_samples > 0 else 0
    avg_inference_time_ms_per_image = (total_inference_time_seconds / total_samples) * 1000 if total_samples > 0 else 0
    return avg_epoch_loss, avg_epoch_acc, avg_inference_time_ms_per_image


# <--- 修改从这里开始 ---
if __name__ == '__main__':
    print("Testing train_eval.py with project-specific components...")

    try:
        # --- Integration Test Setup ---
        # This test now uses the actual project components to ensure compatibility.
        import config as cfg
        from dataset import get_dataloaders
        from model import HybridCNNTransformer
        from torch.optim import AdamW
        from torch.ao.quantization import get_default_qconfig, prepare_qat, convert

        device_test = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device_test}")

        # 1. Load configuration for a baseline experiment
        exp_name = 'Baseline_FP32_Serial_CIFAR10'
        print(f"\n--- Loading configuration for experiment: '{exp_name}' ---")
        params = cfg.get_experiment_params(exp_name)

        # Use a smaller batch size for faster testing
        params['training_args']['batch_size'] = 16
        params['dataset_args']['num_workers'] = 0

        # 2. Load data
        trainloader, testloader, num_classes = get_dataloaders(
            dataset_args=params['dataset_args'],
            training_args=params['training_args']
        )
        # For quick testing, let's just use a small subset of the data
        train_subset = torch.utils.data.Subset(trainloader.dataset, range(128))
        test_subset = torch.utils.data.Subset(testloader.dataset, range(64))

        trainloader_test = DataLoader(train_subset, batch_size=params['training_args']['batch_size'])
        testloader_test = DataLoader(test_subset, batch_size=params['training_args']['batch_size'])

        # 3. Instantiate the actual model
        model_test = HybridCNNTransformer(
            num_classes=num_classes,
            model_args=params['model_args'],
            image_size=params['dataset_args']['image_size']
        ).to(device_test)

        criterion_test = nn.CrossEntropyLoss()
        optimizer_test = AdamW(model_test.parameters(), lr=0.001)

        # --- Test FP32 Training and Evaluation ---
        print("\n--- Testing train_one_epoch and evaluate with HybridCNNTransformer (FP32) ---")
        train_loss, train_acc = train_one_epoch(
            model_test, trainloader_test, criterion_test, optimizer_test, device_test, "Test Train FP32"
        )
        print(f"  Test Train Results: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")

        eval_loss, eval_acc, eval_time = evaluate(
            model_test, testloader_test, criterion_test, device_test, "Test Eval FP32"
        )
        print(f"  Test Eval Results: Loss={eval_loss:.4f}, Acc={eval_acc:.2f}%, Time={eval_time:.3f} ms/img")

        # --- Test QAT Simulation ---
        print("\n--- Simulating QAT workflow with HybridCNNTransformer ---")

        # 1. Instantiate a fresh model for QAT
        qat_model_instance = HybridCNNTransformer(
            num_classes=num_classes,
            model_args=params['model_args'],
            image_size=params['dataset_args']['image_size']
        )
        qat_model_instance.train()  # Set to train mode before fusion and preparation

        # 2. Fuse modules (a required step for real models)
        print("  Fusing modules for QAT...")
        qat_model_instance._fuse_modules_for_quantization(is_qat=True)

        # 3. Prepare for QAT
        qat_model_instance.qconfig = get_default_qconfig('fbgemm')
        print("  Preparing model for QAT...")
        prepared_model = prepare_qat(qat_model_instance, inplace=False).to(device_test)

        # 4. Simulate one QAT training epoch
        print("  Simulating one QAT training epoch...")
        qat_optimizer = AdamW(prepared_model.parameters(), lr=1e-5)
        qat_train_loss, qat_train_acc = train_one_epoch(
            prepared_model, trainloader_test, criterion_test, qat_optimizer, device_test, "Test QAT Train"
        )
        print(f"  QAT Train Results: Loss={qat_train_loss:.4f}, Acc={qat_train_acc:.2f}%")

        # 5. Evaluate the prepared (fake-quantized) model
        print("  Evaluating prepared QAT model...")
        prepared_eval_loss, prepared_eval_acc, prepared_eval_time = evaluate(
            prepared_model, testloader_test, criterion_test, device_test, "Test Eval QAT Prepared"
        )
        print(f"  Prepared Model Eval: Loss={prepared_eval_loss:.4f}, Acc={prepared_eval_acc:.2f}%")

        # 6. Convert to a fully quantized model (on CPU)
        print("  Converting to final INT8 model on CPU...")
        prepared_model.to('cpu')
        converted_model = convert(prepared_model, inplace=False)

        # 7. Evaluate the final INT8 model on CPU
        print("  Evaluating final INT8 model on CPU...")
        final_eval_loss, final_eval_acc, final_eval_time = evaluate(
            converted_model, testloader_test, criterion_test, torch.device('cpu'), "Test Eval INT8"
        )
        print(
            f"  Final INT8 Model Eval: Loss={final_eval_loss:.4f}, Acc={final_eval_acc:.2f}%, Time={final_eval_time:.3f} ms/img")

    except ImportError as e:
        print(f"\n[SKIPPED] Integration test skipped because a project file could not be imported: {e}")
        print("This is normal if you are running train_eval.py standalone without the full project.")
    except Exception as e:
        print(f"\nAn error occurred during the integration test: {e}")
        import traceback

        traceback.print_exc()

    print("\ntrain_eval.py tests finished.")