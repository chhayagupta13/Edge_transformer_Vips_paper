# utils.py
import torch
import torch.nn as nn
import torch.nn.utils.prune as torch_prune
from torch.ao.quantization import get_default_qconfig, prepare_qat, prepare, convert, QuantStub, DeQuantStub
import torch.ao.nn.quantized as nnq
from torch.utils.data import DataLoader, Dataset
import os
import copy
from tqdm import tqdm
from typing import Tuple, Any, Type, Union

# Attempt to import thop
try:
    from thop import profile

    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False


def get_model_stats(
        model: nn.Module,
        input_size: Tuple[int, int, int] = (3, 32, 32),
        device: Union[str, torch.device] = 'cpu',
        label: str = "Model"
) -> Tuple[float, float, float]:
    """
    Calculates and prints model disk size, FLOPs, and parameters.
    """
    model.eval()
    model_on_device = model.to(device)

    print(f"\n--- Stats for {label} ---")

    # 1. Model disk size
    temp_model_path = "temp_model_for_stats.pth"
    try:
        torch.save(model_on_device.state_dict(), temp_model_path)
        size_mb = os.path.getsize(temp_model_path) / 1e6
        print(f"Size (on disk): {size_mb:.2f} MB")
    finally:
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

    # 2. Parameters
    total_params = sum(p.numel() for p in model_on_device.parameters())
    mparams_total = total_params / 1e6
    print(f"Parameters: {mparams_total:.2f} M")

    # 3. FLOPs (using thop)
    gflops = -1.0
    if THOP_AVAILABLE:
        try:
            dummy_input = torch.randn(1, *input_size, device=device)
            macs, _ = profile(copy.deepcopy(model_on_device), inputs=(dummy_input,), verbose=False)
            gflops = macs * 2 / 1e9
            print(f"FLOPs (FP32 equivalent): {gflops:.2f} GFLOPs")
        except Exception as e:
            print(f"Thop profiling failed: {e}. FLOPs calculation skipped.")
    else:
        print("FLOPs calculation skipped (thop not available).")

    print("-------------------------")
    return size_mb, gflops, mparams_total


def apply_channel_pruning(
        model_to_prune: nn.Module,
        pruning_amount: float = 0.3,
        module_types: Tuple[Type[nn.Module], ...] = (nn.Conv2d, nn.Linear)
) -> nn.Module:
    """
    Applies L1-norm based structured pruning to zero out entire channels/features.
    Note: This does not reduce FLOPs without model surgery.
    """
    if not (0.0 <= pruning_amount < 1.0):
        if pruning_amount == 0:
            print("\nPruning amount is 0. Skipping.")
            return model_to_prune
        raise ValueError(f"Pruning amount must be between 0.0 and < 1.0, got {pruning_amount}")

    print(f"\nApplying L1-Structured Channel Pruning (amount={pruning_amount})...")
    pruned_model = copy.deepcopy(model_to_prune)

    for module in pruned_model.modules():
        if isinstance(module, module_types):
            try:
                # For both Conv2d and Linear, dim=0 prunes the output dimension.
                torch_prune.ln_structured(module, name='weight', amount=pruning_amount, n=1, dim=0)
                torch_prune.remove(module, 'weight')  # Make pruning permanent
            except Exception as e:
                print(f"  Could not prune module {type(module).__name__}: {e}")

    print("Channel pruning application complete.")
    return pruned_model


def prepare_model_for_quantization(
        model_fp32: nn.Module,
        is_qat: bool = False,
        backend: str = 'fbgemm'
) -> nn.Module:
    """
    Prepares an FP32 model for quantization (PTQ or QAT).
    """
    quantizable_model = copy.deepcopy(model_fp32)
    quantizable_model.train() if is_qat else quantizable_model.eval()

    if hasattr(quantizable_model, '_fuse_modules_for_quantization'):
        print(f"Calling model's internal fusion method (is_qat={is_qat})...")
        quantizable_model._fuse_modules_for_quantization(is_qat=is_qat)

    quantizable_model.qconfig = get_default_qconfig(backend.lower())
    print(f"Preparing model for {'QAT' if is_qat else 'PTQ'} with backend '{backend.lower()}'...")

    if is_qat:
        prepare_qat(quantizable_model, inplace=True)
    else:
        prepare(quantizable_model, inplace=True)

    return quantizable_model


def calibrate_model_ptq(
        model_prepared: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_batches: int = 20
) -> None:
    """
    Calibrates a PTQ-prepared model with data.
    """
    if num_batches <= 0:
        print("num_batches is 0 or less. Skipping PTQ calibration.")
        return

    print(f"\nCalibrating PTQ model with {num_batches} batches on '{device}'...")
    model_prepared.eval()
    model_prepared.to(device)
    count = 0
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="PTQ Calibration", total=min(num_batches, len(dataloader))):
            if count >= num_batches:
                break
            model_prepared(inputs.to(device))
            count += 1
    print("PTQ Calibration complete.")


# <--- 从这里开始是关键修改 ---
def convert_to_quantized_model(
        model_to_convert: nn.Module
) -> nn.Module:
    """
    Converts a prepared (and calibrated/trained for PTQ/QAT respectively)
    model to an INT8 quantized model.
    """
    print("\nConverting to INT8 quantized model (on CPU)...")

    # 关键修复：先将整个模型（包括其观察器状态）移到 CPU
    model_to_convert.to('cpu')

    # 确保模型处于评估模式
    model_to_convert.eval()

    # 现在，在模型和所有相关状态都在 CPU 上的情况下执行转换
    convert(model_to_convert, inplace=True)

    print("Conversion to INT8 model complete.")
    # 返回被就地修改和转换后的模型
    return model_to_convert


# <--- 关键修改结束 ---


# --- Test block for utils.py ---

class DummyDataset(Dataset):
    def __init__(self, num_samples=100, image_size=(3, 32, 32), num_classes=10):
        self.num_samples, self.image_size, self.num_classes = num_samples, image_size, num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.randn(self.image_size), torch.randint(0, self.num_classes, (1,)).item()


class SimpleCNNForQuantization(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.quant = QuantStub()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False),  # bias=False for fusion
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, num_classes)
        self.dequant = DeQuantStub()

    def _fuse_modules_for_quantization(self, is_qat: bool = False):
        print(f"  SimpleCNN: Explicitly fusing modules (is_qat={is_qat})...")
        fuse_fn = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules
        fuse_fn(self.features, [['0', '1', '2']], inplace=True)

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        x = self.dequant(x)
        return x


if __name__ == '__main__':
    print("Testing utils.py...")
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quant_test_device = torch.device("cpu")  # Quantization ops are CPU-centric
    print(f"Using test device: {test_device}, Quantization device: {quant_test_device}")

    dummy_dataloader = DataLoader(DummyDataset(), batch_size=4)
    fp32_model = SimpleCNNForQuantization().to(test_device)

    print("\n--- Testing get_model_stats ---")
    get_model_stats(fp32_model, device=test_device, label="SimpleCNN_FP32")

    print("\n--- Testing apply_channel_pruning ---")
    pruned_model = apply_channel_pruning(fp32_model, pruning_amount=0.5)

    # Verification of pruning
    conv1_weights = pruned_model.features[0].weight.detach().cpu()
    zeroed_channels = torch.sum(torch.sum(torch.abs(conv1_weights), dim=(1, 2, 3)) == 0).item()
    print(f"  Verified: {zeroed_channels} of {pruned_model.features[0].out_channels} Conv2d channels are zeroed.")

    print("\n--- Testing PTQ Quantization Flow ---")
    ptq_model = SimpleCNNForQuantization().to(test_device)  # Start on test device
    prepared_ptq_model = prepare_model_for_quantization(ptq_model, is_qat=False)
    # Calibrate on the appropriate device
    calibrate_model_ptq(prepared_ptq_model, dummy_dataloader, test_device, num_batches=5)
    quantized_ptq_model = convert_to_quantized_model(prepared_ptq_model)  # This will move it to CPU

    print(f"  Type after PTQ convert: {type(quantized_ptq_model.features[0])}")
    assert isinstance(quantized_ptq_model.features[0], nnq.Conv2d)
    get_model_stats(quantized_ptq_model, device=quant_test_device, label="SimpleCNN_PTQ_INT8")

    print("\n--- Testing QAT Quantization Flow ---")
    qat_model = SimpleCNNForQuantization().to(test_device)  # Start on test device
    prepared_qat_model = prepare_model_for_quantization(qat_model, is_qat=True)
    # In a real scenario, you would train this model now on the test_device
    quantized_qat_model = convert_to_quantized_model(prepared_qat_model)  # This will move it to CPU

    print(f"  Type after QAT convert: {type(quantized_qat_model.features[0])}")
    assert isinstance(quantized_qat_model.features[0], nnq.Conv2d)
    get_model_stats(quantized_qat_model, device=quant_test_device, label="SimpleCNN_QAT_INT8")

    print("\nutils.py tests finished.")