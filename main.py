# main.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
import json
import random
import numpy as np
import argparse
from typing import Dict, Any, Optional

# Import project files
import config as cfg
from dataset import get_dataloaders
from model import HybridCNNTransformer
from utils import (get_model_stats, apply_channel_pruning,
                   prepare_model_for_quantization, calibrate_model_ptq,
                   convert_to_quantized_model)
from train_eval import train_one_epoch, evaluate


def set_seed(seed_value=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def get_optimizer(model_params: Any, optim_config: Dict[str, Any]) -> optim.Optimizer:
    """Creates an optimizer based on configuration."""
    optim_type = optim_config.get('optimizer_type', 'adamw').lower()
    lr = optim_config.get('lr')  # LR should be passed in explicitly
    weight_decay = optim_config.get('weight_decay', 1e-4)

    if optim_type == 'adamw':
        return optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    elif optim_type == 'sgd':
        return optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:  # Default to Adam
        return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer: optim.Optimizer, scheduler_config: Dict[str, Any], num_epochs: int) -> Optional[
    optim.lr_scheduler._LRScheduler]:
    """Creates a learning rate scheduler."""
    scheduler_type = scheduler_config.get('lr_scheduler_type', 'none').lower()
    if num_epochs <= 0: return None

    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(num_epochs))
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_config.get('step_lr_step_size', 7),
                                         gamma=scheduler_config.get('step_lr_gamma', 0.1))
    return None


def run_experiment(exp_params: Dict[str, Any], base_fp32_model_state_dict: Optional[Dict] = None):
    """Runs a single, complete experiment pipeline."""
    exp_name = exp_params['name']
    device = exp_params['device']
    training_args = exp_params['training_args']
    compression_args = exp_params['compression_args']
    dataset_args = exp_params['dataset_args']
    model_args = exp_params['model_args']

    set_seed(training_args.get('global_seed', 42))

    print(f"\n{'=' * 60}\n🚀 Starting Experiment: {exp_name}\n{'=' * 60}")

    results_log = {'config': {k: v for k, v in exp_params.items() if k not in ['device']}, 'metrics': {}}
    model_save_prefix = os.path.join(exp_params['model_save_dir'], exp_name)

    # --- Data and Loss Setup ---
    trainloader, testloader, num_classes = get_dataloaders(
        dataset_args=dataset_args, training_args=training_args
    )
    criterion = nn.CrossEntropyLoss().to(device)

    # --- Model Initialization ---
    current_model = HybridCNNTransformer(
        num_classes=num_classes, model_args=model_args, image_size=dataset_args['image_size']
    ).to(device)

    # --- Stage 1: Initial FP32 Model (Load or Train) ---
    initial_train_epochs = training_args.get('initial_train_epochs', 0)
    if base_fp32_model_state_dict:
        print("\nLoading pre-trained FP32 baseline weights...")
        current_model.load_state_dict(base_fp32_model_state_dict)
        fp32_stage_label = "FP32_Loaded_Baseline"
    elif initial_train_epochs > 0:
        # This path is for training baselines from scratch
        print(f"\nTraining initial FP32 model for {initial_train_epochs} epochs...")
        optimizer = get_optimizer(current_model.parameters(), {'lr': training_args['base_lr'], **training_args})
        scheduler = get_scheduler(optimizer, training_args, initial_train_epochs)
        for epoch in range(initial_train_epochs):
            train_one_epoch(current_model, trainloader, criterion, optimizer, device, f"FP32 Train E{epoch + 1}",
                            scheduler)
        fp32_stage_label = "FP32_Initial_Train"
        if exp_params.get('is_fp32_baseline_generation'):
            torch.save(current_model.state_dict(), f"{model_save_prefix}_fp32_baseline.pth")
    else:
        fp32_stage_label = "FP32_Random_Weights"
        print("\nWarning: Model has random weights. No baseline loaded or initial training performed.")

    # Evaluate the FP32 model
    loss, acc, inf_time = evaluate(current_model, testloader, criterion, device, f"Eval {fp32_stage_label}")
    size, flops, params = get_model_stats(current_model, device=device, label=fp32_stage_label)
    results_log['metrics'][fp32_stage_label] = {'acc': acc, 'loss': loss, 'inf_time_ms': inf_time, 'size_mb': size,
                                                'gflops': flops, 'params_m': params}

    # --- Stage 2: Pruning and Finetuning ---
    pruning_amount = compression_args.get('pruning_amount', 0)
    if pruning_amount > 0:
        current_model = apply_channel_pruning(current_model, pruning_amount)
        # Evaluate after pruning, before fine-tuning
        loss_p, acc_p, inf_time_p = evaluate(current_model, testloader, criterion, device, "Eval Pruned")
        size_p, flops_p, params_p = get_model_stats(current_model, device=device, label="Pruned")
        results_log['metrics']["Pruned"] = {'acc': acc_p, 'loss': loss_p, 'inf_time_ms': inf_time_p, 'size_mb': size_p,
                                            'gflops': flops_p, 'params_m': params_p}

    finetune_epochs = training_args.get('finetune_epochs', 0)
    if finetune_epochs > 0 and pruning_amount > 0:
        print(f"\nFine-tuning pruned model for {finetune_epochs} epochs...")
        optimizer_ft = get_optimizer(current_model.parameters(), {'lr': training_args['lr_finetune'], **training_args})
        scheduler_ft = get_scheduler(optimizer_ft, training_args, finetune_epochs)
        for epoch in range(finetune_epochs):
            train_one_epoch(current_model, trainloader, criterion, optimizer_ft, device, f"Finetune E{epoch + 1}",
                            scheduler_ft)

        loss_ft, acc_ft, inf_time_ft = evaluate(current_model, testloader, criterion, device, "Eval Finetuned")
        size_ft, flops_ft, params_ft = get_model_stats(current_model, device=device, label="Finetuned")
        results_log['metrics']["Finetuned"] = {'acc': acc_ft, 'loss': loss_ft, 'inf_time_ms': inf_time_ft,
                                               'size_mb': size_ft, 'gflops': flops_ft, 'params_m': params_ft}

    # <--- 量化部分已禁用开始 ---
    # quantization_type = compression_args.get('quantization_type')
    # if quantization_type:
    #     print(f"\n--- Stage 3: Applying {quantization_type.upper()} Quantization ---")
    #
    #     if quantization_type == 'ptq':
    #         prepared_model = prepare_model_for_quantization(current_model, is_qat=False,
    #                                                         backend=compression_args['quantization_backend'])
    #
    #         # Use a calibration dataloader with no augmentations
    #         calib_loader, _, _ = get_dataloaders(dataset_args, training_args, for_calibration=True)
    #         calibrate_model_ptq(prepared_model, calib_loader, device, num_batches=compression_args['ptq_calib_batches'])
    #
    #         quantized_model = convert_to_quantized_model(prepared_model)
    #         stage_label = "PTQ_INT8"
    #
    #     elif quantization_type == 'qat':
    #         prepared_model = prepare_model_for_quantization(current_model, is_qat=True,
    #                                                         backend=compression_args['quantization_backend'])
    #         prepared_model.to(device)  # Move to training device
    #
    #         qat_epochs = training_args.get('qat_epochs', 0)
    #         print(f"\nRunning QAT for {qat_epochs} epochs...")
    #         optimizer_qat = get_optimizer(prepared_model.parameters(), {'lr': training_args['lr_qat'], **training_args})
    #         scheduler_qat = get_scheduler(optimizer_qat, training_args, qat_epochs)
    #         for epoch in range(qat_epochs):
    #             train_one_epoch(prepared_model, trainloader, criterion, optimizer_qat, device, f"QAT E{epoch + 1}",
    #                             scheduler_qat)
    #
    #         quantized_model = convert_to_quantized_model(prepared_model)
    #         stage_label = "QAT_INT8"
    #
    #     else:
    #         quantized_model = None
    #         stage_label = "Unknown_Quantization"
    #
    #     if quantized_model:
    #         # Evaluate quantized model on CPU
    #         quant_device = torch.device('cpu')
    #         quantized_model.to(quant_device)
    #         loss_q, acc_q, inf_time_q = evaluate(quantized_model, testloader, criterion, quant_device,
    #                                              f"Eval {stage_label}")
    #         size_q, flops_q, params_q = get_model_stats(quantized_model, device=quant_device, label=stage_label)
    #         results_log['metrics'][stage_label] = {'acc': acc_q, 'loss': loss_q, 'inf_time_ms': inf_time_q,
    #                                                'size_mb': size_q, 'gflops': flops_q, 'params_m': params_q}
    #         torch.save(quantized_model.state_dict(), f"{model_save_prefix}_{stage_label}.pth")
    # <--- 量化部分已禁用结束 ---

    # --- Save Results ---
    results_file_path = os.path.join(exp_params['results_dir'], f"{exp_name}_results.json")
    with open(results_file_path, 'w') as f:
        json.dump(results_log, f, indent=4)
    print(f"\n✅ Experiment {exp_name} Finished. Results saved to {results_file_path}")
    return results_log


def main(args):
    """Main function to orchestrate all experiments."""
    cfg.ensure_dirs()
    print(f"Global device: {cfg.DEVICE}, Models saved to: {cfg.MODEL_SAVE_DIR}, Results to: {cfg.RESULTS_DIR}")

    all_results = {}
    baseline_state_dicts = {}

    # --- 1. Prepare all FP32 baseline models ---
    print("\n--- Preparing FP32 Baselines ---")
    for exp_conf in cfg.EXPERIMENT_CONFIGS:
        if exp_conf.get('is_fp32_baseline_generation'):
            exp_name = exp_conf['name']
            path = os.path.join(cfg.MODEL_SAVE_DIR, f"{exp_name}_fp32_baseline.pth")
            if os.path.exists(path) and not args.retrain_baselines:
                print(f"Loading existing baseline: {exp_name}")
                baseline_state_dicts[exp_name] = torch.load(path, map_location='cpu')
            else:
                print(f"Training new baseline: {exp_name}")
                params = cfg.get_experiment_params(exp_name)
                results = run_experiment(params)
                all_results[exp_name] = results
                # Load the newly saved state_dict
                if os.path.exists(path):
                    baseline_state_dicts[exp_name] = torch.load(path, map_location='cpu')

    # --- 2. Run all dependent experiments ---
    print("\n--- Running All Dependent Experiments ---")
    for exp_conf in cfg.EXPERIMENT_CONFIGS:
        if not exp_conf.get('is_fp32_baseline_generation'):
            params = cfg.get_experiment_params(exp_conf['name'])
            baseline_name = params.get('load_fp32_baseline_from_experiment')

            state_dict = baseline_state_dicts.get(baseline_name)
            if baseline_name and not state_dict:
                print(
                    f"CRITICAL WARNING: Baseline '{baseline_name}' not found for experiment '{exp_conf['name']}'. Skipping.")
                continue

            results = run_experiment(params, base_fp32_model_state_dict=state_dict)
            all_results[exp_conf['name']] = results

    # --- 3. Save summary ---
    summary_path = os.path.join(cfg.RESULTS_DIR, "_ALL_EXPERIMENTS_SUMMARY.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n\nAll experiments finished. Full summary saved to {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run model compression experiments.")
    parser.add_argument('--retrain-baselines', action='store_true',
                        help="Force retraining of all baseline models, even if they exist.")
    args = parser.parse_args()

    set_seed(cfg.DEFAULT_TRAINING_PARAMS.get('global_seed', 42))
    main(args)