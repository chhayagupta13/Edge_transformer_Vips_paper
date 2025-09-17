# model.py
import torch
import torch.nn as nn
import math
from torch.ao.quantization import QuantStub, DeQuantStub
from typing import Tuple, Dict, Any, Optional, List

# 从 config.py 导入全局配置
try:
    import config as cfg
except ImportError:
    print("Warning: config.py not found directly. Using local defaults if model_args not provided.")


    class cfg:
        DEFAULT_MODEL_PARAMS = {
            'cnn_output_channels': 128,
            'cnn_block_configs': [
                {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool_kernel': 2, 'pool_stride': 2},
                {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool_kernel': 2, 'pool_stride': 2},
                {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': False},
            ],
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


class LightweightCNN(nn.Module):
    """
    可配置的轻量级CNN骨干网络。
    所有Conv层都不带bias，以便与BatchNorm融合。
    """

    def __init__(self,
                 input_channels: int = 3,
                 block_configs: Optional[List[Dict[str, Any]]] = None,
                 image_size: int = 32,
                 final_output_channels: Optional[int] = None):
        super(LightweightCNN, self).__init__()
        self.input_channels = input_channels

        if block_configs is None:
            block_configs = [
                {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool_kernel': 2, 'pool_stride': 2},
                {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool_kernel': 2, 'pool_stride': 2},
                {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': False},
            ]

        self.features = nn.Sequential()
        current_channels = input_channels
        current_h, current_w = image_size, image_size

        for i, block_conf in enumerate(block_configs):
            out_c = block_conf['out_channels']
            if i == len(block_configs) - 1 and final_output_channels is not None:
                out_c = final_output_channels

            self.features.add_module(f"conv_block_{i}_conv", nn.Conv2d(
                current_channels, out_c,
                kernel_size=block_conf.get('kernel_size', 3),
                stride=block_conf.get('stride', 1),
                padding=block_conf.get('padding', 1),
                bias=False
            ))
            self.features.add_module(f"conv_block_{i}_bn", nn.BatchNorm2d(out_c))
            self.features.add_module(f"conv_block_{i}_relu", nn.ReLU(inplace=True))

            if block_conf.get('pool', True) and block_conf.get('pool_kernel') and block_conf.get('pool_stride'):
                self.features.add_module(f"conv_block_{i}_pool", nn.MaxPool2d(
                    kernel_size=block_conf['pool_kernel'],
                    stride=block_conf['pool_stride']
                ))
                # Correctly calculate output size after pooling
                current_h = math.floor((current_h + 2 * 0 - block_conf['pool_kernel']) / block_conf['pool_stride']) + 1
                current_w = math.floor((current_w + 2 * 0 - block_conf['pool_kernel']) / block_conf['pool_stride']) + 1

            current_channels = out_c

        self.output_channels = current_channels
        self.feature_map_size_h: int = int(current_h)
        self.feature_map_size_w: int = int(current_w)
        self.num_patches: int = self.feature_map_size_h * self.feature_map_size_w
        if self.num_patches <= 0:
            print(
                f"Warning: Calculated num_patches is {self.num_patches} for LightweightCNN. Image size: {image_size}, Final H/W: {current_h}/{current_w}. Check block_configs and pooling.")

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, learnable: bool = True, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.learnable = learnable
        self.dropout = nn.Dropout(p=dropout)

        # Ensure max_len is at least 1 for parameter creation
        actual_max_len = max(1, max_len)

        if learnable:
            self.pos_embedding = nn.Parameter(torch.randn(1, actual_max_len, d_model))
            nn.init.normal_(self.pos_embedding, std=0.02)
        else:
            # Standard fixed positional encoding
            position = torch.arange(actual_max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe = torch.zeros(1, actual_max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if self.learnable:
            pos_embedding_slice = self.pos_embedding[:, :seq_len, :]
        else:
            pos_embedding_slice = self.pe[:, :seq_len, :]

        x = x + pos_embedding_slice
        return self.dropout(x)


class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes: int, model_args: Dict[str, Any], image_size: int = 32):
        super(HybridCNNTransformer, self).__init__()

        self.fusion_strategy = model_args.get('fusion_strategy', 'serial').lower()

        cnn_block_configs = model_args.get('cnn_block_configs')
        final_cnn_out_channels = model_args.get('cnn_output_channels')

        self.cnn_backbone = LightweightCNN(
            input_channels=3,
            block_configs=cnn_block_configs,
            image_size=image_size,
            final_output_channels=final_cnn_out_channels
        )
        self.transformer_input_dim = self.cnn_backbone.output_channels

        transformer_layers = model_args.get('transformer_layers')
        transformer_heads = model_args.get('transformer_heads')
        transformer_dim_feedforward = model_args.get('transformer_dim_feedforward')
        self.use_pos_encoding = model_args.get('use_pos_encoding')
        transformer_dropout = model_args.get('transformer_dropout')

        if self.fusion_strategy == 'serial':
            self.num_patches = self.cnn_backbone.num_patches
            if self.num_patches <= 0:
                raise ValueError(
                    f"Calculated num_patches is {self.num_patches} from CNN. Check CNN config and image size.")

            if self.use_pos_encoding:
                self.pos_encoder = PositionalEncoding(
                    d_model=self.transformer_input_dim,
                    max_len=self.num_patches + 1,  # +1 for CLS token
                    learnable=model_args.get('pos_encoding_learnable', True),
                    dropout=transformer_dropout
                )

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.transformer_input_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_dim_feedforward,
                dropout=transformer_dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
                bias=False
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=transformer_layers,
                norm=nn.LayerNorm(self.transformer_input_dim)
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.transformer_input_dim))
            self.fc = nn.Linear(self.transformer_input_dim, num_classes)

        elif self.fusion_strategy == 'parallel':
            self.cnn_global_pool = nn.AdaptiveAvgPool2d((1, 1))

            mlp_input_channels = self.cnn_backbone.output_channels
            mlp_hidden_channels = model_args.get('parallel_mlp_hidden_channels')
            mlp_output_channels = model_args.get('parallel_mlp_output_channels')

            self.cnn_mlp_path = nn.Sequential(
                nn.Linear(mlp_input_channels, mlp_hidden_channels),
                nn.ReLU(inplace=True),  # ReLU is better for fusion than GELU
                nn.Dropout(transformer_dropout),
                nn.Linear(mlp_hidden_channels, mlp_output_channels)
            )
            self.fc = nn.Linear(mlp_input_channels + mlp_output_channels, num_classes)
        else:
            raise ValueError(f"Unsupported fusion_strategy: {self.fusion_strategy}.")

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self._initialize_weights(model_args)

    def _initialize_weights(self, model_args: Dict[str, Any]):
        if hasattr(self, 'cls_token'):
            nn.init.normal_(self.cls_token, std=model_args.get('cls_token_init_std', 0.02))
        if hasattr(self, 'fc'):
            nn.init.xavier_uniform_(self.fc.weight)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias, 0)

    def _fuse_modules_for_quantization(self, is_qat: bool = False):
        """ Fuses modules for quantization (PTQ or QAT). """
        print(f"Attempting to fuse modules (is_qat={is_qat})...")
        fuse_fn = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules

        # 1. Fuse CNN backbone (Conv-BN-ReLU)
        if hasattr(self.cnn_backbone, 'features'):
            # This logic correctly identifies sequences of [Conv, BN, ReLU] or [Conv, BN]
            module_list = list(self.cnn_backbone.features.named_children())
            for i in range(len(module_list) - 1):
                # Fuse Conv-BN
                if isinstance(module_list[i][1], nn.Conv2d) and isinstance(module_list[i + 1][1], nn.BatchNorm2d):
                    # Check for optional subsequent ReLU
                    if i + 2 < len(module_list) and isinstance(module_list[i + 2][1], nn.ReLU):
                        group_to_fuse = [module_list[i][0], module_list[i + 1][0], module_list[i + 2][0]]
                        print(f"  Fusing CNN group: {group_to_fuse}")
                        fuse_fn(self.cnn_backbone.features, group_to_fuse, inplace=True)
                    else:
                        group_to_fuse = [module_list[i][0], module_list[i + 1][0]]
                        print(f"  Fusing CNN group: {group_to_fuse}")
                        fuse_fn(self.cnn_backbone.features, group_to_fuse, inplace=True)

        # 2. Fuse Parallel MLP Path (Linear-ReLU)
        if self.fusion_strategy == 'parallel' and hasattr(self, 'cnn_mlp_path'):
            print("  Fusing Parallel MLP Path modules...")
            try:
                # The structure is Linear -> ReLU -> Dropout -> Linear. We can fuse the first Linear-ReLU.
                fuse_fn(self.cnn_mlp_path, ['0', '1'], inplace=True)
                print("    Successfully fused ['0', '1'] (Linear-ReLU) in Parallel MLP Path.")
            except Exception as e:
                print(f"    ERROR fusing Parallel MLP Path: {e}")

        # <--- 关键修改开始 ---
        # 3. Skip quantization for the TransformerEncoder module
        if self.fusion_strategy == 'serial':
            print("  Skipping quantization for TransformerEncoder module due to compatibility issues.")
            if hasattr(self, 'transformer_encoder'):
                self.transformer_encoder.qconfig = None
        # <--- 关键修改结束 ---

        print("Fusion process complete.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        cnn_out = self.cnn_backbone(x)

        if self.fusion_strategy == 'serial':
            # <--- 关键修改开始 ---
            # Dequantize the output of the CNN before mixing it with float tensors (CLS token).
            # This is the correct way to handle boundaries between quantized and non-quantized modules.
            if cnn_out.is_quantized:
                cnn_out_fp32 = torch.dequantize(cnn_out)
            else:
                cnn_out_fp32 = cnn_out

            # Now all operations are in floating point.
            batch_size = cnn_out_fp32.shape[0]
            transformer_seq = cnn_out_fp32.flatten(2).permute(0, 2, 1)

            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            transformer_seq_with_cls = torch.cat((cls_tokens, transformer_seq), dim=1)

            if self.use_pos_encoding:
                transformer_input = self.pos_encoder(transformer_seq_with_cls)
            else:
                transformer_input = transformer_seq_with_cls

            # The Transformer Encoder will run in floating point as its qconfig was set to None.
            transformer_output = self.transformer_encoder(transformer_input)

            cls_token_representation = transformer_output[:, 0, :]
            logits = self.fc(cls_token_representation)
            # <--- 关键修改结束 ---

        elif self.fusion_strategy == 'parallel':
            cnn_pooled_features = self.cnn_global_pool(cnn_out)

            # Dequantize before the MLP and the final concatenation
            if cnn_pooled_features.is_quantized:
                cnn_pooled_features_fp32 = torch.dequantize(cnn_pooled_features).flatten(1)
            else:
                cnn_pooled_features_fp32 = cnn_pooled_features.flatten(1)

            processed_cnn_features = self.cnn_mlp_path(cnn_pooled_features_fp32)

            combined_features = torch.cat((cnn_pooled_features_fp32, processed_cnn_features), dim=1)
            logits = self.fc(combined_features)
        else:
            raise RuntimeError(f"Invalid fusion strategy '{self.fusion_strategy}'.")

        logits = self.dequant(logits)
        return logits


if __name__ == '__main__':
    print("Testing model.py...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use hardcoded mock args for standalone testing
    default_params_mock = {
        'cnn_output_channels': 128, 'transformer_layers': 2, 'transformer_heads': 4,
        'transformer_dim_feedforward': 256, 'use_pos_encoding': True, 'transformer_dropout': 0.1,
        'pos_encoding_learnable': True, 'cls_token_init_std': 0.02,
        'parallel_mlp_hidden_channels': 64, 'parallel_mlp_output_channels': 32,
    }
    test_cnn_block_configs = [
        {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool_kernel': 2, 'pool_stride': 2},  # 32->16
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool_kernel': 2, 'pool_stride': 2},  # 16->8
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': False}  # 8->8
    ]
    mock_model_args_serial = {**default_params_mock, 'fusion_strategy': 'serial',
                              'cnn_block_configs': test_cnn_block_configs, 'cnn_output_channels': 64,
                              'transformer_layers': 1, 'transformer_heads': 2, 'transformer_dim_feedforward': 128,
                              'pos_encoding_learnable': True}
    mock_model_args_parallel = {**default_params_mock, 'fusion_strategy': 'parallel',
                                'cnn_block_configs': test_cnn_block_configs, 'cnn_output_channels': 64,
                                'parallel_mlp_hidden_channels': 32, 'parallel_mlp_output_channels': 16}

    num_classes_test = 10
    image_size_test = 32
    batch_size_test = 4

    print("\n--- Testing Serial Fusion Model ---")
    serial_model = HybridCNNTransformer(num_classes_test, mock_model_args_serial, image_size_test).to(device)
    serial_model.eval()
    dummy_input = torch.randn(batch_size_test, 3, image_size_test, image_size_test).to(device)
    output_serial = serial_model(dummy_input)
    print(f"Serial model output shape: {output_serial.shape}")
    assert output_serial.shape == (batch_size_test, num_classes_test)

    print("\n--- Testing Parallel Fusion Model ---")
    parallel_model = HybridCNNTransformer(num_classes_test, mock_model_args_parallel, image_size_test).to(device)
    parallel_model.eval()
    output_parallel = parallel_model(dummy_input)
    print(f"Parallel model output shape: {output_parallel.shape}")
    assert output_parallel.shape == (batch_size_test, num_classes_test)

    print("\n--- Testing Fusion Logic (PTQ Mode) ---")
    serial_model_to_fuse_ptq = HybridCNNTransformer(num_classes_test, mock_model_args_serial, image_size_test)
    serial_model_to_fuse_ptq.eval()
    serial_model_to_fuse_ptq._fuse_modules_for_quantization(is_qat=False)
    assert serial_model_to_fuse_ptq.transformer_encoder.qconfig is None  # Verify transformer is skipped
    print("  Fusion test passed for PTQ.")

    print("\nModel.py tests finished.")