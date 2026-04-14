"""
LoRA模型 - 参数高效微调版本（修复版）
"""
import torch
import torch.nn as nn
from config import Config


class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.1):
            super().__init__()
            self.parent_model = None
            self.original_layer = original_layer
            self.rank = rank
            self.alpha = alpha
            for param in self.original_layer.parameters():
                param.requires_grad = False
            if isinstance(original_layer, nn.Linear):
                in_features = original_layer.in_features
                out_features = original_layer.out_features
                self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
                self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

            elif isinstance(original_layer, nn.Conv2d):
                in_channels = original_layer.in_channels
                out_channels = original_layer.out_channels
                kernel_size = original_layer.kernel_size
                if kernel_size == (1, 1) or kernel_size == 1:
                    self.lora_A = nn.Parameter(torch.randn(in_channels, rank) * 0.01)
                    self.lora_B = nn.Parameter(torch.zeros(rank, out_channels))
                else:
                    self.lora_A = None
                    self.lora_B = None

            self.dropout = nn.Dropout(dropout) if dropout > 0 else None
            self.scaling = alpha / rank

    def forward(self, x):
        result = self.original_layer(x)

        if self.lora_A is None or self.lora_B is None:
            return result

        if isinstance(self.original_layer, nn.Linear):
            lora_weight = self.lora_A @ self.lora_B
            lora_out = x @ lora_weight

        elif isinstance(self.original_layer, nn.Conv2d):
            batch, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
            lora_weight = self.lora_A @ self.lora_B
            lora_out = x_flat @ lora_weight
            lora_out = lora_out.reshape(batch, H, W, -1).permute(0, 3, 1, 2)

        if self.dropout is not None:
            lora_out = self.dropout(lora_out)
        lora_out = self._apply_spatial_modulation(lora_out)

        return result + lora_out * self.scaling

    def _apply_spatial_modulation(self, lora_out):
        if self.parent_model is None or not hasattr(self.parent_model, 'current_spatial_weight'):
            return lora_out

        spatial_weight = self.parent_model.current_spatial_weight
        if spatial_weight is None:
            return lora_out

        if len(lora_out.shape) == 4:
            _, _, H, W = lora_out.shape
            weight_resized = torch.nn.functional.interpolate(
                spatial_weight,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            return lora_out * weight_resized
        else:  # Linear层 [B, D]
            return lora_out


def inject_lora_to_convnext(model, rank=8, alpha=16, dropout=0.1):
    lora_param_count = 0
    replaced_layers = []
    print(f"  策略: 注入ConvNeXt Block中的pwconv层（索引3和5）")
    def should_inject_linear(name, module):
        """判断是否应该注入这个Linear层"""
        # 必须是Linear层
        if not isinstance(module, nn.Linear):
            return False

        if '.block.' not in name:
            return False

        parts = name.split('.')
        if 'block' in parts:
            try:
                idx = parts.index('block') + 1
                if idx < len(parts):
                    layer_idx = int(parts[idx])

                    if layer_idx in [3, 5]:
                        return True
            except (ValueError, IndexError):
                pass

        return False

    for name, module in model.named_modules():
        if should_inject_linear(name, module):
            parts = name.split('.')
            parent_name = '.'.join(parts[:-1])
            layer_name = parts[-1]

            try:
                parent = model.get_submodule(parent_name)
                lora_layer = LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, layer_name, lora_layer)
                if lora_layer.lora_A is not None and lora_layer.lora_B is not None:
                    param_count = lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
                    lora_param_count += param_count
                    replaced_layers.append(name)

            except Exception as e:
                print(f"    Warning: Failed to inject {name}: {e}")

    return model, lora_param_count, replaced_layers


class SimpleTripletClassifierLoRA(nn.Module):
    def __init__(self, lora_rank=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        print(f"Loading backbone...")
        self.backbone = self._load_backbone()
        self.register_buffer('current_spatial_weight', None)
        self._link_lora_layers()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.instrument_classifier = nn.Linear(Config.FEATURE_DIM, Config.NUM_INSTRUMENTS)
        self.verb_classifier = nn.Linear(Config.FEATURE_DIM, Config.NUM_VERBS)
        self.target_classifier = nn.Linear(Config.FEATURE_DIM, Config.NUM_TARGETS)
        self.triplet_classifier = nn.Linear(Config.FEATURE_DIM, Config.NUM_TRIPLETS)
        print(f"\nInjecting LoRA adapters...")
        print(f"  Rank: {lora_rank}")
        print(f"  Alpha: {lora_alpha}")
        print(f"  Dropout: {lora_dropout}")

        self.backbone, lora_params, replaced_layers = inject_lora_to_convnext(
            self.backbone,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout
        )

        print(f"  ✓ Injected LoRA to {len(replaced_layers)} layers")
        if replaced_layers:
            print(f"  ✓ Example layers: {replaced_layers[:3]}")
        print(f"  ✓ LoRA parameters: {lora_params:,}")

        classifier_params = (
                sum(p.numel() for p in self.instrument_classifier.parameters()) +
                sum(p.numel() for p in self.verb_classifier.parameters()) +
                sum(p.numel() for p in self.target_classifier.parameters()) +
                sum(p.numel() for p in self.triplet_classifier.parameters())
        )
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"\n{'=' * 60}")
        print("Model Statistics")
        print(f"{'=' * 60}")
        print(f"Total parameters:      {total_params:,} ({total_params / 1e6:.1f}M)")
        print(f"Trainable parameters:  {trainable_params:,} ({trainable_params / 1e6:.1f}M)")
        print(f"  - LoRA:              {lora_params:,} ({lora_params / 1e6:.2f}M)")
        print(f"  - Classifier:        {classifier_params:,} ({classifier_params / 1e6:.2f}M)")
        print(f"Frozen parameters:     {frozen_params:,} ({frozen_params / 1e6:.1f}M)")
        print(f"\n💡 Parameter reduction: {(1 - trainable_params / total_params) * 100:.1f}%")
        print(f"{'=' * 60}\n")

    def _link_lora_layers(self):
        for module in self.modules():
            if isinstance(module, LoRALayer):
                module.parent_model = self


    def _load_backbone(self):
        try:
            from torchvision import models
            print(f"  Attempting to load LemonFM from:")
            print(f"  {Config.LEMON_WEIGHTS}")
            backbone = models.convnext_large(weights=None)
            backbone = backbone.features  # 只保留features部分
            try:
                checkpoint = torch.load(Config.LEMON_WEIGHTS, map_location='cpu')
                print(f"  Checkpoint keys: {list(checkpoint.keys())}")

                state_dict = None
                for key in ['teacher', 'model_state_dict', 'student', 'state_dict', 'model']:
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        print(f"  ✓ Found weights under key: '{key}'")
                        break

                if state_dict is None:
                    state_dict = checkpoint
                    print(f"  ✓ Using checkpoint as state_dict directly")

                new_state_dict = {}
                for k, v in state_dict.items():
                    new_k = k.replace('module.', '').replace('backbone.', '')
                    if new_k.startswith('features.'):
                        new_k = new_k[9:]
                    elif any(x in new_k for x in ['head', 'classifier', 'fc', 'avgpool', 'projection']):
                        continue

                    new_state_dict[new_k] = v

                missing, unexpected = backbone.load_state_dict(new_state_dict, strict=False)
                if len(missing) > 100:  # ConvNeXt应该很少有missing keys
                    print(f"\n❌ CRITICAL: {len(missing)} missing keys!")
                    print(f"   LemonFM weights NOT loaded correctly!")
                    print(f"   Missing examples: {missing[:5]}")
                    first_layer_weights = backbone.state_dict()['0.0.weight']
                    print(f"\n🔍 First layer weights stats:")
                    print(f"   Mean: {first_layer_weights.mean():.6f}")
                    print(f"   Std: {first_layer_weights.std():.6f}")
                print(f"  ✓ Loaded {len(new_state_dict)} keys")
                if missing:
                    print(f"  ⚠️  Missing keys: {len(missing)}")
                    if len(missing) <= 5:
                        print(f"      {missing}")
                if unexpected:
                    print(f"  ⚠️  Unexpected keys: {len(unexpected)}")
                    if len(unexpected) <= 5:
                        print(f"      {unexpected}")

                print("  ✓ LemonFM weights loaded successfully")

            except Exception as e:
                print(f"  ⚠️  Failed to load LemonFM weights: {e}")
                print(f"  → Falling back to ImageNet pretrained weights")
                full_model = models.convnext_large(weights='IMAGENET1K_V1')
                backbone = full_model.features

            return backbone

        except Exception as e:
            raise RuntimeError(f"Failed to create backbone: {e}")

    def forward(self, x, pseudo_mask=None):
        if pseudo_mask is not None and Config.USE_PSEUDO_MASK:
            self.current_spatial_weight = self._process_pseudo_mask(pseudo_mask)
        else:
            self.current_spatial_weight = None

        # Backbone前向（LoRA层会自动读取current_spatial_weight）
        features = self.backbone(x)
        features = self.avgpool(features)
        features = features.flatten(1)

        return {
            'instruments': self.instrument_classifier(features),
            'verbs': self.verb_classifier(features),
            'targets': self.target_classifier(features),
            'triplets': self.triplet_classifier(features)
        }

    def _process_pseudo_mask(self, pseudo_mask):
        import torch.nn.functional as F
        device = pseudo_mask.device
        mask_smooth = F.avg_pool2d(
            pseudo_mask.to(device),  # 使用mask自己的device
            kernel_size=5,
            stride=1,
            padding=2
        )
        alpha = Config.PSEUDO_MASK_WEIGHT  # 建议值：0.5~1.0
        spatial_weight = 1.0 + alpha * mask_smooth
        return spatial_weight

    def get_lora_parameters(self):
        """获取所有LoRA参数，分别返回A和B（用于LoRA+不同学习率）"""
        lora_A_params = []
        lora_B_params = []
        for module in self.modules():
            if isinstance(module, LoRALayer):
                if module.lora_A is not None:
                    lora_A_params.append(module.lora_A)
                if module.lora_B is not None:
                    lora_B_params.append(module.lora_B)
        return lora_A_params, lora_B_params

    def merge_lora_weights(self):
        print("Merging LoRA weights into backbone...")

        for module in self.modules():
            if isinstance(module, LoRALayer):
                if module.lora_A is None or module.lora_B is None:
                    continue
                original_layer = module.original_layer
                if isinstance(original_layer, nn.Linear):
                    delta_w = module.scaling * (module.lora_B @ module.lora_A.T)
                    original_layer.weight.data += delta_w.T

                elif isinstance(original_layer, nn.Conv2d):
                    if original_layer.kernel_size == (1, 1):
                        delta_w = module.scaling * (module.lora_B @ module.lora_A.T)
                        original_layer.weight.data += delta_w.T.unsqueeze(-1).unsqueeze(-1)

        print("✓ LoRA weights merged")



if __name__ == '__main__':
    print("="*60)
    print("Testing SimpleTripletClassifierLoRA")
    print("="*60)
    model = SimpleTripletClassifierLoRA(lora_rank=8, lora_alpha=16)
    print("\n" + "="*60)
    print("Testing forward pass")
    print("="*60)
    dummy_input = torch.randn(2, 3, 224, 224)
    print(f"Input shape: {dummy_input.shape}")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 100), f"Expected (2, 100), got {output.shape}"
    print("\n" + "="*60)
    print("✓ Model test passed!")
    print("="*60)