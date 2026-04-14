"""
时序三元组LoRA (Temporal Triplet LoRA, TT-LoRA)

核心创新：
1. 空间LoRA - 逐帧提取空间特征（器械定位、目标识别）
2. 时序LoRA - 跨帧建模运动模式（动作识别）
3. 解耦设计 - 不同组件使用不同的适配策略

为什么需要时序LoRA？
- 器械(Instrument): 静态外观 → 空间特征足够
- 目标(Target): 静态解剖结构 → 空间特征足够
- 动作(Verb): 需要运动信息 → 必须时序建模！
  例如: "dissect" vs "retract" 在单帧中器械可能看起来一样，
       但运动方向/模式完全不同
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from config import Config


# ==================== 时序LoRA核心模块 ====================

class TemporalLoRALayer(nn.Module):
    """
    时序LoRA层 - 在时间维度上进行低秩适配

    不同于空间LoRA处理单帧，时序LoRA处理帧序列
    """

    def __init__(self, in_features, out_features, seq_len=4,
                 rank=8, alpha=16, dropout=0.1):
        super().__init__()

        self.seq_len = seq_len
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 时序LoRA的A和B矩阵
        # A: [seq_len * in_features, rank] - 将时序展开的特征映射到低秩空间
        # B: [rank, out_features] - 从低秩空间映射回输出
        self.lora_A = nn.Parameter(torch.randn(seq_len * in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, features]
        Returns:
            out: [batch, features] - 时序聚合后的特征
        """
        batch_size = x.size(0)

        # 展平时序维度: [batch, seq_len, features] -> [batch, seq_len * features]
        x_flat = x.reshape(batch_size, -1)

        # LoRA前向: x @ A @ B
        lora_out = x_flat @ self.lora_A @ self.lora_B
        lora_out = self.dropout(lora_out)

        return lora_out * self.scaling


class TemporalAttentionLoRA(nn.Module):
    """
    基于注意力的时序LoRA - 更灵活的时序建模

    使用自注意力机制学习帧间关系，然后用LoRA进行高效适配
    """

    def __init__(self, dim, seq_len=4, num_heads=4, rank=8, alpha=16, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 原始的Q/K/V投影（冻结）
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

        # LoRA适配Q和V（K保持不变，参考论文经验）
        self.lora_q_A = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.lora_q_B = nn.Parameter(torch.zeros(rank, dim))
        self.lora_v_A = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.lora_v_B = nn.Parameter(torch.zeros(rank, dim))

        # 时序位置编码（可学习）
        self.temporal_pos = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)

        # 输出投影
        self.proj = nn.Linear(dim, dim)
        self.lora_proj_A = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.lora_proj_B = nn.Parameter(torch.zeros(rank, dim))

        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            out: [batch, dim] - 时序聚合后的特征
        """
        B, T, D = x.shape

        # 添加时序位置编码
        x = x + self.temporal_pos[:, :T, :]

        # 原始QKV
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # LoRA增量
        delta_q = (x @ self.lora_q_A @ self.lora_q_B).reshape(B, T, self.num_heads, self.head_dim)
        delta_q = delta_q.permute(0, 2, 1, 3) * self.scaling
        delta_v = (x @ self.lora_v_A @ self.lora_v_B).reshape(B, T, self.num_heads, self.head_dim)
        delta_v = delta_v.permute(0, 2, 1, 3) * self.scaling

        q = q + delta_q
        v = v + delta_v

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # 加权聚合
        out = attn @ v  # [B, heads, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, D)  # [B, T, D]

        # 输出投影 + LoRA
        out = self.proj(out) + (out @ self.lora_proj_A @ self.lora_proj_B) * self.scaling

        # 时序聚合：取最后一帧或平均
        out = out.mean(dim=1)  # [B, D]

        return out


class TemporalConvLoRA(nn.Module):
    """
    基于1D卷积的时序LoRA - 轻量级时序建模

    使用1D卷积捕捉局部时序模式，计算效率高
    """

    def __init__(self, dim, seq_len=4, kernel_size=3, rank=8, alpha=16, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.seq_len = seq_len

        # 原始时序卷积（冻结）
        self.temporal_conv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim  # Depthwise，减少参数
        )

        # LoRA适配：用1x1卷积实现
        # 等价于对每个时间步的特征做线性变换
        self.lora_A = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, dim))

        # 时序聚合
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            out: [batch, dim]
        """
        B, T, D = x.shape

        # [B, T, D] -> [B, D, T] for Conv1d
        x = x.permute(0, 2, 1)

        # 时序卷积
        out = self.temporal_conv(x)  # [B, D, T]

        # LoRA增量
        # [B, D, T] -> [B, T, D] -> LoRA -> [B, T, D] -> [B, D, T]
        x_for_lora = x.permute(0, 2, 1)  # [B, T, D]
        delta = (x_for_lora @ self.lora_A @ self.lora_B) * self.scaling  # [B, T, D]
        delta = delta.permute(0, 2, 1)  # [B, D, T]

        out = out + delta
        out = self.dropout(out)

        # 时序聚合
        out = self.temporal_pool(out).squeeze(-1)  # [B, D]

        return out


# ==================== 完整的TT-LoRA模型 ====================

class TemporalTripletLoRA(nn.Module):
    """
    时序三元组LoRA (TT-LoRA) 完整模型

    架构:
    1. 空间LoRA骨干：处理每帧的空间特征
    2. 时序LoRA分支：专门处理动作(verb)的时序信息
    3. 解耦分类头：器械/目标用空间特征，动作用时序特征
    """

    def __init__(self,
                 spatial_lora_rank=8,
                 temporal_lora_rank=8,
                 seq_len=4,
                 temporal_type='attention',  # 'attention', 'conv', 'simple'
                 lora_alpha=16,
                 lora_dropout=0.1):
        super().__init__()

        self.seq_len = seq_len
        self.temporal_type = temporal_type

        print(f"\n{'=' * 60}")
        print("Initializing Temporal Triplet LoRA (TT-LoRA)")
        print(f"{'=' * 60}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Spatial LoRA rank: {spatial_lora_rank}")
        print(f"  Temporal LoRA rank: {temporal_lora_rank}")
        print(f"  Temporal type: {temporal_type}")

        # 1. 加载骨干网络并注入空间LoRA
        self.backbone = self._load_backbone()
        self.backbone, spatial_lora_params, _ = inject_lora_to_convnext(
            self.backbone,
            rank=spatial_lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout
        )
        print(f"  ✓ Spatial LoRA params: {spatial_lora_params:,}")

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 2. 时序LoRA模块（专门用于verb）
        if temporal_type == 'attention':
            self.temporal_lora = TemporalAttentionLoRA(
                dim=Config.FEATURE_DIM,
                seq_len=seq_len,
                num_heads=8,
                rank=temporal_lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout
            )
        elif temporal_type == 'conv':
            self.temporal_lora = TemporalConvLoRA(
                dim=Config.FEATURE_DIM,
                seq_len=seq_len,
                kernel_size=3,
                rank=temporal_lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout
            )
        else:  # simple
            self.temporal_lora = TemporalLoRALayer(
                in_features=Config.FEATURE_DIM,
                out_features=Config.FEATURE_DIM,
                seq_len=seq_len,
                rank=temporal_lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout
            )

        temporal_params = sum(p.numel() for p in self.temporal_lora.parameters())
        print(f"  ✓ Temporal LoRA params: {temporal_params:,}")

        # 3. 解耦分类头
        # 器械和目标：使用空间特征（当前帧）
        self.instrument_classifier = nn.Linear(Config.FEATURE_DIM, Config.NUM_INSTRUMENTS)
        self.target_classifier = nn.Linear(Config.FEATURE_DIM, Config.NUM_TARGETS)

        # 动作：使用时序特征
        self.verb_classifier = nn.Linear(Config.FEATURE_DIM, Config.NUM_VERBS)

        # Triplet：融合空间+时序特征
        self.triplet_fusion = nn.Sequential(
            nn.Linear(Config.FEATURE_DIM * 2, Config.FEATURE_DIM),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.triplet_classifier = nn.Linear(Config.FEATURE_DIM, Config.NUM_TRIPLETS)

        # 统计
        self._print_param_stats()

    def _load_backbone(self):
        """加载预训练骨干网络"""
        from torchvision import models

        try:
            backbone = models.convnext_large(weights=None)
            backbone = backbone.features

            checkpoint = torch.load(Config.LEMON_WEIGHTS, map_location='cpu')
            state_dict = checkpoint.get('teacher', checkpoint.get('model_state_dict', checkpoint))

            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('module.', '').replace('backbone.', '')
                if new_k.startswith('features.'):
                    new_k = new_k[9:]
                if not any(x in new_k for x in ['head', 'classifier', 'fc']):
                    new_state_dict[new_k] = v

            backbone.load_state_dict(new_state_dict, strict=False)
            print("  ✓ LemonFM weights loaded")

        except Exception as e:
            print(f"  ⚠ Failed to load LemonFM: {e}")
            full_model = models.convnext_large(weights='IMAGENET1K_V1')
            backbone = full_model.features
            print("  ✓ Using ImageNet pretrained weights")

        return backbone

    def _print_param_stats(self):
        """打印参数统计"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n{'=' * 60}")
        print("TT-LoRA Parameter Statistics")
        print(f"{'=' * 60}")
        print(f"  Total params: {total:,} ({total / 1e6:.1f}M)")
        print(f"  Trainable params: {trainable:,} ({trainable / 1e6:.2f}M)")
        print(f"  Trainable ratio: {trainable / total * 100:.2f}%")
        print(f"{'=' * 60}\n")

    def extract_spatial_features(self, x, mask=None):
        """
        提取单帧空间特征

        🔥 新增：可选mask空间加权，增强器械/目标区域

        Args:
            x: [B, 3, H, W] 输入图像
            mask: [B, 1, H, W] 伪标签mask（可选）

        Returns:
            features: [B, D] 空间特征
        """
        features = self.backbone(x)  # [B, C, H', W']

        # 🔥 伪标签空间加权
        if mask is not None and Config.USE_PSEUDO_MASK:
            # 下采样mask到特征图尺寸
            mask_down = F.interpolate(
                mask,
                size=features.shape[2:],  # [H', W']
                mode='nearest'
            )
            # 软加权：1 + weight * mask
            # 背景区域权重=1（保留），前景区域权重=1+weight（增强）
            weight = 1.0 + Config.PSEUDO_MASK_WEIGHT * mask_down
            features = features * weight

        features = self.avgpool(features)
        features = features.flatten(1)
        return features

    def forward(self, x, mask=None, return_features=False):
        """
        前向传播

        Args:
            x: 输入图像
               - 单帧模式: [batch, 3, H, W]
               - 序列模式: [batch, seq_len, 3, H, W]
            mask: 伪标签mask（可选）  # 🔥 新增
               - 单帧模式: [batch, 1, H, W]
               - 序列模式: [batch, seq_len, 1, H, W]
            return_features: 是否返回中间特征
        """
        if x.dim() == 4:
            return self._forward_single_frame(x, mask)
        else:
            return self._forward_sequence(x, mask, return_features)

    def _forward_single_frame(self, x, mask=None):
        """单帧前向（兼容模式）"""
        spatial_feat = self.extract_spatial_features(x, mask)  # 🔥 传入mask

        return {
            'instruments': self.instrument_classifier(spatial_feat),
            'verbs': self.verb_classifier(spatial_feat),
            'targets': self.target_classifier(spatial_feat),
            'triplets': self.triplet_classifier(spatial_feat)
        }

    def _forward_sequence(self, x, mask=None, return_features=False):
        """
        序列前向 - TT-LoRA的核心

        Args:
            x: [batch, seq_len, 3, H, W]
            mask: [batch, seq_len, 1, H, W] 伪标签mask（可选）  # 🔥 新增
        """
        B, T, C, H, W = x.shape

        # 1. 提取每帧的空间特征
        x_flat = x.view(B * T, C, H, W)

        # 🔥 处理序列mask
        if mask is not None:
            # mask: [B, T, 1, Hm, Wm] -> [B*T, 1, Hm, Wm]
            mask_flat = mask.view(B * T, 1, mask.shape[-2], mask.shape[-1])
        else:
            mask_flat = None

        spatial_feats = self.extract_spatial_features(x_flat, mask_flat)  # 🔥 传入mask
        spatial_feats = spatial_feats.view(B, T, -1)  # [B, T, D]

        # 2. 当前帧的空间特征（用于器械和目标）
        # 使用最后一帧作为当前帧
        current_spatial = spatial_feats[:, -1, :]  # [B, D]

        # 3. 时序特征（用于动作）
        temporal_feat = self.temporal_lora(spatial_feats)  # [B, D]

        # 4. 解耦分类
        # 器械和目标：基于当前帧空间特征
        inst_logits = self.instrument_classifier(current_spatial)
        target_logits = self.target_classifier(current_spatial)

        # 动作：基于时序特征
        verb_logits = self.verb_classifier(temporal_feat)

        # Triplet：融合空间+时序特征
        fused_feat = self.triplet_fusion(
            torch.cat([current_spatial, temporal_feat], dim=-1)
        )
        triplet_logits = self.triplet_classifier(fused_feat)

        outputs = {
            'instruments': inst_logits,
            'verbs': verb_logits,
            'targets': target_logits,
            'triplets': triplet_logits
        }

        if return_features:
            outputs['spatial_features'] = current_spatial
            outputs['temporal_features'] = temporal_feat
            outputs['fused_features'] = fused_feat

        return outputs

    def get_spatial_lora_parameters(self):
        """获取空间LoRA参数"""
        params = []
        for module in self.backbone.modules():
            if isinstance(module, LoRALayer):
                if module.lora_A is not None:
                    params.append(module.lora_A)
                if module.lora_B is not None:
                    params.append(module.lora_B)
        return params

    def get_temporal_lora_parameters(self):
        """获取时序LoRA参数"""
        return list(self.temporal_lora.parameters())

    def get_classifier_parameters(self):
        """获取分类头参数"""
        return (
                list(self.instrument_classifier.parameters()) +
                list(self.verb_classifier.parameters()) +
                list(self.target_classifier.parameters()) +
                list(self.triplet_fusion.parameters()) +
                list(self.triplet_classifier.parameters())
        )


# ==================== 空间LoRA注入（复用原有代码） ====================

class LoRALayer(nn.Module):
    """空间LoRA层（与原model_lora.py相同）"""

    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.1):
        super().__init__()
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
            if original_layer.kernel_size == (1, 1) or original_layer.kernel_size == 1:
                self.lora_A = nn.Parameter(torch.randn(in_channels, rank) * 0.01)
                self.lora_B = nn.Parameter(torch.zeros(rank, out_channels))
            else:
                self.lora_A = None
                self.lora_B = None
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
            lora_out = x @ self.lora_A @ self.lora_B
        elif isinstance(self.original_layer, nn.Conv2d):
            batch, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
            lora_out = x_flat @ self.lora_A @ self.lora_B
            lora_out = lora_out.reshape(batch, H, W, -1).permute(0, 3, 1, 2)

        if self.dropout is not None:
            lora_out = self.dropout(lora_out)

        return result + lora_out * self.scaling


def inject_lora_to_convnext(model, rank=8, alpha=16, dropout=0.1):
    """向ConvNeXt注入空间LoRA"""
    lora_param_count = 0
    replaced_layers = []

    def should_inject(name, module):
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
                    return layer_idx in [3, 5]
            except:
                pass
        return False

    for name, module in model.named_modules():
        if should_inject(name, module):
            parts = name.split('.')
            parent_name = '.'.join(parts[:-1])
            layer_name = parts[-1]

            try:
                parent = model.get_submodule(parent_name)
                lora_layer = LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, layer_name, lora_layer)

                if lora_layer.lora_A is not None:
                    lora_param_count += lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
                    replaced_layers.append(name)
            except Exception as e:
                print(f"  Warning: Failed to inject {name}: {e}")

    return model, lora_param_count, replaced_layers


# ==================== 序列数据集 ====================

class TemporalCholecT50Dataset(torch.utils.data.Dataset):
    """
    时序版本的CholecT50数据集

    返回连续T帧的序列，而不是单帧
    """

    def __init__(self, base_dataset, seq_len=4, stride=1):
        """
        Args:
            base_dataset: 原始CholecT50Dataset
            seq_len: 序列长度
            stride: 采样间隔（stride=1表示连续帧，stride=2表示隔帧采样）
        """
        self.base_dataset = base_dataset
        self.seq_len = seq_len
        self.stride = stride

        # 构建有效的序列索引
        self.valid_indices = self._build_valid_indices()
        print(f"  ✓ Temporal dataset: {len(self.valid_indices)} sequences "
              f"(seq_len={seq_len}, stride={stride})")

    def _build_valid_indices(self):
        """
        构建有效的序列起始索引

        确保：
        1. 序列中的帧来自同一个视频
        2. 序列长度足够
        """
        valid_indices = []
        samples = self.base_dataset.samples

        # 按视频分组
        video_groups = {}
        for idx, sample in enumerate(samples):
            video_id = sample['video_id']
            if video_id not in video_groups:
                video_groups[video_id] = []
            video_groups[video_id].append(idx)

        # 对每个视频，找出有效的序列起始点
        for video_id, indices in video_groups.items():
            # 确保索引是排序的（按帧号）
            indices = sorted(indices, key=lambda i: samples[i]['frame'])

            # 需要的帧数
            needed_frames = (self.seq_len - 1) * self.stride + 1

            for i in range(len(indices) - needed_frames + 1):
                valid_indices.append(indices[i])

        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Returns:
            images: [seq_len, 3, H, W]
            labels: [100] - 使用最后一帧的标签
            video_id: str
            masks: [seq_len, 1, H, W] - 🔥 新增
        """
        start_idx = self.valid_indices[idx]
        samples = self.base_dataset.samples

        images = []
        masks = []  # 🔥 新增
        labels = None
        video_id = samples[start_idx]['video_id']

        for t in range(self.seq_len):
            frame_idx = start_idx + t * self.stride

            # 🔥 获取单帧数据（现在返回4个元素）
            img, label, vid, mask = self.base_dataset[frame_idx]
            images.append(img)
            masks.append(mask)  # 🔥 收集mask

            if t == self.seq_len - 1:
                labels = label

        images = torch.stack(images, dim=0)  # [T, 3, H, W]
        masks = torch.stack(masks, dim=0)  # [T, 1, H, W] 🔥

        return images, labels, video_id, masks  # 🔥 返回4个元素


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Temporal Triplet LoRA (TT-LoRA)")
    print("=" * 60)

    # 测试不同的时序LoRA类型
    for temporal_type in ['simple', 'conv', 'attention']:
        print(f"\n--- Testing {temporal_type} temporal LoRA ---")

        model = TemporalTripletLoRA(
            spatial_lora_rank=8,
            temporal_lora_rank=8,
            seq_len=4,
            temporal_type=temporal_type
        )

        # 测试序列输入
        batch_size = 2
        seq_len = 4
        x_seq = torch.randn(batch_size, seq_len, 3, 256, 448)

        print(f"Input shape: {x_seq.shape}")

        with torch.no_grad():
            outputs = model(x_seq)

        print(f"Output shapes:")
        print(f"  instruments: {outputs['instruments'].shape}")
        print(f"  verbs: {outputs['verbs'].shape}")
        print(f"  targets: {outputs['targets'].shape}")
        print(f"  triplets: {outputs['triplets'].shape}")

        # 测试单帧输入（兼容模式）
        x_single = torch.randn(batch_size, 3, 256, 448)
        with torch.no_grad():
            outputs_single = model(x_single)
        print(f"Single frame mode: OK")

    print("\n" + "=" * 60)
    print("✓ All TT-LoRA tests passed!")
    print("=" * 60)