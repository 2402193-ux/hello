"""
PG-HCN 修复版 (v2) - 完整版
============================

修复内容:
1. TripletCombiner: 直接在logits空间操作（数值稳定）
2. PhaseEncoder: 添加先验平滑（处理27个零概率三元组）
3. constraint_penalty: 增大到15.0
4. ConstraintMaskGenerator: 可选LogSumExp近似max（可微）
5. 包含 ConstraintAwareLoss（之前缺失）
6. 添加兼容别名 PriorGuidedHCN = PriorGuidedHCNV2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, Optional
import weakref
from config import Config


# ==================== 配置 ====================

NUM_PHASES = 8  # 0-6 + unknown
PHASE_UNKNOWN = 7

# 100个Triplet定义
TRIPLET_CLASSES = [
    ('bipolar', 'coagulate', 'abdominal-wall/cavity'),
    ('bipolar', 'coagulate', 'blood-vessel'),
    ('bipolar', 'coagulate', 'cystic-artery'),
    ('bipolar', 'coagulate', 'cystic-duct'),
    ('bipolar', 'coagulate', 'cystic-pedicle'),
    ('bipolar', 'coagulate', 'cystic-plate'),
    ('bipolar', 'coagulate', 'gallbladder'),
    ('bipolar', 'coagulate', 'liver'),
    ('bipolar', 'coagulate', 'omentum'),
    ('bipolar', 'coagulate', 'peritoneum'),
    ('bipolar', 'dissect', 'adhesion'),
    ('bipolar', 'dissect', 'cystic-artery'),
    ('bipolar', 'dissect', 'cystic-duct'),
    ('bipolar', 'dissect', 'cystic-plate'),
    ('bipolar', 'dissect', 'gallbladder'),
    ('bipolar', 'dissect', 'omentum'),
    ('bipolar', 'grasp', 'cystic-plate'),
    ('bipolar', 'grasp', 'liver'),
    ('bipolar', 'grasp', 'specimen-bag'),
    ('bipolar', 'null-verb', 'null-target'),
    ('bipolar', 'retract', 'cystic-duct'),
    ('bipolar', 'retract', 'cystic-pedicle'),
    ('bipolar', 'retract', 'gallbladder'),
    ('bipolar', 'retract', 'liver'),
    ('bipolar', 'retract', 'omentum'),
    ('clipper', 'clip', 'blood-vessel'),
    ('clipper', 'clip', 'cystic-artery'),
    ('clipper', 'clip', 'cystic-duct'),
    ('clipper', 'clip', 'cystic-pedicle'),
    ('clipper', 'clip', 'cystic-plate'),
    ('clipper', 'null-verb', 'null-target'),
    ('grasper', 'dissect', 'cystic-plate'),
    ('grasper', 'dissect', 'gallbladder'),
    ('grasper', 'dissect', 'omentum'),
    ('grasper', 'grasp', 'cystic-artery'),
    ('grasper', 'grasp', 'cystic-duct'),
    ('grasper', 'grasp', 'cystic-pedicle'),
    ('grasper', 'grasp', 'cystic-plate'),
    ('grasper', 'grasp', 'gallbladder'),
    ('grasper', 'grasp', 'gut'),
    ('grasper', 'grasp', 'liver'),
    ('grasper', 'grasp', 'omentum'),
    ('grasper', 'grasp', 'peritoneum'),
    ('grasper', 'grasp', 'specimen-bag'),
    ('grasper', 'null-verb', 'null-target'),
    ('grasper', 'pack', 'gallbladder'),
    ('grasper', 'retract', 'cystic-duct'),
    ('grasper', 'retract', 'cystic-pedicle'),
    ('grasper', 'retract', 'cystic-plate'),
    ('grasper', 'retract', 'gallbladder'),
    ('grasper', 'retract', 'gut'),
    ('grasper', 'retract', 'liver'),
    ('grasper', 'retract', 'omentum'),
    ('grasper', 'retract', 'peritoneum'),
    ('hook', 'coagulate', 'blood-vessel'),
    ('hook', 'coagulate', 'cystic-artery'),
    ('hook', 'coagulate', 'cystic-duct'),
    ('hook', 'coagulate', 'cystic-pedicle'),
    ('hook', 'coagulate', 'cystic-plate'),
    ('hook', 'coagulate', 'gallbladder'),
    ('hook', 'coagulate', 'liver'),
    ('hook', 'coagulate', 'omentum'),
    ('hook', 'cut', 'blood-vessel'),
    ('hook', 'cut', 'peritoneum'),
    ('hook', 'dissect', 'blood-vessel'),
    ('hook', 'dissect', 'cystic-artery'),
    ('hook', 'dissect', 'cystic-duct'),
    ('hook', 'dissect', 'cystic-plate'),
    ('hook', 'dissect', 'gallbladder'),
    ('hook', 'dissect', 'omentum'),
    ('hook', 'dissect', 'peritoneum'),
    ('hook', 'null-verb', 'null-target'),
    ('hook', 'retract', 'gallbladder'),
    ('hook', 'retract', 'liver'),
    ('irrigator', 'aspirate', 'fluid'),
    ('irrigator', 'dissect', 'cystic-duct'),
    ('irrigator', 'dissect', 'cystic-pedicle'),
    ('irrigator', 'dissect', 'cystic-plate'),
    ('irrigator', 'dissect', 'gallbladder'),
    ('irrigator', 'dissect', 'omentum'),
    ('irrigator', 'irrigate', 'abdominal-wall/cavity'),
    ('irrigator', 'irrigate', 'cystic-pedicle'),
    ('irrigator', 'irrigate', 'liver'),
    ('irrigator', 'null-verb', 'null-target'),
    ('irrigator', 'retract', 'gallbladder'),
    ('irrigator', 'retract', 'liver'),
    ('irrigator', 'retract', 'omentum'),
    ('scissors', 'coagulate', 'omentum'),
    ('scissors', 'cut', 'adhesion'),
    ('scissors', 'cut', 'blood-vessel'),
    ('scissors', 'cut', 'cystic-artery'),
    ('scissors', 'cut', 'cystic-duct'),
    ('scissors', 'cut', 'cystic-plate'),
    ('scissors', 'cut', 'liver'),
    ('scissors', 'cut', 'omentum'),
    ('scissors', 'cut', 'peritoneum'),
    ('scissors', 'dissect', 'cystic-plate'),
    ('scissors', 'dissect', 'gallbladder'),
    ('scissors', 'dissect', 'omentum'),
    ('scissors', 'null-verb', 'null-target'),
]

INSTRUMENTS = ['bipolar', 'clipper', 'grasper', 'hook', 'irrigator', 'scissors']
VERBS = ['aspirate', 'clip', 'coagulate', 'cut', 'dissect', 'grasp', 'irrigate', 'null-verb', 'pack', 'retract']
TARGETS = ['abdominal-wall/cavity', 'adhesion', 'blood-vessel', 'cystic-artery',
           'cystic-duct', 'cystic-pedicle', 'cystic-plate', 'fluid', 'gallbladder',
           'gut', 'liver', 'null-target', 'omentum', 'peritoneum', 'specimen-bag']


# ==================== FiLM层 ====================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation"""
    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(condition_dim, feature_dim)
        self.beta_net = nn.Linear(condition_dim, feature_dim)

        nn.init.ones_(self.gamma_net.weight.data.mean(dim=1, keepdim=True))
        nn.init.zeros_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_net(condition)
        beta = self.beta_net(condition)
        return gamma * x + beta


# ==================== 约束Mask生成器 ====================

class ConstraintMaskGenerator(nn.Module):
    """
    约束Mask生成器（修复版v2）
    使用LogSumExp近似max（可微）
    """
    def __init__(
        self,
        iv_constraint: np.ndarray,
        vt_constraint: np.ndarray,
        it_constraint: np.ndarray,
        use_soft_max: bool = True,
        temperature: float = 10.0
    ):
        super().__init__()

        self.register_buffer('iv_mask', torch.from_numpy(iv_constraint).float())
        self.register_buffer('vt_mask', torch.from_numpy(vt_constraint).float())
        self.register_buffer('it_mask', torch.from_numpy(it_constraint).float())

        self.use_soft_max = use_soft_max
        self.temperature = temperature

    def _soft_max(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """LogSumExp近似max（可微）"""
        if self.use_soft_max:
            return torch.logsumexp(x * self.temperature, dim=dim) / self.temperature
        else:
            return x.max(dim=dim)[0]

    def get_verb_mask_given_instrument(self, instrument_probs: torch.Tensor) -> torch.Tensor:
        expanded = instrument_probs.unsqueeze(-1) * self.iv_mask.unsqueeze(0)
        verb_mask = self._soft_max(expanded, dim=1)
        return verb_mask.clamp(0, 1)

    def get_target_mask_given_iv(
        self,
        instrument_probs: torch.Tensor,
        verb_probs: torch.Tensor
    ) -> torch.Tensor:
        it_expanded = instrument_probs.unsqueeze(-1) * self.it_mask.unsqueeze(0)
        it_mask = self._soft_max(it_expanded, dim=1)

        vt_expanded = verb_probs.unsqueeze(-1) * self.vt_mask.unsqueeze(0)
        vt_mask = self._soft_max(vt_expanded, dim=1)

        target_mask = torch.min(it_mask, vt_mask)
        return target_mask.clamp(0, 1)


# ==================== Phase编码器 ====================

class PhaseEncoder(nn.Module):
    """Phase编码器（带先验平滑）"""
    def __init__(
        self,
        num_phases: int = NUM_PHASES,
        embed_dim: int = 64,
        phase_prior: Optional[np.ndarray] = None,
        prior_smoothing: float = 1e-4
    ):
        super().__init__()

        self.num_phases = num_phases
        self.embed_dim = embed_dim

        self.phase_embedding = nn.Embedding(num_phases, embed_dim)

        with torch.no_grad():
            known_embeds = self.phase_embedding.weight[:7]
            self.phase_embedding.weight[7] = known_embeds.mean(dim=0)

        if phase_prior is not None:
            # 添加平滑，处理27个全零三元组
            phase_prior_smoothed = phase_prior + prior_smoothing

            phase_prior_tensor = torch.from_numpy(phase_prior_smoothed).float()
            unknown_prior = phase_prior_tensor.mean(dim=1, keepdim=True)
            extended_prior = torch.cat([phase_prior_tensor, unknown_prior], dim=1)

            self.register_buffer('phase_prior', extended_prior.T)

            self.prior_encoder = nn.Sequential(
                nn.Linear(100, 128),
                nn.ReLU(),
                nn.Linear(128, embed_dim)
            )
            self.fusion = nn.Linear(embed_dim * 2, embed_dim)
            self.use_prior = True

            print(f"✓ Phase prior smoothed with ε={prior_smoothing}")
        else:
            self.use_prior = False

    def forward(self, phase_ids: torch.Tensor) -> torch.Tensor:
        embed = self.phase_embedding(phase_ids)

        if self.use_prior:
            prior_dist = self.phase_prior[phase_ids]
            prior_embed = self.prior_encoder(prior_dist)
            embed = self.fusion(torch.cat([embed, prior_embed], dim=-1))

        return embed


# ==================== 三元组组合器（修复版） ====================

class TripletCombinerV2(nn.Module):
    """
    三元组组合器（修复版v2）
    直接在logits空间操作，避免数值不稳定
    """
    def __init__(
        self,
        num_instruments: int = 6,
        num_verbs: int = 10,
        num_targets: int = 15,
        num_triplets: int = 100
    ):
        super().__init__()

        self.register_buffer('triplet_to_ivt', self._build_triplet_mapping())
        self.triplet_bias = nn.Parameter(torch.zeros(num_triplets))
        self.component_weights = nn.Parameter(torch.ones(3) / 3)

    def _build_triplet_mapping(self) -> torch.Tensor:
        mapping = []
        for inst, verb, targ in TRIPLET_CLASSES:
            mapping.append([
                INSTRUMENTS.index(inst),
                VERBS.index(verb),
                TARGETS.index(targ)
            ])
        return torch.tensor(mapping, dtype=torch.long)

    def forward(
        self,
        instrument_logits: torch.Tensor,
        verb_logits: torch.Tensor,
        target_logits: torch.Tensor
    ) -> torch.Tensor:
        """直接在logits空间组合，数值稳定"""
        i_idx = self.triplet_to_ivt[:, 0]
        v_idx = self.triplet_to_ivt[:, 1]
        t_idx = self.triplet_to_ivt[:, 2]

        l_i = instrument_logits[:, i_idx]
        l_v = verb_logits[:, v_idx]
        l_t = target_logits[:, t_idx]

        w = F.softmax(self.component_weights, dim=0)
        triplet_logits = w[0] * l_i + w[1] * l_v + w[2] * l_t + self.triplet_bias

        return triplet_logits


# ==================== 层次组合分类器 ====================

class HierarchicalClassifierV2(nn.Module):
    """层次组合分类器（修复版v2）"""
    def __init__(
        self,
        feature_dim: int = 1536,
        hidden_dim: int = 512,
        num_instruments: int = 6,
        num_verbs: int = 10,
        num_targets: int = 15,
        num_triplets: int = 100,
        phase_embed_dim: int = 64,
        dropout: float = 0.3,
        constraint_penalty: float = 15.0
    ):
        super().__init__()

        self.constraint_penalty = constraint_penalty

        # FiLM层
        self.film_instrument = FiLMLayer(hidden_dim, phase_embed_dim)
        self.film_verb = FiLMLayer(hidden_dim, phase_embed_dim)
        self.film_target = FiLMLayer(hidden_dim, phase_embed_dim)

        # 特征投影
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Instrument分类器
        self.instrument_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_instruments)
        )

        # Verb分类器（条件于Instrument）
        self.instrument_to_verb = nn.Linear(num_instruments, hidden_dim // 4)
        self.verb_classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_verbs)
        )

        # Target分类器（条件于I+V）
        self.iv_to_target = nn.Linear(num_instruments + num_verbs, hidden_dim // 4)
        self.target_classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_targets)
        )

        # 组合器
        self.triplet_combiner = TripletCombinerV2(
            num_instruments, num_verbs, num_targets, num_triplets
        )

        # 直接Triplet分类器
        self.direct_triplet_classifier = nn.Linear(hidden_dim, num_triplets)

        # LayerNorm
        self.composed_ln = nn.LayerNorm(num_triplets)
        self.direct_ln = nn.LayerNorm(num_triplets)

        # 组合权重
        self.combine_weight = nn.Parameter(torch.tensor(0.5))

    def _apply_constraint(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        use_hard: bool
    ) -> torch.Tensor:
        if use_hard:
            return logits + (mask - 1) * 1e9
        else:
            penalty = self.constraint_penalty * (1 - mask)
            return logits - penalty

    def forward(
        self,
        features: torch.Tensor,
        phase_embed: torch.Tensor,
        constraint_generator: Optional[ConstraintMaskGenerator] = None,
        use_hard_constraint: bool = False
    ) -> Dict[str, torch.Tensor]:

        # 1. 特征投影
        hidden = self.feature_proj(features)

        # 2. Instrument预测
        hidden_i = self.film_instrument(hidden, phase_embed)
        instrument_logits = self.instrument_classifier(hidden_i)
        instrument_probs = torch.sigmoid(instrument_logits)

        # 3. Verb预测
        instrument_embed = self.instrument_to_verb(instrument_probs)
        hidden_v = self.film_verb(hidden, phase_embed)
        verb_input = torch.cat([hidden_v, instrument_embed], dim=-1)
        verb_logits = self.verb_classifier(verb_input)

        if constraint_generator is not None:
            verb_mask = constraint_generator.get_verb_mask_given_instrument(instrument_probs)
            verb_logits = self._apply_constraint(verb_logits, verb_mask, use_hard_constraint)

        verb_probs = torch.sigmoid(verb_logits)

        # 4. Target预测
        iv_embed = self.iv_to_target(torch.cat([instrument_probs, verb_probs], dim=-1))
        hidden_t = self.film_target(hidden, phase_embed)
        target_input = torch.cat([hidden_t, iv_embed], dim=-1)
        target_logits = self.target_classifier(target_input)

        if constraint_generator is not None:
            target_mask = constraint_generator.get_target_mask_given_iv(instrument_probs, verb_probs)
            target_logits = self._apply_constraint(target_logits, target_mask, use_hard_constraint)

        target_probs = torch.sigmoid(target_logits)

        # 5. 组合Triplet（传入logits）
        triplet_composed = self.triplet_combiner(
            instrument_logits,
            verb_logits,
            target_logits
        )

        # 6. 直接预测Triplet
        triplet_direct = self.direct_triplet_classifier(hidden)

        # 7. 融合
        triplet_composed_norm = self.composed_ln(triplet_composed)
        triplet_direct_norm = self.direct_ln(triplet_direct)

        w = torch.sigmoid(self.combine_weight)
        triplet_logits = w * triplet_composed_norm + (1 - w) * triplet_direct_norm

        return {
            'instruments': instrument_logits,
            'verbs': verb_logits,
            'targets': target_logits,
            'triplets': triplet_logits,
            'triplets_composed': triplet_composed,
            'triplets_direct': triplet_direct,
            'instrument_probs': instrument_probs,
            'verb_probs': verb_probs,
            'target_probs': target_probs
        }


# ==================== 完整模型 ====================

class PriorGuidedHCNV2(nn.Module):
    """Prior-Guided Hierarchical Compositional Network (PG-HCN) v2"""
    def __init__(
        self,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        num_phases: int = NUM_PHASES,
        phase_embed_dim: int = 64,
        constraint_dir: str = None,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        prior_smoothing: float = 1e-4,
        constraint_penalty: float = 15.0
    ):
        super().__init__()

        if constraint_dir is None:
            constraint_dir = Config.DATA_ROOT

        # 1. 加载约束矩阵
        iv_constraint = np.load(f'{constraint_dir}/instrument_verb_constraint.npy')
        vt_constraint = np.load(f'{constraint_dir}/verb_target_constraint.npy')
        it_constraint = np.load(f'{constraint_dir}/instrument_target_constraint.npy')
        print(f"✓ Loaded constraint matrices")

        # 2. 加载Phase先验
        import pandas as pd
        phase_prior_path = f'{constraint_dir}/phase_triplet_prior.csv'
        if os.path.exists(phase_prior_path):
            phase_prior_df = pd.read_csv(phase_prior_path)
            phase_cols = [f'phase_{i}' for i in range(7)]
            phase_prior = phase_prior_df[phase_cols].values
            print(f"✓ Loaded phase prior: {phase_prior.shape}")

            zero_prior_count = (phase_prior.sum(axis=1) == 0).sum()
            print(f"  Zero-prior triplets: {zero_prior_count} (will be smoothed)")
        else:
            phase_prior = None
            print(f"⚠️ Phase prior not found")

        # 3. 视觉Backbone
        from model_lora import SimpleTripletClassifierLoRA, LoRALayer
        self.backbone_model = SimpleTripletClassifierLoRA(
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        self.backbone = self.backbone_model.backbone
        self.avgpool = self.backbone_model.avgpool

        # 4. 重新链接LoRA层
        self._relink_lora_layers()

        # 5. Phase编码器
        self.phase_encoder = PhaseEncoder(
            num_phases=num_phases,
            embed_dim=phase_embed_dim,
            phase_prior=phase_prior,
            prior_smoothing=prior_smoothing
        )

        # 6. 约束Mask生成器
        self.constraint_generator = ConstraintMaskGenerator(
            iv_constraint=iv_constraint,
            vt_constraint=vt_constraint,
            it_constraint=it_constraint,
            use_soft_max=True,
            temperature=10.0
        )

        # 7. 层次组合分类器
        self.hierarchical_classifier = HierarchicalClassifierV2(
            feature_dim=Config.FEATURE_DIM,
            hidden_dim=hidden_dim,
            num_instruments=Config.NUM_INSTRUMENTS,
            num_verbs=Config.NUM_VERBS,
            num_targets=Config.NUM_TARGETS,
            num_triplets=Config.NUM_TRIPLETS,
            phase_embed_dim=phase_embed_dim,
            dropout=dropout,
            constraint_penalty=constraint_penalty
        )

        # 8. Spatial weight buffer
        self.register_buffer('current_spatial_weight', None)

        self._print_model_stats()

    def _relink_lora_layers(self):
        from model_lora import LoRALayer
        count = 0
        for module in self.backbone.modules():
            if isinstance(module, LoRALayer):
                # 使用弱引用，避免循环引用导致递归
                module.parent_model = weakref.ref(self)
                count += 1
        print(f"✓ Relinked {count} LoRA layers (weakref)")

    def _print_model_stats(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"PG-HCN V2 Model Statistics")
        print(f"{'='*60}")
        print(f"Total parameters:     {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"{'='*60}\n")

    def forward(
        self,
        images: torch.Tensor,
        phase_ids: torch.Tensor,
        pseudo_mask: Optional[torch.Tensor] = None,
        use_hard_constraint: bool = False
    ) -> Dict[str, torch.Tensor]:

        if pseudo_mask is not None and Config.USE_PSEUDO_MASK:
            self.current_spatial_weight = self._process_pseudo_mask(pseudo_mask)
        else:
            self.current_spatial_weight = None

        features = self.backbone(images)
        features = self.avgpool(features)
        features = features.flatten(1)

        phase_embed = self.phase_encoder(phase_ids)

        outputs = self.hierarchical_classifier(
            features=features,
            phase_embed=phase_embed,
            constraint_generator=self.constraint_generator,
            use_hard_constraint=use_hard_constraint
        )

        return outputs

    def _process_pseudo_mask(self, pseudo_mask: torch.Tensor) -> torch.Tensor:
        mask_smooth = F.avg_pool2d(pseudo_mask, kernel_size=5, stride=1, padding=2)
        alpha = Config.PSEUDO_MASK_WEIGHT
        return 1.0 + alpha * mask_smooth

    def get_lora_parameters(self):
        return self.backbone_model.get_lora_parameters()


# ==================== 兼容别名（重要！） ====================
# 让训练脚本可以直接用 PriorGuidedHCN 导入
PriorGuidedHCN = PriorGuidedHCNV2


# ==================== 约束感知损失（之前缺失！） ====================

class ConstraintAwareLoss(nn.Module):
    """
    约束感知损失函数

    L_total = L_triplet + λ_comp * L_component + λ_const * L_constraint
    """
    def __init__(
        self,
        iv_constraint: np.ndarray,
        vt_constraint: np.ndarray,
        it_constraint: np.ndarray,
        lambda_component: float = 0.5,
        lambda_constraint: float = 0.1,
        gamma_neg: float = 4,
        gamma_pos: float = 1
    ):
        super().__init__()

        self.register_buffer('iv_mask', torch.from_numpy(iv_constraint).float())
        self.register_buffer('vt_mask', torch.from_numpy(vt_constraint).float())
        self.register_buffer('it_mask', torch.from_numpy(it_constraint).float())

        self.lambda_component = lambda_component
        self.lambda_constraint = lambda_constraint

        from losses import AsymmetricLoss
        self.asl = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=0.05)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        triplet_labels: torch.Tensor,
        instrument_labels: torch.Tensor,
        verb_labels: torch.Tensor,
        target_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        # 1. 三元组损失
        loss_triplet = self.asl(outputs['triplets'], triplet_labels)

        # 2. 组件损失
        loss_i = self.asl(outputs['instruments'], instrument_labels)
        loss_v = self.asl(outputs['verbs'], verb_labels)
        loss_t = self.asl(outputs['targets'], target_labels)
        loss_component = (loss_i + loss_v + loss_t) / 3

        # 3. 约束违反惩罚
        loss_constraint = self._compute_constraint_loss(outputs)

        # 4. 总损失
        loss_total = (
            loss_triplet +
            self.lambda_component * loss_component +
            self.lambda_constraint * loss_constraint
        )

        return {
            'total': loss_total,
            'triplet': loss_triplet,
            'component': loss_component,
            'constraint': loss_constraint,
            'loss_i': loss_i,
            'loss_v': loss_v,
            'loss_t': loss_t
        }

    def _compute_constraint_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算约束违反惩罚"""
        i_probs = outputs['instrument_probs']
        v_probs = outputs['verb_probs']
        t_probs = outputs['target_probs']

        # IV约束违反
        iv_joint = i_probs.unsqueeze(-1) * v_probs.unsqueeze(1)
        iv_violation = iv_joint * (1 - self.iv_mask.unsqueeze(0))
        loss_iv = iv_violation.sum(dim=[1, 2]).mean()

        # VT约束违反
        vt_joint = v_probs.unsqueeze(-1) * t_probs.unsqueeze(1)
        vt_violation = vt_joint * (1 - self.vt_mask.unsqueeze(0))
        loss_vt = vt_violation.sum(dim=[1, 2]).mean()

        # IT约束违反
        it_joint = i_probs.unsqueeze(-1) * t_probs.unsqueeze(1)
        it_violation = it_joint * (1 - self.it_mask.unsqueeze(0))
        loss_it = it_violation.sum(dim=[1, 2]).mean()

        return loss_iv + loss_vt + loss_it


# ==================== 测试 ====================

if __name__ == '__main__':
    print("="*70)
    print(" PG-HCN V2 Complete Test")
    print("="*70)

    batch_size = 4

    # 1. 测试TripletCombinerV2
    print("\n1. TripletCombinerV2 (logits space)")
    combiner = TripletCombinerV2()
    i_logits = torch.randn(batch_size, 6)
    v_logits = torch.randn(batch_size, 10)
    t_logits = torch.randn(batch_size, 15)
    output = combiner(i_logits, v_logits, t_logits)
    print(f"   ✓ Output shape: {output.shape}")

    # 2. 测试兼容别名
    print("\n2. 兼容别名测试")
    print(f"   PriorGuidedHCN is PriorGuidedHCNV2: {PriorGuidedHCN is PriorGuidedHCNV2}")

    # 3. 测试ConstraintAwareLoss存在
    print("\n3. ConstraintAwareLoss")
    print(f"   ✓ Class defined: {ConstraintAwareLoss}")

    print("\n" + "="*70)
    print(" ✓ All components ready!")
    print("="*70)