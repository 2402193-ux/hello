"""
PG-HCN 训练脚本 V2 - 修复版 (NaN-Safe)
==========================================

修复内容:
1. 添加 NaN 检测和跳过机制
2. 降低 Stage 2 学习率 (lr_B 从 16x 降到 4x)
3. 添加梯度 NaN 检测和恢复
4. 添加更平滑的 warm-up 到 full 模式过渡
5. 添加模型状态备份和恢复机制
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# 使用旧版 API，兼容性最好
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import os
import json
import copy
from datetime import datetime

from config import Config
from dataset_pghcn import CholecT50DatasetPGHCN, NUM_PHASES

from model_prior_guided import PriorGuidedHCN, ConstraintAwareLoss

from model_lora import LoRALayer
from utils import set_seed, compute_map, save_checkpoint
from triplet_config import build_component_to_triplets


def check_for_nan(tensor, name="tensor"):
    """检查张量是否包含 NaN 或 Inf"""
    if tensor is None:
        return False
    if isinstance(tensor, dict):
        return any(check_for_nan(v, f"{name}.{k}") for k, v in tensor.items())
    if isinstance(tensor, (list, tuple)):
        return any(check_for_nan(v, f"{name}[{i}]") for i, v in enumerate(tensor))
    if isinstance(tensor, torch.Tensor):
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan or has_inf:
            return True
    return False


class PGHCNTrainer:
    """PG-HCN训练器 - 修复版"""

    def __init__(
        self,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        phase_embed_dim: int = 64,
        hidden_dim: int = 512
    ):
        set_seed(Config.SEED)
        Config.create_dirs()

        if Config.TRAIN_VIDEOS is None:
            Config.initialize()

        self.device = torch.device(Config.DEVICE)
        print(f"\n{'='*60}")
        print(f" PG-HCN V2 Trainer Initialization (NaN-Safe)")
        print(f"{'='*60}")
        print(f"Device: {self.device}")

        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            torch.cuda.empty_cache()

        # 创建数据加载器
        self._create_dataloaders()

        # 创建模型
        print(f"\n{'='*60}")
        print(f" Creating PG-HCN V2 Model")
        print(f"{'='*60}")

        self.model = PriorGuidedHCN(
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_phases=NUM_PHASES,
            phase_embed_dim=phase_embed_dim,
            constraint_dir=Config.DATA_ROOT,
            hidden_dim=hidden_dim,
            dropout=Config.DROPOUT
        ).to(self.device)

        # 加载约束矩阵用于损失函数
        iv_constraint = np.load(f'{Config.DATA_ROOT}/instrument_verb_constraint.npy')
        vt_constraint = np.load(f'{Config.DATA_ROOT}/verb_target_constraint.npy')
        it_constraint = np.load(f'{Config.DATA_ROOT}/instrument_target_constraint.npy')

        # 创建约束感知损失
        self.criterion = ConstraintAwareLoss(
            iv_constraint=iv_constraint,
            vt_constraint=vt_constraint,
            it_constraint=it_constraint,
            lambda_component=0.5,
            lambda_constraint=0.1
        ).to(self.device)

        # 训练历史
        self.history = {
            'stage1': {'train_loss': [], 'val_map': [], 'all_metrics': []},
            'stage2': {'train_loss': [], 'val_map': [], 'all_metrics': []}
        }

        # 阶段配置 (修复版)
        self.stage_configs = self._get_stage_configs()

        # 组件映射（用于分解标签）
        self.comp_map = build_component_to_triplets()

        # ====== 新增：模型状态备份 ======
        self.best_model_state = None
        self.nan_count = 0
        self.max_nan_tolerance = 10  # 最多容忍10次 NaN

    def _create_dataloaders(self):
        """创建数据加载器"""
        print(f"\n{'='*60}")
        print(f" Loading Datasets")
        print(f"{'='*60}")

        train_dataset = CholecT50DatasetPGHCN(
            Config.TRAIN_VIDEOS,
            is_train=True,
            phase_dir=os.path.join(Config.DATA_ROOT, 'phase'),
            include_no_phase_videos=True
        )

        val_dataset = CholecT50DatasetPGHCN(
            Config.VAL_VIDEOS,
            is_train=False,
            phase_dir=os.path.join(Config.DATA_ROOT, 'phase'),
            include_no_phase_videos=True
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True if Config.NUM_WORKERS > 0 else False
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True if Config.NUM_WORKERS > 0 else False
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        print(f"\n📊 Dataset Summary:")
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Val:   {len(val_dataset)} samples")

        phase_dist = train_dataset.get_phase_distribution()
        print(f"\n📊 Phase Distribution (Train):")
        for phase_id, count in phase_dist.items():
            phase_name = f"Phase {phase_id}" if phase_id < 7 else "Unknown"
            pct = count / len(train_dataset) * 100
            print(f"   {phase_name}: {count:6d} ({pct:5.1f}%)")

    def _get_stage_configs(self):
        """获取阶段配置 - 修复版"""
        return {
            'stage1': {
                'name': 'Hierarchical Classifier Training',
                'train_lora': False,
                'learning_rate': 1e-3,
                'batch_size': 16,
                'max_epochs': 50,
                'patience': 5,
                'warmup_epochs': 5,
                'lambda_constraint': 0.05,
                'use_pseudo_mask': False,
                # ====== 新增：过渡期配置 ======
                'transition_epochs': 3,  # warm-up 后的过渡期
            },
            'stage2': {
                'name': 'LoRA Fine-tuning',
                'train_lora': True,
                # ====== 修复：降低学习率 ======
                'learning_rate': 5e-5,  # 从 1e-4 降到 5e-5
                'lora_lr_multiplier': 4,  # 从 16 降到 4
                'batch_size': 8,
                'max_epochs': 50,
                'patience': 5,
                'warmup_epochs': 3,  # 添加 Stage 2 的 warm-up
                'lambda_constraint': 0.1,
                'use_pseudo_mask': True
            }
        }

    def _set_stage_config(self, stage_name: str):
        """设置阶段配置"""
        config = self.stage_configs[stage_name]

        print(f"\n{'='*60}")
        print(f"  {config['name'].upper()}")
        print(f"{'='*60}")

        # 1. 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. 解冻层次分类器
        for param in self.model.hierarchical_classifier.parameters():
            param.requires_grad = True
        print("✓ Hierarchical Classifier: TRAINABLE")

        # 3. 解冻Phase编码器
        for param in self.model.phase_encoder.parameters():
            param.requires_grad = True
        print("✓ Phase Encoder: TRAINABLE")

        # 4. 收集可训练参数
        trainable_params = []

        classifier_params = (
            list(self.model.hierarchical_classifier.parameters()) +
            list(self.model.phase_encoder.parameters())
        )
        trainable_params.append({
            'params': classifier_params,
            'lr': config['learning_rate']
        })

        # 5. 根据阶段设置LoRA
        if config['train_lora']:
            lora_A_params, lora_B_params = self.model.get_lora_parameters()

            for param in lora_A_params:
                param.requires_grad = True
            for param in lora_B_params:
                param.requires_grad = True

            lora_lr = config['learning_rate']
            lr_multiplier = config.get('lora_lr_multiplier', 4)  # 默认 4x

            trainable_params.append({'params': lora_A_params, 'lr': lora_lr})
            trainable_params.append({'params': lora_B_params, 'lr': lora_lr * lr_multiplier})

            print(f"✓ LoRA: TRAINABLE (LoRA+ strategy)")
            print(f"  lr_A = {lora_lr}, lr_B = {lora_lr * lr_multiplier}")
        else:
            for module in self.model.modules():
                if isinstance(module, LoRALayer):
                    if module.lora_A is not None:
                        module.lora_A.requires_grad = False
                    if module.lora_B is not None:
                        module.lora_B.requires_grad = False
            print("✓ LoRA: FROZEN")

        # 6. 创建优化器
        self.optimizer = optim.AdamW(trainable_params, weight_decay=1e-4)

        # 7. 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )

        # 8. 混合精度
        if Config.USE_AMP:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # 9. 更新约束损失权重
        self.criterion.lambda_constraint = config['lambda_constraint']

        # 10. 更新pseudo mask设置
        Config.USE_PSEUDO_MASK = config['use_pseudo_mask']
        print(f"✓ Pseudo Mask: {'ENABLED' if config['use_pseudo_mask'] else 'DISABLED'}")

        # 11. 如果batch size变化，重建dataloader
        if config['batch_size'] != Config.BATCH_SIZE:
            Config.BATCH_SIZE = config['batch_size']
            self._create_dataloaders()

        # 12. 统计参数
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())

        print(f"\n📊 Parameter Statistics:")
        print(f"   Total:     {total:,}")
        print(f"   Trainable: {trainable:,} ({trainable/total*100:.1f}%)")
        print(f"\n⚙️  Training Config:")
        print(f"   Learning Rate: {config['learning_rate']}")
        print(f"   Batch Size:    {config['batch_size']}")
        print(f"   Max Epochs:    {config['max_epochs']}")
        print(f"   Patience:      {config['patience']}")
        print(f"   Warmup Epochs: {config['warmup_epochs']}")
        print(f"   λ_constraint:  {config['lambda_constraint']}")
        print(f"{'='*60}\n")

        return config

    def _decompose_labels(self, triplet_labels: torch.Tensor):
        """从triplet labels分解出component labels"""
        batch_size = triplet_labels.shape[0]
        device = triplet_labels.device

        instrument_labels = torch.zeros(batch_size, Config.NUM_INSTRUMENTS, device=device)
        verb_labels = torch.zeros(batch_size, Config.NUM_VERBS, device=device)
        target_labels = torch.zeros(batch_size, Config.NUM_TARGETS, device=device)

        for i in range(Config.NUM_INSTRUMENTS):
            triplet_ids = self.comp_map['instruments'][i]
            if triplet_ids:
                instrument_labels[:, i] = triplet_labels[:, triplet_ids].max(dim=1)[0]

        for v in range(Config.NUM_VERBS):
            triplet_ids = self.comp_map['verbs'][v]
            if triplet_ids:
                verb_labels[:, v] = triplet_labels[:, triplet_ids].max(dim=1)[0]

        for t in range(Config.NUM_TARGETS):
            triplet_ids = self.comp_map['targets'][t]
            if triplet_ids:
                target_labels[:, t] = triplet_labels[:, triplet_ids].max(dim=1)[0]

        return instrument_labels, verb_labels, target_labels

    def _compute_loss_weight(self, epoch: int, warmup_epochs: int, transition_epochs: int = 3):
        """
        计算损失权重，实现平滑过渡

        Args:
            epoch: 当前 epoch (1-based)
            warmup_epochs: warm-up 期数
            transition_epochs: 过渡期数

        Returns:
            triplet_weight, component_weight: 损失权重
        """
        if epoch <= warmup_epochs:
            # Warm-up: 主要训练 component
            return 0.1, 1.0
        elif epoch <= warmup_epochs + transition_epochs:
            # 过渡期: 逐渐增加 triplet 权重
            progress = (epoch - warmup_epochs) / transition_epochs
            triplet_w = 0.1 + 0.9 * progress  # 0.1 -> 1.0
            component_w = 1.0 - 0.5 * progress  # 1.0 -> 0.5
            return triplet_w, component_w
        else:
            # Full: 使用标准权重
            return 1.0, 0.5

    def train_epoch(self, epoch: int, warmup_epochs: int = 0, transition_epochs: int = 3):
        """训练一个epoch - 修复版"""
        self.model.train()

        total_losses = {
            'total': 0, 'triplet': 0, 'component': 0, 'constraint': 0
        }
        valid_batches = 0
        nan_batches = 0

        self.optimizer.zero_grad()

        # 计算当前 epoch 的损失权重
        triplet_w, component_w = self._compute_loss_weight(epoch, warmup_epochs, transition_epochs)

        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for idx, batch_data in enumerate(pbar):
                images, labels, video_ids, pseudo_masks, phase_ids = batch_data

                images = images.to(self.device)
                triplet_labels = labels.to(self.device)
                pseudo_masks = pseudo_masks.to(self.device)
                phase_ids = phase_ids.to(self.device)

                instrument_labels, verb_labels, target_labels = \
                    self._decompose_labels(triplet_labels)

                try:
                    if Config.USE_AMP and self.scaler is not None:
                        with autocast():
                            outputs = self.model(
                                images, phase_ids, pseudo_masks,
                                use_hard_constraint=False
                            )

                            # ====== 检查输出是否有 NaN ======
                            if check_for_nan(outputs):
                                nan_batches += 1
                                if nan_batches > self.max_nan_tolerance:
                                    print(f"\n⚠️ Too many NaN batches ({nan_batches}), stopping...")
                                    break
                                continue

                            losses = self.criterion(
                                outputs, triplet_labels,
                                instrument_labels, verb_labels, target_labels
                            )

                            # ====== 检查损失是否有 NaN ======
                            if check_for_nan(losses):
                                nan_batches += 1
                                if nan_batches > self.max_nan_tolerance:
                                    print(f"\n⚠️ Too many NaN losses ({nan_batches}), stopping...")
                                    break
                                continue

                            # 使用平滑过渡的权重
                            loss = (triplet_w * losses['triplet'] +
                                   component_w * losses['component'] +
                                   self.criterion.lambda_constraint * losses['constraint'])

                            loss = loss / Config.ACCUMULATION_STEPS

                        self.scaler.scale(loss).backward()

                        if (idx + 1) % Config.ACCUMULATION_STEPS == 0:
                            self.scaler.unscale_(self.optimizer)

                            # ====== 检查梯度是否有 NaN ======
                            grad_has_nan = False
                            for param in self.model.parameters():
                                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                    grad_has_nan = True
                                    break

                            if grad_has_nan:
                                nan_batches += 1
                                self.optimizer.zero_grad()
                                if nan_batches > self.max_nan_tolerance:
                                    print(f"\n⚠️ Too many NaN gradients ({nan_batches}), stopping...")
                                    break
                                continue

                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                    else:
                        outputs = self.model(
                            images, phase_ids, pseudo_masks,
                            use_hard_constraint=False
                        )

                        if check_for_nan(outputs):
                            nan_batches += 1
                            continue

                        losses = self.criterion(
                            outputs, triplet_labels,
                            instrument_labels, verb_labels, target_labels
                        )

                        if check_for_nan(losses):
                            nan_batches += 1
                            continue

                        loss = (triplet_w * losses['triplet'] +
                               component_w * losses['component'] +
                               self.criterion.lambda_constraint * losses['constraint'])

                        loss = loss / Config.ACCUMULATION_STEPS
                        loss.backward()

                        if (idx + 1) % Config.ACCUMULATION_STEPS == 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                    # 更新统计
                    for k in total_losses:
                        if not torch.isnan(losses[k]) and not torch.isinf(losses[k]):
                            total_losses[k] += losses[k].item()
                    valid_batches += 1

                    # 确定模式
                    if epoch <= warmup_epochs:
                        status = "WARMUP"
                    elif epoch <= warmup_epochs + transition_epochs:
                        status = "TRANS"
                    else:
                        status = "FULL"

                    pbar.set_postfix({
                        'mode': status,
                        'loss': f"{losses['total'].item():.4f}",
                        'trip': f"{losses['triplet'].item():.3f}",
                        'comp': f"{losses['component'].item():.3f}",
                        'const': f"{losses['constraint'].item():.3f}",
                        'nan': nan_batches
                    })

                except RuntimeError as e:
                    if "nan" in str(e).lower() or "inf" in str(e).lower():
                        nan_batches += 1
                        self.optimizer.zero_grad()
                        continue
                    else:
                        raise e

        # 处理剩余的梯度
        if len(self.train_loader) % Config.ACCUMULATION_STEPS != 0:
            if Config.USE_AMP and self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        # 计算平均损失
        if valid_batches > 0:
            for k in total_losses:
                total_losses[k] /= valid_batches

        print(f"Train Loss: {total_losses['total']:.4f} "
              f"(trip:{total_losses['triplet']:.3f} "
              f"comp:{total_losses['component']:.3f} "
              f"const:{total_losses['constraint']:.3f}) "
              f"[valid:{valid_batches}, nan:{nan_batches}]")

        return total_losses['total']

    def evaluate(self, use_hard_constraint: bool = True):
        """评估模型（Video-level RDV Protocol）"""
        from utils import decompose_triplet_to_components, decompose_triplet_to_pairs

        self.model.eval()

        print(f"\n{'='*70}")
        print(f" EVALUATION (RDV Protocol, Hard Constraint: {use_hard_constraint})")
        print(f"{'='*70}")

        video_ids = self.val_dataset.get_all_video_ids()
        print(f"Evaluating on {len(video_ids)} videos...")

        video_metrics = {
            'APi': [], 'APv': [], 'APt': [],
            'APiv': [], 'APit': [], 'APivt': []
        }

        with torch.no_grad():
            for video_id in video_ids:
                video_dataset = self.val_dataset.get_video_dataset(video_id)
                video_loader = DataLoader(
                    video_dataset,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True
                )

                video_labels = []
                video_preds = []

                for batch_data in video_loader:
                    images, labels, _, pseudo_masks, phase_ids = batch_data

                    images = images.to(self.device)
                    pseudo_masks = pseudo_masks.to(self.device)
                    phase_ids = phase_ids.to(self.device)

                    if Config.USE_AMP:
                        with autocast():
                            outputs = self.model(
                                images, phase_ids, pseudo_masks,
                                use_hard_constraint=use_hard_constraint
                            )
                    else:
                        outputs = self.model(
                            images, phase_ids, pseudo_masks,
                            use_hard_constraint=use_hard_constraint
                        )

                    probs = torch.sigmoid(outputs['triplets'])

                    # ====== 处理 NaN 预测 ======
                    if torch.isnan(probs).any():
                        probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)

                    video_labels.append(labels.numpy())
                    video_preds.append(probs.cpu().numpy())

                video_labels = np.concatenate(video_labels, axis=0)
                video_preds = np.concatenate(video_preds, axis=0)

                # ====== 处理 NaN 预测 ======
                video_preds = np.nan_to_num(video_preds, nan=0.0)

                video_ap_ivt, _ = compute_map(video_labels, video_preds)

                components = decompose_triplet_to_components(video_labels, video_preds)
                pairs = decompose_triplet_to_pairs(video_labels, video_preds)

                video_ap_i, _ = compute_map(
                    components['instrument_labels'],
                    components['instrument_probs']
                )
                video_ap_v, _ = compute_map(
                    components['verb_labels'],
                    components['verb_probs']
                )
                video_ap_t, _ = compute_map(
                    components['target_labels'],
                    components['target_probs']
                )
                video_ap_iv, _ = compute_map(pairs['iv_labels'], pairs['iv_probs'])
                video_ap_it, _ = compute_map(pairs['it_labels'], pairs['it_probs'])

                video_metrics['APi'].append(video_ap_i)
                video_metrics['APv'].append(video_ap_v)
                video_metrics['APt'].append(video_ap_t)
                video_metrics['APiv'].append(video_ap_iv)
                video_metrics['APit'].append(video_ap_it)
                video_metrics['APivt'].append(video_ap_ivt)

                print(f"  {video_id}: APivt={video_ap_ivt:.3f} "
                      f"APi={video_ap_i:.3f} APv={video_ap_v:.3f} APt={video_ap_t:.3f} "
                      f"({len(video_labels):4d} frames)")

        final_metrics = {
            key: np.mean(values) for key, values in video_metrics.items()
        }

        print(f"\n{'='*70}")
        print(f" VIDEO-LEVEL RESULTS (averaged over {len(video_ids)} videos)")
        print(f"{'='*70}")

        print("\n📊 Component Detection:")
        print(f"   APi (Instrument): {final_metrics['APi']*100:6.2f}%")
        print(f"   APv (Verb):       {final_metrics['APv']*100:6.2f}%")
        print(f"   APt (Target):     {final_metrics['APt']*100:6.2f}%")

        print("\n🔗 Triplet Association:")
        print(f"   APiv (I-V):       {final_metrics['APiv']*100:6.2f}%")
        print(f"   APit (I-T):       {final_metrics['APit']*100:6.2f}%")

        print("\n⭐ Main Metric:")
        print(f"   APivt (I-V-T):    {final_metrics['APivt']*100:6.2f}%  ← MAIN")

        print(f"\n📈 Statistics:")
        print(f"   Min:  {min(video_metrics['APivt'])*100:.2f}%")
        print(f"   Max:  {max(video_metrics['APivt'])*100:.2f}%")
        print(f"   Std:  {np.std(video_metrics['APivt'])*100:.2f}%")

        print(f"\n📖 Comparison to RDV (SOTA):")
        print("-" * 70)
        rdv = {'APi': 0.920, 'APv': 0.607, 'APt': 0.383,
               'APiv': 0.394, 'APit': 0.369, 'APivt': 0.299}
        print(f"  {'Metric':<8} {'Yours':>8} {'RDV':>8} {'Diff':>8}")
        print("-" * 70)
        for key in ['APi', 'APv', 'APt', 'APiv', 'APit', 'APivt']:
            yours = final_metrics[key]
            diff = yours - rdv[key]
            marker = " ⭐" if key == 'APivt' else ""
            print(f"  {key:<8} {yours*100:7.2f}% {rdv[key]*100:7.2f}% {diff*100:+7.2f}pp{marker}")

        print(f"\n{'='*70}\n")

        return final_metrics['APivt'], final_metrics

    def train_stage(self, stage_name: str, start_epoch: int = 0, best_map: float = 0.0):
        """训练一个阶段"""
        config = self._set_stage_config(stage_name)

        best_map_stage = best_map
        patience_counter = 0
        warmup_epochs = config['warmup_epochs']
        transition_epochs = config.get('transition_epochs', 3)

        # ====== 备份初始模型状态 ======
        self.best_model_state = copy.deepcopy(self.model.state_dict())

        for epoch in range(start_epoch, start_epoch + config['max_epochs']):
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            train_loss = self.train_epoch(
                epoch + 1,
                warmup_epochs=warmup_epochs,
                transition_epochs=transition_epochs
            )

            # ====== 检查训练损失是否为 NaN ======
            if np.isnan(train_loss):
                print(f"\n⚠️ NaN detected in training loss!")
                print(f"   Restoring best model state...")
                self.model.load_state_dict(self.best_model_state)
                self.nan_count += 1

                if self.nan_count >= 3:
                    print(f"   Too many NaN occurrences, stopping stage...")
                    break
                continue

            if epoch + 1 <= warmup_epochs:
                print(f"  [Warm-up: {epoch + 1}/{warmup_epochs}] Training components only")
            elif epoch + 1 <= warmup_epochs + transition_epochs:
                print(f"  [Transition: {epoch + 1 - warmup_epochs}/{transition_epochs}] Smooth transition")

            val_map, all_metrics = self.evaluate(use_hard_constraint=True)
            print(f"Validation APivt: {val_map:.4f}")

            self.history[stage_name]['train_loss'].append(train_loss)
            self.history[stage_name]['val_map'].append(val_map)
            self.history[stage_name]['all_metrics'].append({
                k: float(v) for k, v in all_metrics.items()
            })

            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_map)
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print(f"✓ LR reduced: {old_lr:.6f} -> {new_lr:.6f}")

            # 更新最佳模型
            if val_map > best_map_stage:
                best_map_stage = val_map
                patience_counter = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())

                save_path = os.path.join(
                    Config.CHECKPOINT_DIR,
                    f'pghcn_v2_{stage_name}_best.pth'
                )
                save_checkpoint(self.model, self.optimizer, epoch, val_map, save_path)

                if epoch + 1 <= warmup_epochs:
                    print(f"★ New best (warm-up)! APivt: {val_map:.4f}")
                elif epoch + 1 <= warmup_epochs + transition_epochs:
                    print(f"★ New best (transition)! APivt: {val_map:.4f}")
                else:
                    print(f"★ New best! APivt: {val_map:.4f}")
            else:
                if epoch + 1 > warmup_epochs + transition_epochs:
                    patience_counter += 1
                    print(f"Patience: {patience_counter}/{config['patience']}")

                    if patience_counter >= config['patience']:
                        print(f"\n✓ Early stopping at epoch {epoch + 1}")
                        break

            if self.device.type == 'cuda':
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                cached = torch.cuda.memory_reserved(0) / 1024**3
                print(f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

            print()

        print(f"\n✓ {config['name']} completed")
        print(f"Best mAP: {best_map_stage:.4f}")

        return best_map_stage

    def train(self):
        """完整训练流程"""
        print("\n" + "="*60)
        print(" "*15 + "PG-HCN V2 TRAINING (NaN-Safe)")
        print("="*60)

        print("\n🚀 Starting Stage 1: Hierarchical Classifier Training")
        stage1_map = self.train_stage('stage1')

        stage1_checkpoint = os.path.join(Config.CHECKPOINT_DIR, 'pghcn_v2_stage1_best.pth')
        if os.path.exists(stage1_checkpoint):
            checkpoint = torch.load(stage1_checkpoint, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\n✓ Loaded Stage 1 best checkpoint (mAP: {checkpoint['best_map']:.4f})")

        print("\n" + "="*60)
        print("🔄 Switching to Stage 2 (LoRA Fine-tuning)...")
        print("="*60)

        # ====== 重置 NaN 计数器 ======
        self.nan_count = 0

        print("\n🚀 Starting Stage 2: LoRA Fine-tuning")
        stage2_map = self.train_stage('stage2', best_map=stage1_map)

        self._print_final_summary(stage1_map, stage2_map)
        self._save_training_history()

        return stage2_map

    def _print_final_summary(self, stage1_map: float, stage2_map: float):
        """打印最终总结"""
        print("\n" + "="*60)
        print(" "*15 + "TRAINING SUMMARY")
        print("="*60)

        print(f"\n📊 Stage 1: Hierarchical Classifier Training")
        print(f"   Best mAP: {stage1_map:.4f}")
        print(f"   Epochs:   {len(self.history['stage1']['val_map'])}")

        print(f"\n📊 Stage 2: LoRA Fine-tuning")
        print(f"   Best mAP: {stage2_map:.4f}")
        print(f"   Epochs:   {len(self.history['stage2']['val_map'])}")

        improvement = stage2_map - stage1_map
        if stage1_map > 0:
            pct = improvement / stage1_map * 100
        else:
            pct = 0

        print(f"\n🎯 Overall Performance")
        print(f"   Stage 1 → Stage 2: {stage1_map:.4f} → {stage2_map:.4f}")
        print(f"   Improvement: {improvement:+.4f} ({pct:+.2f}%)")

        total_epochs = len(self.history['stage1']['val_map']) + len(self.history['stage2']['val_map'])
        print(f"\n⏱️  Total Training")
        print(f"   Epochs:    {total_epochs}")
        print(f"   Final mAP: {stage2_map:.4f}")

        print("\n" + "="*60)
        print("✓ PG-HCN V2 Training Completed!")
        print("="*60 + "\n")

    def _save_training_history(self):
        """保存训练历史"""
        history_path = os.path.join(Config.LOG_DIR, 'training_history_pghcn_v2.json')

        history_data = {
            'timestamp': datetime.now().isoformat(),
            'version': 'v2-nan-safe',
            'stage1': {
                'train_loss': [float(x) if not np.isnan(x) else 0.0 for x in self.history['stage1']['train_loss']],
                'val_map': [float(x) for x in self.history['stage1']['val_map']],
                'all_metrics': self.history['stage1']['all_metrics']
            },
            'stage2': {
                'train_loss': [float(x) if not np.isnan(x) else 0.0 for x in self.history['stage2']['train_loss']],
                'val_map': [float(x) for x in self.history['stage2']['val_map']],
                'all_metrics': self.history['stage2']['all_metrics']
            },
            'config': {
                'stage1': self.stage_configs['stage1'],
                'stage2': self.stage_configs['stage2']
            }
        }

        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)

        print(f"✓ Training history saved: {history_path}")


def main():
    """主函数"""
    trainer = PGHCNTrainer(
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        phase_embed_dim=64,
        hidden_dim=512
    )

    stage1_checkpoint = os.path.join(Config.CHECKPOINT_DIR, 'pghcn_v2_stage1_best.pth')

    if os.path.exists(stage1_checkpoint):
        print("\n✓ Found Stage 1 checkpoint, skipping to Stage 2")
        checkpoint = torch.load(stage1_checkpoint, weights_only=False)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        stage1_map = checkpoint['best_map']

        stage2_map = trainer.train_stage('stage2', best_map=stage1_map)
        trainer._print_final_summary(stage1_map, stage2_map)
        trainer._save_training_history()
    else:
        trainer.train()

    print(f"\n✓ Training completed!")


if __name__ == '__main__':
    main()