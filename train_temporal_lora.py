"""
TT-LoRA 训练脚本 - 时序三元组LoRA

三阶段训练策略：
Stage 1: Linear Probing（只训练分类头）
Stage 2: 空间LoRA微调（训练分类头 + 空间LoRA）
Stage 3: 时序LoRA微调（训练分类头 + 空间LoRA + 时序LoRA）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import os
import json
from datetime import datetime

from config import Config
from dataset import CholecT50Dataset
from model_temporal_lora import (
    TemporalTripletLoRA,
    TemporalCholecT50Dataset,
    LoRALayer
)
from losses import AsymmetricLoss
from utils import set_seed, compute_map, save_checkpoint


class TTLoRATrainer:
    """时序三元组LoRA训练器"""

    def __init__(self,
                 spatial_lora_rank=8,
                 temporal_lora_rank=8,
                 seq_len=4,
                 temporal_type='attention'):

        set_seed(Config.SEED)
        Config.create_dirs()
        if Config.TRAIN_VIDEOS is None:
            Config.initialize()

        self.device = torch.device(Config.DEVICE)
        self.seq_len = seq_len

        print(f"\n{'=' * 60}")
        print("TT-LoRA Trainer Initialization")
        print(f"{'=' * 60}")
        print(f"Device: {self.device}")
        print(f"Sequence length: {seq_len}")
        print(f"Temporal type: {temporal_type}")

        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()

        # 创建模型
        self.model = TemporalTripletLoRA(
            spatial_lora_rank=spatial_lora_rank,
            temporal_lora_rank=temporal_lora_rank,
            seq_len=seq_len,
            temporal_type=temporal_type
        ).to(self.device)

        # 损失函数
        self.criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)

        # 历史记录
        self.history = {
            'stage1': {'train_loss': [], 'val_map': []},
            'stage2': {'train_loss': [], 'val_map': []},
            'stage3': {'train_loss': [], 'val_map': []}
        }

        # 阶段配置
        self.stage_configs = self._get_stage_configs()

    def _get_stage_configs(self):
        """获取三个阶段的配置"""
        return {
            'stage1': {
                'name': 'Linear Probing',
                'train_spatial_lora': False,
                'train_temporal_lora': False,
                'use_sequence': False,  # 单帧训练
                'learning_rate': 1e-3,
                'batch_size': 24,
                'max_epochs': 30,
                'patience': 5
            },
            'stage2': {
                'name': 'Spatial LoRA Fine-tuning',
                'train_spatial_lora': True,
                'train_temporal_lora': False,
                'use_sequence': False,  # 仍然单帧
                'learning_rate': 1e-4,
                'batch_size': 8,
                'max_epochs': 30,
                'patience': 5
            },
            'stage3': {
                'name': 'Temporal LoRA Fine-tuning',
                'train_spatial_lora': True,
                'train_temporal_lora': True,
                'use_sequence': True,  # 序列训练！
                'learning_rate': 5e-5,
                'batch_size': 4,  # 序列需要更小batch
                'max_epochs': 30,
                'patience': 5
            }
        }

    def _create_dataloaders(self, use_sequence=False):
        """创建数据加载器"""
        print(f"\n--- Creating DataLoaders (sequence={use_sequence}) ---")

        # 基础数据集
        train_base = CholecT50Dataset(Config.TRAIN_VIDEOS, is_train=True)
        val_base = CholecT50Dataset(Config.VAL_VIDEOS, is_train=False)

        if use_sequence:
            # 时序数据集
            train_dataset = TemporalCholecT50Dataset(
                train_base,
                seq_len=self.seq_len,
                stride=1
            )
            val_dataset = TemporalCholecT50Dataset(
                val_base,
                seq_len=self.seq_len,
                stride=1
            )
        else:
            train_dataset = train_base
            val_dataset = val_base

        batch_size = self.current_config['batch_size']

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )

        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Batch size: {batch_size}")

    def _set_stage_config(self, stage_name):
        """设置当前阶段的配置"""
        config = self.stage_configs[stage_name]
        self.current_config = config

        print(f"\n{'=' * 60}")
        print(f"  {config['name'].upper()}")
        print(f"{'=' * 60}")

        # 1. 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. 解冻分类头
        for param in self.model.instrument_classifier.parameters():
            param.requires_grad = True
        for param in self.model.verb_classifier.parameters():
            param.requires_grad = True
        for param in self.model.target_classifier.parameters():
            param.requires_grad = True
        for param in self.model.triplet_fusion.parameters():
            param.requires_grad = True
        for param in self.model.triplet_classifier.parameters():
            param.requires_grad = True
        print("✓ Classifiers: TRAINABLE")

        # 3. 空间LoRA
        if config['train_spatial_lora']:
            for module in self.model.backbone.modules():
                if isinstance(module, LoRALayer):
                    if module.lora_A is not None:
                        module.lora_A.requires_grad = True
                    if module.lora_B is not None:
                        module.lora_B.requires_grad = True
            print("✓ Spatial LoRA: TRAINABLE")
        else:
            print("✗ Spatial LoRA: FROZEN")

        # 4. 时序LoRA
        if config['train_temporal_lora']:
            for param in self.model.temporal_lora.parameters():
                param.requires_grad = True
            print("✓ Temporal LoRA: TRAINABLE")
        else:
            print("✗ Temporal LoRA: FROZEN")

        # 创建数据加载器
        self._create_dataloaders(use_sequence=config['use_sequence'])

        # 创建优化器（不同参数组不同学习率）
        param_groups = []

        # 分类头
        classifier_params = self.model.get_classifier_parameters()
        param_groups.append({
            'params': classifier_params,
            'lr': config['learning_rate']
        })

        # 空间LoRA（如果训练）
        if config['train_spatial_lora']:
            spatial_params = self.model.get_spatial_lora_parameters()
            param_groups.append({
                'params': spatial_params,
                'lr': config['learning_rate'] * 0.5  # 稍低的学习率
            })

        # 时序LoRA（如果训练）
        if config['train_temporal_lora']:
            temporal_params = self.model.get_temporal_lora_parameters()
            param_groups.append({
                'params': temporal_params,
                'lr': config['learning_rate']
            })

        self.optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )

        self.scaler = GradScaler() if Config.USE_AMP else None

        # 打印参数统计
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"\nTrainable: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")
        print(f"{'=' * 60}\n")

        return config

    def train_epoch(self, epoch, use_sequence=False):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_loss_i, total_loss_v, total_loss_t, total_loss_trip = 0, 0, 0, 0

        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_data in pbar:
                if use_sequence:
                    # 🔥 序列模式：现在也有masks
                    images, labels, video_ids, masks = batch_data
                else:
                    # 🔥 单帧模式：正确解包4个元素
                    images, labels, video_ids, masks = batch_data

                images = images.to(self.device)
                labels = labels.to(self.device)

                # 🔥 mask也要放到GPU
                masks = masks.to(self.device)

                # 分解标签
                inst_labels, verb_labels, target_labels = self._decompose_labels(labels)

                self.optimizer.zero_grad()

                if Config.USE_AMP and self.scaler is not None:
                    with autocast():
                        # 🔥 传入mask
                        outputs = self.model(images, mask=masks)

                        loss_i = self.criterion(outputs['instruments'], inst_labels)
                        loss_v = self.criterion(outputs['verbs'], verb_labels)
                        loss_t = self.criterion(outputs['targets'], target_labels)
                        loss_trip = self.criterion(outputs['triplets'], labels)

                        loss = (loss_i + 2.0 * loss_v + loss_t) / 4 + loss_trip

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 🔥 传入mask
                    outputs = self.model(images, mask=masks)

                    loss_i = self.criterion(outputs['instruments'], inst_labels)
                    loss_v = self.criterion(outputs['verbs'], verb_labels)
                    loss_t = self.criterion(outputs['targets'], target_labels)
                    loss_trip = self.criterion(outputs['triplets'], labels)

                    loss = (loss_i + 2.0 * loss_v + loss_t) / 4 + loss_trip

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                total_loss_i += loss_i.item()
                total_loss_v += loss_v.item()
                total_loss_t += loss_t.item()
                total_loss_trip += loss_trip.item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'L_v': f'{loss_v.item():.3f}',  # 重点关注verb
                    'L_trip': f'{loss_trip.item():.3f}'
                })

        n = len(self.train_loader)
        print(f'Train Loss: {total_loss / n:.4f} '
              f'(I:{total_loss_i / n:.3f} V:{total_loss_v / n:.3f} '
              f'T:{total_loss_t / n:.3f} Trip:{total_loss_trip / n:.3f})')

        return total_loss / n

    def _decompose_labels(self, triplet_labels):
        """从triplet标签分解出component标签"""
        from triplet_config import build_component_to_triplets

        comp_map = build_component_to_triplets()
        batch_size = triplet_labels.shape[0]
        device = triplet_labels.device

        inst_labels = torch.zeros(batch_size, Config.NUM_INSTRUMENTS, device=device)
        verb_labels = torch.zeros(batch_size, Config.NUM_VERBS, device=device)
        target_labels = torch.zeros(batch_size, Config.NUM_TARGETS, device=device)

        for i in range(Config.NUM_INSTRUMENTS):
            triplet_ids = comp_map['instruments'][i]
            if triplet_ids:
                inst_labels[:, i] = triplet_labels[:, triplet_ids].max(dim=1)[0]

        for v in range(Config.NUM_VERBS):
            triplet_ids = comp_map['verbs'][v]
            if triplet_ids:
                verb_labels[:, v] = triplet_labels[:, triplet_ids].max(dim=1)[0]

        for t in range(Config.NUM_TARGETS):
            triplet_ids = comp_map['targets'][t]
            if triplet_ids:
                target_labels[:, t] = triplet_labels[:, triplet_ids].max(dim=1)[0]

        return inst_labels, verb_labels, target_labels

    @torch.no_grad()
    def evaluate(self, use_sequence=False):
        """评估模型"""
        self.model.eval()

        all_labels = []
        all_preds = []
        all_verb_labels = []
        all_verb_preds = []

        for batch_data in tqdm(self.val_loader, desc='Evaluating'):
            # 🔥 统一解包4个元素
            images, labels, _, masks = batch_data

            images = images.to(self.device)
            masks = masks.to(self.device)  # 🔥 mask也要放到GPU

            if Config.USE_AMP:
                with autocast():
                    outputs = self.model(images, mask=masks)  # 🔥 传入mask
            else:
                outputs = self.model(images, mask=masks)  # 🔥 传入mask

            triplet_probs = torch.sigmoid(outputs['triplets'])
            verb_probs = torch.sigmoid(outputs['verbs'])

            all_labels.append(labels.cpu().numpy())
            all_preds.append(triplet_probs.cpu().numpy())

            # 分解verb标签
            inst_labels, verb_labels, target_labels = self._decompose_labels(labels.to(self.device))
            all_verb_labels.append(verb_labels.cpu().numpy())
            all_verb_preds.append(verb_probs.cpu().numpy())

        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_verb_labels = np.concatenate(all_verb_labels, axis=0)
        all_verb_preds = np.concatenate(all_verb_preds, axis=0)

        # 计算triplet mAP
        triplet_map, _ = compute_map(all_labels, all_preds)

        # 计算verb mAP（关键指标！）
        verb_map, _ = compute_map(all_verb_labels, all_verb_preds)

        print(f"\n{'=' * 40}")
        print(f"Triplet mAP: {triplet_map * 100:.2f}%")
        print(f"Verb mAP:    {verb_map * 100:.2f}%  ← Key metric for temporal")
        print(f"{'=' * 40}\n")

        return triplet_map, verb_map

    def train_stage(self, stage_name, start_epoch=0, best_map=0.0):
        """训练一个阶段"""
        config = self._set_stage_config(stage_name)

        best_map_stage = best_map
        patience_counter = 0

        for epoch in range(start_epoch, start_epoch + config['max_epochs']):
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            train_loss = self.train_epoch(
                epoch + 1,
                use_sequence=config['use_sequence']
            )

            triplet_map, verb_map = self.evaluate(
                use_sequence=config['use_sequence']
            )

            self.history[stage_name]['train_loss'].append(train_loss)
            self.history[stage_name]['val_map'].append(triplet_map)

            self.scheduler.step(triplet_map)

            if triplet_map > best_map_stage:
                best_map_stage = triplet_map
                patience_counter = 0

                save_path = os.path.join(
                    Config.CHECKPOINT_DIR,
                    f'{stage_name}_best_ttlora.pth'
                )
                save_checkpoint(self.model, self.optimizer, epoch, triplet_map, save_path)
                print(f'★ New best! Triplet mAP: {triplet_map:.4f}, Verb mAP: {verb_map:.4f}')
            else:
                patience_counter += 1
                print(f'Patience: {patience_counter}/{config["patience"]}')

            if patience_counter >= config['patience']:
                print(f'\n✓ Early stopping at epoch {epoch + 1}')
                break

        print(f'\n✓ {config["name"]} completed. Best mAP: {best_map_stage:.4f}')
        return best_map_stage

    def train(self):
        """完整的三阶段训练"""
        print("\n" + "=" * 60)
        print(" " * 15 + "TT-LoRA THREE-STAGE TRAINING")
        print("=" * 60)

        # Stage 1: Linear Probing
        print("\n🚀 Stage 1: Linear Probing")
        stage1_map = self.train_stage('stage1')

        # 加载Stage 1最佳权重
        self._load_best_checkpoint('stage1')

        # Stage 2: Spatial LoRA
        print("\n🚀 Stage 2: Spatial LoRA Fine-tuning")
        stage2_map = self.train_stage('stage2', best_map=stage1_map)

        # 加载Stage 2最佳权重
        self._load_best_checkpoint('stage2')

        # Stage 3: Temporal LoRA（核心创新！）
        print("\n🚀 Stage 3: Temporal LoRA Fine-tuning (KEY INNOVATION)")
        stage3_map = self.train_stage('stage3', best_map=stage2_map)

        # 最终总结
        self._print_summary(stage1_map, stage2_map, stage3_map)

    def _load_best_checkpoint(self, stage_name):
        """加载指定阶段的最佳权重"""
        path = os.path.join(Config.CHECKPOINT_DIR, f'{stage_name}_best_ttlora.pth')
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded {stage_name} best checkpoint (mAP: {checkpoint['best_map']:.4f})")

    def _print_summary(self, stage1_map, stage2_map, stage3_map):
        """打印训练总结"""
        print("\n" + "=" * 60)
        print(" " * 20 + "TRAINING SUMMARY")
        print("=" * 60)
        print(f"\nStage 1 (Linear Probing):     {stage1_map * 100:.2f}%")
        print(f"Stage 2 (Spatial LoRA):       {stage2_map * 100:.2f}%")
        print(f"Stage 3 (Temporal LoRA):      {stage3_map * 100:.2f}%  ← Final")
        print(f"\nImprovement from temporal:    {(stage3_map - stage2_map) * 100:+.2f}%")
        print("=" * 60)


def main():
    trainer = TTLoRATrainer(
        spatial_lora_rank=8,
        temporal_lora_rank=8,
        seq_len=4,
        temporal_type='attention'  # 推荐attention
    )

    trainer.train()


if __name__ == '__main__':
    main()