"""
LoRA训练脚本 - 两阶段训练

阶段1: 只训练分类头 (Linear Probing)
阶段2: 训练分类头 + LoRA参数 (LoRA Fine-tuning)

相比全量微调的优势:
1. 参数量减少98% (196M → 2-5M)
2. 内存占用减少50%+
3. 训练速度更快
4. 效果通常不比全量微调差
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
from model_lora import SimpleTripletClassifierLoRA, LoRALayer  # ← 添加 LoRALayer
from utils import set_seed, compute_map, save_checkpoint


class LoRATwoStageTrainer:
    def __init__(self, lora_rank=8, lora_alpha=16, lora_dropout=0.1):
        set_seed(Config.SEED)
        Config.create_dirs()
        if Config.TRAIN_VIDEOS is None:
            Config.initialize()
        self.device = torch.device(Config.DEVICE)
        print(f"\nUsing device: {self.device}")

        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
            torch.cuda.empty_cache()

        self._create_dataloaders()
        self.model = SimpleTripletClassifierLoRA(
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        ).to(self.device)
        from losses import AsymmetricLoss
        self.criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
        self.history = {
            'stage1': {'train_loss': [], 'val_map': []},
            'stage2': {'train_loss': [], 'val_map': []}
        }
        self.stage_configs = self._get_stage_configs()

    def _create_dataloaders(self):
        print("\n=== Loading Datasets ===")
        train_dataset = CholecT50Dataset(Config.TRAIN_VIDEOS, is_train=True)
        val_dataset = CholecT50Dataset(Config.VAL_VIDEOS, is_train=False)

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

        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")

    def _get_stage_configs(self):
        """获取两个阶段的配置"""
        return {
            'stage1': {
                'name': 'Linear Probing',
                'train_lora': False,  # 不训练LoRA
                'learning_rate': 1e-3,
                'batch_size': 24,
                'max_epochs': 50,
                'patience': 5,
                'weight_decay': 1e-4
            },
            'stage2': {
                'name': 'LoRA Fine-tuning',
                'train_lora': True,  # 训练LoRA
                'learning_rate': 1e-4,
                'batch_size': 8,
                'max_epochs': 50,
                'patience': 3,
                'weight_decay': 1e-4
            }
        }

    def _set_stage_config(self, stage_name):
        """设置当前阶段的配置"""
        config = self.stage_configs[stage_name]

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
        for param in self.model.triplet_classifier.parameters():
            param.requires_grad = True
        print("✓ All 4 Classifiers: TRAINABLE")

        # 3. 收集分类头参数
        classifier_params = (
                list(self.model.instrument_classifier.parameters()) +
                list(self.model.verb_classifier.parameters()) +
                list(self.model.target_classifier.parameters()) +
                list(self.model.triplet_classifier.parameters())
        )

        # 4. 根据阶段创建优化器（只创建一次）
        if config['train_lora']:
            # Stage 2: LoRA+ 策略
            lora_A_params, lora_B_params = self.model.get_lora_parameters()

            # ✅ 关键修复：显式解冻LoRA参数
            for param in lora_A_params:
                param.requires_grad = True
            for param in lora_B_params:
                param.requires_grad = True

            lora_lr = config['learning_rate']
            all_params = [
                {'params': classifier_params, 'lr': config['learning_rate']},
                {'params': lora_A_params, 'lr': lora_lr},
                {'params': lora_B_params, 'lr': lora_lr * 16}
            ]
            self.optimizer = optim.AdamW(all_params, weight_decay=config['weight_decay'])
            print(f"✓ LoRA+ enabled: lr_A={lora_lr}, lr_B={lora_lr * 16}")
            print("✓ Optimizing: All Classifiers + LoRA")
        else:
            # Stage 1: 只训练分类头，冻结LoRA
            lora_count = 0
            for module in self.model.modules():
                if isinstance(module, LoRALayer):
                    if module.lora_A is not None:
                        module.lora_A.requires_grad = False
                        lora_count += 1
                    if module.lora_B is not None:
                        module.lora_B.requires_grad = False
                        lora_count += 1
            print(f"✓ LoRA parameters: FROZEN ({lora_count} parameters)")

            self.optimizer = optim.Adam(
                classifier_params,
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
            print("✓ Optimizing: All 4 Classifiers only")

        print(f"Learning Rate: {config['learning_rate']}")
        print(f"Batch Size: {config['batch_size']}")
        print(f"Max Epochs: {config['max_epochs']}")
        print(f"Patience: {config['patience']}")

        # 5. 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )

        # 6. 混合精度
        self.scaler = GradScaler() if Config.USE_AMP else None

        # 7. 如果batch size变化，重建dataloader
        if config['batch_size'] != Config.BATCH_SIZE:
            Config.BATCH_SIZE = config['batch_size']
            self._create_dataloaders()

        # 8. 参数验证
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n🔍 Parameter Verification:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({trainable_params / total_params * 100:.1f}%)")

        classifier_params_count = (
                sum(p.numel() for p in self.model.instrument_classifier.parameters()) +
                sum(p.numel() for p in self.model.verb_classifier.parameters()) +
                sum(p.numel() for p in self.model.target_classifier.parameters()) +
                sum(p.numel() for p in self.model.triplet_classifier.parameters())
        )
        lora_trainable = trainable_params - classifier_params_count
        print(f"  - All Classifiers: {classifier_params_count:,}")
        print(f"  - LoRA: {lora_trainable:,}")

        # 9. 阶段验证
        if not config['train_lora']:
            expected_min = 150_000
            expected_max = 250_000
            if trainable_params < expected_min or trainable_params > expected_max:
                print(f"\n⚠️  WARNING: Unexpected trainable params for Linear Probing!")
                print(f"   Expected: ~{expected_min:,} - {expected_max:,}")
                print(f"   Got: {trainable_params:,}")
            else:
                print(f"\n✅ Linear Probing verified: Only classifiers are trainable")
        else:
            if trainable_params < 2_000_000:
                print(f"\n⚠️  WARNING: Too few trainable params for LoRA Fine-tuning")
                print(f"   Expected: ~2.5M")
                print(f"   Got: {trainable_params:,}")

        print(f"{'=' * 60}\n")

        return config

    def train_epoch(self, epoch, warmup_epochs=5):
        """训练一个epoch - 支持Multi-Task Learning + Warm-up"""
        self.model.train()
        total_loss = 0
        total_loss_i = 0
        total_loss_v = 0
        total_loss_t = 0
        total_loss_triplet = 0

        self.optimizer.zero_grad()

        mask_stats = {'total': 0, 'with_mask': 0, 'all_ones': 0}

        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for idx, batch_data in enumerate(pbar):
                if len(batch_data) == 4:
                    images, labels, _, pseudo_masks = batch_data
                    pseudo_masks = pseudo_masks.to(self.device)

                    # 统计有效mask
                    mask_stats['total'] += pseudo_masks.shape[0]

                    # 检测是否是全1的mask（无效mask）
                    for i in range(pseudo_masks.shape[0]):
                        mask_min = pseudo_masks[i].min().item()
                        mask_max = pseudo_masks[i].max().item()
                        if mask_min >= 0.99 and mask_max <= 1.01:
                            mask_stats['all_ones'] += 1
                        if mask_max <= 0.01:
                            mask_stats['all_ones'] += 1
                        else:
                            mask_stats['with_mask'] += 1
                else:
                    # 没有mask数据
                    images, labels, _ = batch_data
                    pseudo_masks = None
                    mask_stats['total'] += images.shape[0]
                    mask_stats['all_ones'] += images.shape[0]
                images = images.to(self.device)
                triplet_labels = labels.to(self.device)

                # 从triplet labels分解出component labels
                instrument_labels, verb_labels, target_labels = self._decompose_labels(triplet_labels)

                # 混合精度训练
                if Config.USE_AMP and self.scaler is not None:
                    with autocast():
                        outputs = self.model(images, pseudo_mask=pseudo_masks)

                        # Multi-task loss
                        loss_i = self.criterion(outputs['instruments'], instrument_labels)
                        loss_v = self.criterion(outputs['verbs'], verb_labels)
                        loss_t = self.criterion(outputs['targets'], target_labels)
                        loss_triplet = self.criterion(outputs['triplets'], triplet_labels)

                        # Component loss（简单平均）
                        loss_component = (loss_i + loss_v + loss_t) / 3

                        if epoch <= warmup_epochs:
                            # Warm-up: 主要训练component，辅助训练triplet
                            loss = loss_component + 0.1 * loss_triplet
                        else:
                            # Full training: 全权重训练
                            loss = loss_component + loss_triplet

                        loss = loss / Config.ACCUMULATION_STEPS

                    self.scaler.scale(loss).backward()

                    if (idx + 1) % Config.ACCUMULATION_STEPS == 0:
                        # 梯度裁剪
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=1.0
                        )

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    outputs = self.model(images, pseudo_mask=pseudo_masks)

                    # Multi-task loss
                    loss_i = self.criterion(outputs['instruments'], instrument_labels)
                    loss_v = self.criterion(outputs['verbs'], verb_labels)
                    loss_t = self.criterion(outputs['targets'], target_labels)
                    loss_triplet = self.criterion(outputs['triplets'], triplet_labels)

                    # Component loss
                    loss_component = (loss_i + loss_v + loss_t) / 3

                    # ✅ Warm-up策略
                    if epoch <= warmup_epochs:
                        loss = loss_component+0.1 * loss_triplet
                    else:
                        loss = loss_component + loss_triplet

                    loss = loss / Config.ACCUMULATION_STEPS

                    loss.backward()

                    if (idx + 1) % Config.ACCUMULATION_STEPS == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=1.0
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                # 统计loss
                total_loss += loss.item() * Config.ACCUMULATION_STEPS
                total_loss_i += loss_i.item()
                total_loss_v += loss_v.item()
                total_loss_t += loss_t.item()
                total_loss_triplet += loss_triplet.item()

                # ✅ 更新进度条（显示warm-up状态）
                status = "WARMUP" if epoch <= warmup_epochs else "FULL"
                pbar.set_postfix({
                    'mode': status,
                    'loss': f'{loss.item() * Config.ACCUMULATION_STEPS:.4f}',
                    'L_i': f'{loss_i.item():.3f}',
                    'L_v': f'{loss_v.item():.3f}',
                    'L_t': f'{loss_t.item():.3f}',
                    'L_trip': f'{loss_triplet.item():.3f}'
                })

        # 处理最后不足ACCUMULATION_STEPS的batch
        if len(self.train_loader) % Config.ACCUMULATION_STEPS != 0:
            if Config.USE_AMP and self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / len(self.train_loader)
        avg_loss_i = total_loss_i / len(self.train_loader)
        avg_loss_v = total_loss_v / len(self.train_loader)
        avg_loss_t = total_loss_t / len(self.train_loader)
        avg_loss_triplet = total_loss_triplet / len(self.train_loader)

        print(
            f'Train Loss: {avg_loss:.4f} (I:{avg_loss_i:.3f} V:{avg_loss_v:.3f} T:{avg_loss_t:.3f} Trip:{avg_loss_triplet:.3f})')

        return avg_loss

    def _decompose_labels(self, triplet_labels):
        """
        从triplet labels分解出component labels（用于训练的auxiliary supervision）

        Args:
            triplet_labels: [batch, 100] torch.Tensor, 0/1 binary labels

        Returns:
            instrument_labels: [batch, 6]
            verb_labels: [batch, 10]
            target_labels: [batch, 15]
        """
        from triplet_config import build_component_to_triplets

        comp_map = build_component_to_triplets()

        batch_size = triplet_labels.shape[0]
        device = triplet_labels.device

        instrument_labels = torch.zeros(batch_size, Config.NUM_INSTRUMENTS, device=device)
        verb_labels = torch.zeros(batch_size, Config.NUM_VERBS, device=device)
        target_labels = torch.zeros(batch_size, Config.NUM_TARGETS, device=device)

        # 对每个component，取所有包含它的triplet的max
        for i in range(Config.NUM_INSTRUMENTS):
            triplet_ids = comp_map['instruments'][i]
            if triplet_ids:
                instrument_labels[:, i] = triplet_labels[:, triplet_ids].max(dim=1)[0]

        for v in range(Config.NUM_VERBS):
            triplet_ids = comp_map['verbs'][v]
            if triplet_ids:
                verb_labels[:, v] = triplet_labels[:, triplet_ids].max(dim=1)[0]

        for t in range(Config.NUM_TARGETS):
            triplet_ids = comp_map['targets'][t]
            if triplet_ids:
                target_labels[:, t] = triplet_labels[:, triplet_ids].max(dim=1)[0]

        return instrument_labels, verb_labels, target_labels

    def evaluate(self):
        from torch.utils.data import DataLoader
        self.model.eval()
        print("\n" + "=" * 70)
        print(" " * 20 + "EVALUATION (RDV Protocol)")
        print("=" * 70)
        print(f"\nEvaluating on {len(Config.VAL_VIDEOS)} videos...")
        val_dataset = CholecT50Dataset(Config.VAL_VIDEOS, is_train=False)
        video_ids = val_dataset.get_all_video_ids()

        video_dataloaders = []
        for video_id in video_ids:
            video_dataset = val_dataset.get_video_dataset(video_id)
            video_loader = DataLoader(
                video_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True
            )
            video_dataloaders.append((video_id, video_loader))

        video_metrics = {
            'APi': [], 'APv': [], 'APt': [],
            'APiv': [], 'APit': [], 'APivt': []
        }

        with torch.no_grad():
            for video_id, video_loader in video_dataloaders:
                video_labels = []
                video_preds = []
                for batch_data in video_loader:
                    if len(batch_data) == 4:
                        images, labels, _, pseudo_masks = batch_data
                        pseudo_masks = pseudo_masks.to(self.device)
                    else:
                        images, labels, _ = batch_data
                        pseudo_masks = None
                    images = images.to(self.device)
                    if Config.USE_AMP:
                        with autocast():
                            outputs = self.model(images, pseudo_mask=pseudo_masks)
                    else:
                        outputs = self.model(images, pseudo_mask=pseudo_masks)

                    logits = outputs['triplets']
                    probs = torch.sigmoid(logits)

                    video_labels.append(labels.cpu().numpy())
                    video_preds.append(probs.cpu().numpy())

                video_labels = np.concatenate(video_labels, axis=0)  # [N_frames, 100]
                video_preds = np.concatenate(video_preds, axis=0)
                video_ap_ivt, _ = compute_map(video_labels, video_preds)
                from utils import decompose_triplet_to_components, decompose_triplet_to_pairs

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
                video_ap_iv, _ = compute_map(
                    pairs['iv_labels'],
                    pairs['iv_probs']
                )
                video_ap_it, _ = compute_map(
                    pairs['it_labels'],
                    pairs['it_probs']
                )

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
        print("\n" + "=" * 70)
        print(" " * 20 + "VIDEO-LEVEL RESULTS")
        print("=" * 70)
        print(f"(averaged over {len(video_ids)} videos)\n")
        print("📊 Component Detection:")
        print("-" * 70)
        print(f"  APi (Instrument):     {final_metrics['APi'] * 100:6.2f}%")
        print(f"  APv (Verb):           {final_metrics['APv'] * 100:6.2f}%")
        print(f"  APt (Target):         {final_metrics['APt'] * 100:6.2f}%")
        print("\n🔗 Triplet Association:")
        print("-" * 70)
        print(f"  APiv (I-V pairs):     {final_metrics['APiv'] * 100:6.2f}%")
        print(f"  APit (I-T pairs):     {final_metrics['APit'] * 100:6.2f}%")
        print("\n⭐ Main Metric:")
        print("-" * 70)
        print(f"  APivt (I-V-T):        {final_metrics['APivt'] * 100:6.2f}%  ← MAIN METRIC")
        print(f"\n📈 Statistics:")
        print(f"  Min:  {min(video_metrics['APivt']) * 100:.2f}%")
        print(f"  Max:  {max(video_metrics['APivt']) * 100:.2f}%")
        print(f"  Std:  {np.std(video_metrics['APivt']) * 100:.2f}%")
        print(f"\n📖 Comparison to RDV (SOTA):")
        print("-" * 70)
        rdv_results = {
            'APi': 0.920, 'APv': 0.607, 'APt': 0.383,
            'APiv': 0.394, 'APit': 0.369, 'APivt': 0.299
        }
        print(f"  {'Metric':<8} {'Yours':>8} {'RDV':>8} {'Diff':>8}")
        print("-" * 70)
        for key in ['APi', 'APv', 'APt', 'APiv', 'APit', 'APivt']:
            yours = final_metrics[key]
            rdv = rdv_results[key]
            diff = yours - rdv
            marker = " ⭐" if key == 'APivt' else ""
            print(f"  {key:<8} {yours * 100:7.2f}% {rdv * 100:7.2f}% "
                  f"{diff * 100:+7.2f}pp{marker}")

        print("\n" + "=" * 70 + "\n")
        return final_metrics['APivt'], final_metrics

    def train_stage(self, stage_name, start_epoch=0, best_map=0.0):
        config = self._set_stage_config(stage_name)

        if stage_name == 'stage1':
            Config.USE_PSEUDO_MASK = False
            print(f"\n{'=' * 60}")
            print("  📍 PSEUDO MASK: DISABLED (Stage 1 - Global Learning)")
            print(f"{'=' * 60}")
        else:
            Config.USE_PSEUDO_MASK = True
            print(f"\n{'=' * 60}")
            print("  📍 PSEUDO MASK: ENABLED (Stage 2 - Spatial Refinement)")
            print(f"  📍 Weight: {Config.PSEUDO_MASK_WEIGHT}")
            print(f"{'=' * 60}")


        best_map_this_stage = best_map
        patience_counter = 0
        warmup_epochs = 5 if stage_name == 'stage1' else 0

        for epoch in range(start_epoch, start_epoch + config['max_epochs']):
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            train_loss = self.train_epoch(epoch + 1, warmup_epochs=warmup_epochs)
            if epoch + 1 <= warmup_epochs:
                print(f'  [Warm-up phase: {epoch + 1}/{warmup_epochs}] Training components only')
            else:
                print(f'  [Full training] Components + Triplets')

            val_map, all_metrics = self.evaluate()
            print(f'Validation APivt: {val_map:.4f}')
            self.history[stage_name]['train_loss'].append(train_loss)
            self.history[stage_name]['val_map'].append(val_map)

            if 'all_metrics' not in self.history[stage_name]:
                self.history[stage_name]['all_metrics'] = []
            self.history[stage_name]['all_metrics'].append({
                'APi': float(all_metrics['APi']),
                'APv': float(all_metrics['APv']),
                'APt': float(all_metrics['APt']),
                'APiv': float(all_metrics['APiv']),
                'APit': float(all_metrics['APit']),
                'APivt': float(all_metrics['APivt'])
            })

            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_map)
            new_lr = self.optimizer.param_groups[0]['lr']

            if new_lr < old_lr:
                print(f'✓ LR reduced: {old_lr:.6f} -> {new_lr:.6f}')

            if epoch + 1 > warmup_epochs:
                if val_map > best_map_this_stage:
                    best_map_this_stage = val_map
                    patience_counter = 0

                    save_path = os.path.join(
                        Config.CHECKPOINT_DIR,
                        f'{stage_name}_best_lora.pth'
                    )
                    save_checkpoint(self.model, self.optimizer, epoch, val_map, save_path)
                    print(f'★ New best! APivt: {val_map:.4f}')
                else:
                    patience_counter += 1
                    print(f'Patience: {patience_counter}/{config["patience"]}')

                if patience_counter >= config['patience']:
                    print(f'\n✓ Early stopping triggered at epoch {epoch + 1}')
                    print(f'Best {config["name"]} mAP: {best_map_this_stage:.4f}')
                    return best_map_this_stage, True
            else:
                if val_map > best_map_this_stage:
                    best_map_this_stage = val_map
                    save_path = os.path.join(
                        Config.CHECKPOINT_DIR,
                        f'{stage_name}_best_lora.pth'
                    )
                    save_checkpoint(self.model, self.optimizer, epoch, val_map, save_path)
                    print(f'★ New best (warm-up)! APivt: {val_map:.4f}')

                print(f'[Warm-up: patience checking disabled]')

            if self.device.type == 'cuda':
                allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
                cached = torch.cuda.memory_reserved(0) / 1024 ** 3
                print(f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

            print()

        print(f'\n✓ {config["name"]} completed')
        print(f'Best mAP: {best_map_this_stage:.4f}')
        return best_map_this_stage, False

    def train(self):
        print("\n" + "=" * 60)
        print(" " * 15 + "LORA TWO-STAGE TRAINING")
        print("=" * 60)
        print("\n🚀 Starting Stage 1: Linear Probing")
        stage1_map, stage1_stopped_early = self.train_stage('stage1')
        stage1_summary = {
            'best_map': float(stage1_map),
            'stopped_early': stage1_stopped_early,
            'train_loss': [float(x) for x in self.history['stage1']['train_loss']],
            'val_map': [float(x) for x in self.history['stage1']['val_map']]
        }

        # 阶段切换
        print("\n" + "=" * 60)
        print("🔄 Switching to Stage 2 (LoRA Fine-tuning)...")
        print("=" * 60)

        stage1_checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'stage1_best_lora.pth')
        if os.path.exists(stage1_checkpoint_path):
            checkpoint = torch.load(stage1_checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded Stage 1 best checkpoint (mAP: {checkpoint['best_map']:.4f})")
        else:
            print("⚠️  Stage 1 checkpoint not found, continuing with current weights")
        print("\n🚀 Starting Stage 2: LoRA Fine-tuning")
        stage2_map, stage2_stopped_early = self.train_stage(
            'stage2',
            start_epoch=len(self.history['stage1']['val_map']),
            best_map=stage1_map
        )

        stage2_summary = {
            'best_map': float(stage2_map),
            'stopped_early': stage2_stopped_early,
            'train_loss': [float(x) for x in self.history['stage2']['train_loss']],
            'val_map': [float(x) for x in self.history['stage2']['val_map']]
        }
        self._print_final_summary(stage1_summary, stage2_summary)
        self._save_training_history(stage1_summary, stage2_summary)
        print("\n" + "=" * 60)
        print("💡 Optional: Merge LoRA weights for faster inference?")
        print("   Run: model.merge_lora_weights()")
        print("=" * 60)

    def _print_final_summary(self, stage1_summary, stage2_summary):
        """打印最终总结"""
        print("\n" + "=" * 60)
        print(" " * 15 + "TRAINING SUMMARY")
        print("=" * 60)
        print("\n📊 Stage 1: Linear Probing")
        print(f"  Best mAP: {stage1_summary['best_map']:.4f}")
        print(f"  Epochs: {len(stage1_summary['val_map'])}")
        print(f"  Early stopped: {stage1_summary['stopped_early']}")
        print("\n📊 Stage 2: LoRA Fine-tuning")
        print(f"  Best mAP: {stage2_summary['best_map']:.4f}")
        print(f"  Epochs: {len(stage2_summary['val_map'])}")
        print(f"  Early stopped: {stage2_summary['stopped_early']}")
        improvement = stage2_summary['best_map'] - stage1_summary['best_map']
        print(f"\n🎯 Overall Performance")
        print(f"  Stage 1 → Stage 2: {stage1_summary['best_map']:.4f} → {stage2_summary['best_map']:.4f}")
        print(f"  Improvement: {improvement:+.4f} ({improvement / stage1_summary['best_map'] * 100:+.2f}%)")
        total_epochs = len(stage1_summary['val_map']) + len(stage2_summary['val_map'])
        print(f"\n⏱️  Total Training")
        print(f"  Epochs: {total_epochs}")
        print(f"  Final mAP: {stage2_summary['best_map']:.4f}")
        print("\n" + "=" * 60)
        print("✓ LoRA Two-Stage Training Completed!")
        print("=" * 60 + "\n")

    def _save_training_history(self, stage1_summary, stage2_summary):
        history_path = os.path.join(Config.LOG_DIR, 'training_history_lora.json')
        history_data = {
            'timestamp': datetime.now().isoformat(),
            'stage1': stage1_summary,
            'stage2': stage2_summary,
            'config': {
                'stage1': self.stage_configs['stage1'],
                'stage2': self.stage_configs['stage2']
            }
        }

        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)

        print(f"✓ Training history saved: {history_path}")


def main():
    trainer = LoRATwoStageTrainer(
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1
    )

    stage1_checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'stage1_best_lora.pth')

    if os.path.exists(stage1_checkpoint_path):
        print("\n✓ Found Stage 1 checkpoint, skipping to Stage 2")
        checkpoint = torch.load(stage1_checkpoint_path)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        stage1_map = checkpoint['best_map']
    else:
        print("\n✓ No Stage 1 checkpoint found, training from scratch")
        stage1_map, _ = trainer.train_stage('stage1')

    # Stage 2
    stage2_map, _ = trainer.train_stage('stage2', start_epoch=0, best_map=stage1_map)

    print(f"\n✓ Training completed! Final mAP: {stage2_map:.4f}")


if __name__ == '__main__':
    main()