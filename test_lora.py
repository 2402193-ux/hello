"""
训练后模型测试脚本

用途：
1. 加载训练好的权重
2. 在测试集上评估
3. 计算完整的6个指标
4. 与RDV论文对比
"""
import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast

from config import Config
from dataset import CholecT50Dataset
from model_lora import SimpleTripletClassifierLoRA
from utils import compute_map, decompose_triplet_to_components, decompose_triplet_to_pairs


class ModelTester:
    """训练后模型测试器"""

    def __init__(self, checkpoint_path, use_test_set=True):
        """
        Args:
            checkpoint_path: 权重文件路径（通常是stage2_best_lora.pth）
            use_test_set: True=测试集, False=验证集
        """
        self.device = torch.device(Config.DEVICE)
        self.checkpoint_path = checkpoint_path

        # 选择评估集
        if use_test_set:
            self.eval_videos = Config.TEST_VIDEOS
            self.dataset_name = "Test Set"
        else:
            self.eval_videos = Config.VAL_VIDEOS
            self.dataset_name = "Validation Set"

        print(f"\n{'='*70}")
        print(f"{'Model Testing':^70}")
        print(f"{'='*70}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Dataset: {self.dataset_name} ({len(self.eval_videos)} videos)")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")

        # 加载模型
        self.model = self._load_model()

        # 创建数据集
        self.dataset = CholecT50Dataset(self.eval_videos, is_train=False)
        print(f"Total samples: {len(self.dataset):,}\n")

    def _load_model(self):
        """加载训练好的模型"""
        print("Loading model...")

        # 创建模型
        model = SimpleTripletClassifierLoRA(
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.1
        ).to(self.device)

        # 加载权重
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
        print(f"✓ Best training mAP: {checkpoint['best_map']:.4f}\n")

        return model

    def evaluate(self):
        """
        完整评估（对齐RDV协议）

        Returns:
            dict: 包含所有6个指标的字典
        """
        print(f"{'='*70}")
        print(f"{'EVALUATION (RDV Protocol)':^70}")
        print(f"{'='*70}\n")

        print(f"Evaluating on {len(self.eval_videos)} videos...\n")

        # 获取所有video IDs
        video_ids = self.dataset.get_all_video_ids()

        # 为每个video创建独立的DataLoader
        video_dataloaders = []
        for video_id in video_ids:
            video_dataset = self.dataset.get_video_dataset(video_id)
            video_loader = torch.utils.data.DataLoader(
                video_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True
            )
            video_dataloaders.append((video_id, video_loader))

        # 对每个video分别计算指标
        video_metrics = {
            'APi': [], 'APv': [], 'APt': [],
            'APiv': [], 'APit': [], 'APivt': []
        }

        with torch.no_grad():
            for video_id, video_loader in video_dataloaders:
                video_labels = []
                video_preds = []

                for batch_data in tqdm(video_loader, desc=f"  {video_id}", leave=False):
                    # ← 修改这里，接收4个值
                    images, labels, _, pseudo_masks = batch_data
                    images = images.to(self.device)
                    pseudo_masks = pseudo_masks.to(self.device)

                    if Config.USE_AMP:
                        with autocast():
                            outputs = self.model(images, pseudo_mask=pseudo_masks)  # ← 传入mask
                            logits = outputs['triplets']
                    else:
                        outputs = self.model(images, pseudo_mask=pseudo_masks)  # ← 传入mask
                        logits = outputs['triplets']

                    probs = torch.sigmoid(logits)
                    video_labels.append(labels.cpu().numpy())
                    video_preds.append(probs.cpu().numpy())

                # 合并该video的所有frame
                video_labels = np.concatenate(video_labels, axis=0)
                video_preds = np.concatenate(video_preds, axis=0)

                # 计算该video的triplet AP
                video_ap_ivt, _ = compute_map(video_labels, video_preds)

                # 分解triplet到components和pairs
                components = decompose_triplet_to_components(video_labels, video_preds)
                pairs = decompose_triplet_to_pairs(video_labels, video_preds)

                # 计算该video的所有指标
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

                # 保存该video的结果
                video_metrics['APi'].append(video_ap_i)
                video_metrics['APv'].append(video_ap_v)
                video_metrics['APt'].append(video_ap_t)
                video_metrics['APiv'].append(video_ap_iv)
                video_metrics['APit'].append(video_ap_it)
                video_metrics['APivt'].append(video_ap_ivt)

                print(f"  {video_id}: APivt={video_ap_ivt:.3f} ({len(video_labels):4d} frames)")

        # 对所有video的AP求平均（Video-level averaging）
        final_metrics = {
            key: np.mean(values) for key, values in video_metrics.items()
        }

        # 打印结果
        self._print_results(final_metrics, video_metrics)

        return final_metrics

    def _print_results(self, final_metrics, video_metrics):
        """打印评估结果"""
        print(f"\n{'='*70}")
        print(f"{'VIDEO-LEVEL RESULTS':^70}")
        print(f"{'='*70}")
        print(f"(averaged over {len(self.eval_videos)} videos)\n")

        print("📊 Component Detection:")
        print("-"*70)
        print(f"  APi (Instrument):     {final_metrics['APi']*100:6.2f}%")
        print(f"  APv (Verb):           {final_metrics['APv']*100:6.2f}%")
        print(f"  APt (Target):         {final_metrics['APt']*100:6.2f}%")

        print("\n🔗 Triplet Association:")
        print("-"*70)
        print(f"  APiv (I-V pairs):     {final_metrics['APiv']*100:6.2f}%")
        print(f"  APit (I-T pairs):     {final_metrics['APit']*100:6.2f}%")

        print("\n⭐ Main Metric:")
        print("-"*70)
        print(f"  APivt (I-V-T):        {final_metrics['APivt']*100:6.2f}%  ← MAIN METRIC")

        # 统计信息
        print(f"\n📈 Statistics (APivt across videos):")
        apivt_values = video_metrics['APivt']
        print(f"  Min:  {min(apivt_values)*100:.2f}%")
        print(f"  Max:  {max(apivt_values)*100:.2f}%")
        print(f"  Std:  {np.std(apivt_values)*100:.2f}%")

        # 与RDV对比
        print(f"\n📖 Comparison to RDV (SOTA):")
        print("-"*70)
        rdv_results = {
            'APi': 0.920, 'APv': 0.607, 'APt': 0.383,
            'APiv': 0.394, 'APit': 0.369, 'APivt': 0.299
        }

        print(f"  {'Metric':<8} {'Yours':>8} {'RDV':>8} {'Diff':>8}")
        print("-"*70)
        for key in ['APi', 'APv', 'APt', 'APiv', 'APit', 'APivt']:
            yours = final_metrics[key]
            rdv = rdv_results[key]
            diff = yours - rdv
            marker = " ⭐" if key == 'APivt' else ""
            print(f"  {key:<8} {yours*100:7.2f}% {rdv*100:7.2f}% "
                  f"{diff*100:+7.2f}pp{marker}")

        print(f"\n{'='*70}\n")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Test trained LoRA model')
    parser.add_argument('--checkpoint', type=str,
                       default='./checkpoints/stage2_best_lora.pth',
                       help='Path to checkpoint file')
    parser.add_argument('--stage', type=str, default='stage2',
                       choices=['stage1', 'stage2'],
                       help='Which stage checkpoint to test')
    parser.add_argument('--split', type=str, default='test',
                       choices=['val', 'test'],
                       help='Evaluate on validation or test set')

    args = parser.parse_args()

    # 初始化配置
    Config.initialize()

    # 确定checkpoint路径
    if args.checkpoint == './checkpoints/stage2_best_lora.pth':
        checkpoint_path = f'./checkpoints/{args.stage}_best_lora.pth'
    else:
        checkpoint_path = args.checkpoint

    # 创建测试器
    use_test_set = (args.split == 'test')
    tester = ModelTester(checkpoint_path, use_test_set=use_test_set)

    # 评估
    metrics = tester.evaluate()

    # 保存结果
    import json
    from datetime import datetime

    result_file = f'{args.stage}_{args.split}_results.json'
    results = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': checkpoint_path,
        'dataset': args.split,
        'num_videos': len(tester.eval_videos),
        'metrics': {k: float(v) for k, v in metrics.items()}
    }

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to {result_file}")


if __name__ == '__main__':
    main()