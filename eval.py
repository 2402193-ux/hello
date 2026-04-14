"""
评估脚本 - 支持详细指标
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from config import Config
from dataset import CholecT50Dataset
from model import SimpleTripletClassifier
from utils import compute_detailed_metrics, print_detailed_metrics

def main():
    print("\n" + "="*70)
    print(" "*20 + "TESTING ON TEST SET")
    print("="*70)

    # 确保视频列表已初始化
    if Config.TEST_VIDEOS is None:
        print("Initializing video lists...")
        Config.initialize()

    # 创建测试集
    print("\n📁 Loading test dataset...")
    test_dataset = CholecT50Dataset(Config.TEST_VIDEOS, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    print(f"✓ Test set: {len(test_dataset)} samples from {len(Config.TEST_VIDEOS)} videos")

    # 加载模型
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Loading model on {device}...")

    model = SimpleTripletClassifier().to(device)

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"✓ Model loaded from: {checkpoint_path}")
    print(f"  Trained for: {checkpoint['epoch']+1} epochs")
    print(f"  Best validation mAP: {checkpoint['best_map']*100:.2f}%")

    # 评估
    print("\n🔍 Evaluating...")
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)

            # 前向传播
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    # 合并所有batch
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    print(f"✓ Predictions collected: {all_labels.shape}")

    # 计算详细指标
    print("\n📊 Computing detailed metrics...")
    metrics = compute_detailed_metrics(all_labels, all_preds)

    # 打印详细结果
    print_detailed_metrics(metrics)

    # 保存结果
    save_path = os.path.join(Config.CHECKPOINT_DIR, 'test_metrics.txt')
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Test Set Evaluation Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"APi (Instrument):   {metrics['APi']*100:.2f}%\n")
        f.write(f"APv (Verb):         {metrics['APv']*100:.2f}%\n")
        f.write(f"APt (Target):       {metrics['APt']*100:.2f}%\n")
        f.write(f"APiv (I-V):         {metrics['APiv']*100:.2f}%\n")
        f.write(f"APit (I-T):         {metrics['APit']*100:.2f}%\n")
        f.write(f"APivt (I-V-T):      {metrics['APivt']*100:.2f}%\n")

    print(f"\n💾 Results saved to: {save_path}")

    print("\n" + "="*70)
    print(" "*25 + "DONE!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()