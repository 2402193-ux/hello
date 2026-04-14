"""
Dataset for PG-HCN - 支持Phase信息的数据集

特点:
1. 继承原dataset.py的核心功能
2. 添加Phase标注支持
3. 正确处理无Phase标注的视频 (VID92, VID96, VID103, VID110, VID111)
4. 支持"unknown phase"模式
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from config import Config
from typing import Dict, Set, List, Optional

# ==================== Phase配置 ====================

# 无Phase标注的视频（不在Cholec80中）
NO_PHASE_ANNOTATION_VIDEOS = {'VID92', 'VID96', 'VID103', 'VID110', 'VID111'}

# Phase ID定义
PHASE_UNKNOWN = 7  # 使用7作为"unknown phase"（正常phase是0-6）
NUM_PHASES = 8  # 0-6 + unknown


# ==================== 主数据集类 ====================

class CholecT50DatasetPGHCN(Dataset):
    """
    CholecT50数据集 - PG-HCN版本

    返回: (image, label, video_id, pseudo_mask, phase_id)
    """

    def __init__(
            self,
            video_list: List[str],
            is_train: bool = True,
            phase_dir: Optional[str] = None,
            include_no_phase_videos: bool = True
    ):
        self.video_list = video_list
        self.is_train = is_train
        self.include_no_phase_videos = include_no_phase_videos

        # Phase目录
        if phase_dir is None:
            self.phase_dir = os.path.join(Config.DATA_ROOT, 'phase')
        else:
            self.phase_dir = phase_dir

        # 样本列表
        self.samples = []

        # Phase缓存
        self.phase_cache: Dict[str, Dict[str, int]] = {}

        # 统计信息
        self.videos_with_phase: Set[str] = set()
        self.videos_without_phase: Set[str] = set()

        # 设置transforms
        self._setup_transforms()

        # 加载样本
        self._load_samples()

        # 加载Phase标注
        self._load_phase_annotations()

        # 如果不包含无标注视频，过滤样本
        if not include_no_phase_videos:
            self._filter_samples_without_phase()

    def _setup_transforms(self):
        """设置图像变换"""
        normalize = transforms.Normalize(
            mean=Config.IMG_MEAN,
            std=Config.IMG_STD
        )

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=(256, 448),
                    scale=(0.5, 1.5),
                    ratio=(448 / 256, 448 / 256),
                    interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.0,
                    hue=0.0
                ),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 448)),
                transforms.ToTensor(),
                normalize
            ])

    def _load_samples(self):
        """加载所有图像和标签"""
        print(f"\n{'=' * 60}")
        print(f"Loading CholecT50 Dataset (PG-HCN)")
        print(f"{'=' * 60}")

        total_samples = 0

        for video_id in self.video_list:
            img_dir = os.path.join(Config.DATA_ROOT, 'data', video_id)
            triplet_file = os.path.join(Config.DATA_ROOT, 'triplet', f'{video_id}.txt')

            if not os.path.exists(img_dir):
                print(f"  ⚠️  {video_id}: Image dir not found, skipping")
                continue
            if not os.path.exists(triplet_file):
                print(f"  ⚠️  {video_id}: Triplet file not found, skipping")
                continue

            # 读取triplet标签
            with open(triplet_file, 'r') as f:
                triplet_labels = f.readlines()

            # 获取图像文件
            img_files = sorted([f for f in os.listdir(img_dir)
                                if f.endswith(('.png', '.jpg', '.jpeg'))])

            # 确保数量匹配
            min_len = min(len(img_files), len(triplet_labels))
            img_files = img_files[:min_len]
            triplet_labels = triplet_labels[:min_len]

            # 添加样本
            video_samples = 0
            for img_file, label_line in zip(img_files, triplet_labels):
                img_path = os.path.join(img_dir, img_file)

                # 解析标签
                values = label_line.strip().split(',')
                if len(values) == 101:
                    label = [int(x) for x in values[1:]]
                elif len(values) == 100:
                    label = [int(x) for x in values]
                else:
                    continue

                if len(label) != Config.NUM_TRIPLETS:
                    continue

                self.samples.append({
                    'image': img_path,
                    'label': label,
                    'video_id': video_id,
                    'frame': img_file
                })
                video_samples += 1

            total_samples += video_samples
            print(f"  ✓  {video_id}: {video_samples} samples")

        print(f"\n📊 Total: {len(self.samples)} samples from {len(self.video_list)} videos")

    def _load_phase_annotations(self):
        """加载Phase标注"""
        print(f"\n{'=' * 60}")
        print(f"Loading Phase Annotations")
        print(f"{'=' * 60}")

        all_video_ids = set([s['video_id'] for s in self.samples])

        for video_id in sorted(all_video_ids):
            # 检查是否是已知无标注视频
            if video_id in NO_PHASE_ANNOTATION_VIDEOS:
                self.videos_without_phase.add(video_id)
                self.phase_cache[video_id] = None
                print(f"  ⚠️  {video_id}: No phase annotation (not in Cholec80)")
                continue

            # 尝试加载Phase文件
            phase_file = os.path.join(self.phase_dir, f'{video_id}_phase.txt')

            if os.path.exists(phase_file):
                phase_dict = self._parse_phase_file(phase_file)

                if phase_dict:
                    self.phase_cache[video_id] = phase_dict
                    self.videos_with_phase.add(video_id)
                    print(f"  ✓  {video_id}: {len(phase_dict)} frames with phase")
                else:
                    self.videos_without_phase.add(video_id)
                    self.phase_cache[video_id] = None
                    print(f"  ⚠️  {video_id}: Empty phase file")
            else:
                self.videos_without_phase.add(video_id)
                self.phase_cache[video_id] = None
                print(f"  ⚠️  {video_id}: Phase file not found")

        print(f"\n📊 Phase Statistics:")
        print(f"   With phase:    {len(self.videos_with_phase)} videos")
        print(f"   Without phase: {len(self.videos_without_phase)} videos")
        if self.videos_without_phase:
            print(f"   Unknown videos: {sorted(self.videos_without_phase)}")

    def _parse_phase_file(self, phase_file: str) -> Dict[str, int]:
        """解析Phase标注文件 (格式: frame_id phase_id)"""
        phase_dict = {}

        try:
            with open(phase_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) >= 2:
                        frame_id = parts[0]  # '000000'
                        phase_id = int(parts[1])  # 0-6

                        if 0 <= phase_id <= 6:
                            phase_dict[frame_id] = phase_id
        except Exception as e:
            print(f"    Error parsing {phase_file}: {e}")
            return {}

        return phase_dict

    def _filter_samples_without_phase(self):
        """过滤掉无Phase标注视频的样本"""
        original_count = len(self.samples)

        self.samples = [
            s for s in self.samples
            if s['video_id'] in self.videos_with_phase
        ]

        filtered_count = original_count - len(self.samples)
        print(f"\n⚠️  Filtered {filtered_count} samples without phase")
        print(f"   Remaining: {len(self.samples)} samples")

    def _get_phase_id(self, video_id: str, frame_name: str) -> int:
        """获取Phase ID，无标注返回PHASE_UNKNOWN"""
        frame_id = frame_name.split('.')[0]  # '000001.png' -> '000001'

        phase_dict = self.phase_cache.get(video_id)

        if phase_dict is None:
            return PHASE_UNKNOWN

        return phase_dict.get(frame_id, PHASE_UNKNOWN)

    def _load_pseudo_mask(self, video_id: str, frame_name: str) -> torch.Tensor:
        """加载伪标签mask"""
        if not Config.USE_PSEUDO_MASK:
            return torch.zeros(1, 256, 448, dtype=torch.float32)

        video_num = video_id.lower().replace('vid', 'video')
        frame_num = frame_name.split('.')[0]

        mask_path = os.path.join(
            Config.PSEUDO_MASK_ROOT,
            video_num,
            'cleaned_masks',
            f'{frame_num}_cleaned_masks.npy'
        )

        try:
            mask = np.load(mask_path)

            if mask.ndim == 3:
                mask = mask[:, :, 0]
            elif mask.ndim != 2:
                return torch.zeros(1, 256, 448, dtype=torch.float32)

            mask = torch.from_numpy(mask).float()
            if mask.max() > 1.0:
                mask = mask / 255.0

            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = torch.nn.functional.interpolate(
                mask, size=(256, 448), mode='nearest'
            )
            mask = mask.squeeze(0)

            return mask

        except FileNotFoundError:
            return torch.zeros(1, 256, 448, dtype=torch.float32)
        except Exception as e:
            return torch.zeros(1, 256, 448, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像
        image = Image.open(sample['image']).convert('RGB')
        image = self.transform(image)

        # 标签
        label = torch.tensor(sample['label'], dtype=torch.float32)

        # 伪标签mask
        pseudo_mask = self._load_pseudo_mask(sample['video_id'], sample['frame'])

        # Phase ID
        phase_id = self._get_phase_id(sample['video_id'], sample['frame'])
        phase_id = torch.tensor(phase_id, dtype=torch.long)

        return image, label, sample['video_id'], pseudo_mask, phase_id

    # ==================== Video-level evaluation ====================

    def get_all_video_ids(self) -> List[str]:
        """获取所有video ID"""
        return sorted(list(set([s['video_id'] for s in self.samples])))

    def get_video_dataset(self, video_id: str):
        """为指定video创建子数据集"""
        video_samples = [s for s in self.samples if s['video_id'] == video_id]

        if not video_samples:
            raise ValueError(f"No samples found for video_id: {video_id}")

        parent = self

        class VideoSubset(Dataset):
            def __init__(self, samples, transform):
                self.samples = samples
                self.transform = transform

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                sample = self.samples[idx]

                image = Image.open(sample['image']).convert('RGB')
                image = self.transform(image)

                label = torch.tensor(sample['label'], dtype=torch.float32)

                pseudo_mask = parent._load_pseudo_mask(
                    sample['video_id'], sample['frame']
                )

                phase_id = parent._get_phase_id(
                    sample['video_id'], sample['frame']
                )
                phase_id = torch.tensor(phase_id, dtype=torch.long)

                return image, label, sample['video_id'], pseudo_mask, phase_id

        return VideoSubset(video_samples, self.transform)

    def get_phase_distribution(self) -> Dict[int, int]:
        """统计Phase分布"""
        dist = {i: 0 for i in range(NUM_PHASES)}

        for sample in self.samples:
            phase_id = self._get_phase_id(sample['video_id'], sample['frame'])
            dist[phase_id] += 1

        return dist


# ==================== 测试 ====================

if __name__ == '__main__':
    print("=" * 70)
    print(" CholecT50DatasetPGHCN Test")
    print("=" * 70)

    # 初始化Config
    Config.initialize()

    # 测试训练集
    print("\n[Training Set]")
    train_dataset = CholecT50DatasetPGHCN(
        Config.TRAIN_VIDEOS,
        is_train=True,
        include_no_phase_videos=True
    )

    print(f"\nTotal samples: {len(train_dataset)}")

    # 测试Phase分布
    phase_dist = train_dataset.get_phase_distribution()
    print("\nPhase Distribution:")
    for phase_id, count in phase_dist.items():
        phase_name = f"Phase {phase_id}" if phase_id < 7 else "Unknown"
        print(f"  {phase_name}: {count} samples ({count / len(train_dataset) * 100:.1f}%)")

    # 测试单个样本
    print("\n[Sample Test]")
    image, label, video_id, pseudo_mask, phase_id = train_dataset[0]
    print(f"  Image shape: {image.shape}")
    print(f"  Label shape: {label.shape}")
    print(f"  Video ID: {video_id}")
    print(f"  Pseudo mask shape: {pseudo_mask.shape}")
    print(f"  Phase ID: {phase_id.item()}")

    print("\n" + "=" * 70)
    print(" ✓ Test completed!")
    print("=" * 70)