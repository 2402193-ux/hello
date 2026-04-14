"""
数据集加载器 - 与LemonFM完全对齐
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from config import Config
import numpy as np
class CholecT50Dataset(Dataset):

    def get_all_video_ids(self):
        """返回数据集中所有唯一的video ID"""
        return sorted(list(set([s['video_id'] for s in self.samples])))

    # 找到 get_video_dataset 方法，替换整个 VideoSubset 类

    def get_video_dataset(self, video_id):
        """为指定video创建子数据集（用于Video-level evaluation）"""
        video_samples = [s for s in self.samples if s['video_id'] == video_id]

        if not video_samples:
            raise ValueError(f"No samples found for video_id: {video_id}")

        parent_dataset = self

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
                pseudo_mask = parent_dataset._load_pseudo_mask(
                    sample['video_id'],
                    sample['frame']
                )

                return image, label, sample['video_id'], pseudo_mask  # ← 返回4个值

        return VideoSubset(video_samples, self.transform)


    def __init__(self, video_list, is_train=True):
        self.video_list = video_list
        self.is_train = is_train
        self.samples = []
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=(256, 448),  # 目标尺寸
                    scale=(0.5, 1.5),  # ✨ RDV标准：random scaling范围
                    ratio=(448 / 256, 448 / 256),  # 保持16:9宽高比
                    interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,  # ±20%
                    contrast=0.2,  # ±20%
                    saturation=0.0,  # 不改变饱和度
                    hue=0.0  # 不改变色调
                ),

                transforms.ToTensor(),
                normalize
            ])

            print("✓ Train augmentation (RDV standard):")
            print("  - RandomResizedCrop: size=(256,448), scale=(0.5,1.5)")
            print("  - RandomHorizontalFlip: p=0.5")
            print("  - ColorJitter: brightness=0.1, contrast=0.1")
        else:
            # 验证/测试时：只做固定resize和归一化
            self.transform = transforms.Compose([
                transforms.Resize((256, 448)),
                transforms.ToTensor(),
                normalize
            ])

            print("✓ Val/Test augmentation:")
            print("  - Resize: (256, 448)")
            print("  - No augmentation")

        self._load_samples()

    def _load_samples(self):
        """加载所有图像和标签路径"""
        total_samples = 0
        total_positives = 0
        triplet_counts = [0] * Config.NUM_TRIPLETS

        for video_id in self.video_list:
            img_dir = os.path.join(Config.DATA_ROOT, 'data', video_id)
            triplet_file = os.path.join(Config.DATA_ROOT, 'triplet', f'{video_id}.txt')

            if not os.path.exists(img_dir):
                print(f"Warning: {img_dir} not found, skipping...")
                continue
            if not os.path.exists(triplet_file):
                print(f"Warning: {triplet_file} not found, skipping...")
                continue

            # 读取triplet标签
            with open(triplet_file, 'r') as f:
                triplet_labels = f.readlines()

            # 获取所有图像文件（按文件名排序）
            img_files = sorted([f for f in os.listdir(img_dir)
                              if f.endswith(('.png', '.jpg', '.jpeg'))])

            # 确保数量匹配
            if len(img_files) != len(triplet_labels):
                print(f"Warning: {video_id} has {len(img_files)} images but "
                      f"{len(triplet_labels)} labels, using minimum")
                min_len = min(len(img_files), len(triplet_labels))
                img_files = img_files[:min_len]
                triplet_labels = triplet_labels[:min_len]

            # 添加样本
            video_samples = 0
            for img_file, label_line in zip(img_files, triplet_labels):
                img_path = os.path.join(img_dir, img_file)

                # 解析标签
                values = label_line.strip().split(',')

                # 跳过第一列（帧号）
                if len(values) == 101:
                    label = [int(x) for x in values[1:]]
                elif len(values) == 100:
                    label = [int(x) for x in values]
                else:
                    print(f"Warning: {video_id}/{img_file} has {len(values)} values, skipping...")
                    continue

                if len(label) != Config.NUM_TRIPLETS:
                    print(f"Warning: {video_id}/{img_file} label length {len(label)}, skipping...")
                    continue

                self.samples.append({
                    'image': img_path,
                    'label': label,
                    'video_id': video_id,
                    'frame': img_file
                })

                video_samples += 1
                total_positives += sum(label)

                for i, val in enumerate(label):
                    if val == 1:
                        triplet_counts[i] += 1

            total_samples += video_samples
            print(f"  {video_id}: {video_samples} samples")

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found!")

        print(f"\n✓ Loaded {len(self.samples)} samples from {len(self.video_list)} videos")
        print(f"✓ Avg positive labels per frame: {total_positives/len(self.samples):.2f}")

        # 统计稀有triplet
        rare_triplets = [i for i, count in enumerate(triplet_counts) if count < 10]
        if rare_triplets:
            print(f"⚠️  {len(rare_triplets)} triplets have <10 samples")

        zero_triplets = [i for i, count in enumerate(triplet_counts) if count == 0]
        if zero_triplets:
            print(f"⚠️  {len(zero_triplets)} triplets have 0 samples: {zero_triplets}")

    def __len__(self):
        return len(self.samples)

    def _load_pseudo_mask(self, video_id, frame_name):
        """
        加载伪标签mask（如果存在）

        Args:
            video_id: 如 'VID01'
            frame_name: 如 '000001.png'

        Returns:
            mask: [1, 256, 448] tensor，如果不存在则返回全1
        """
        if not Config.USE_PSEUDO_MASK:
            return torch.zeros(1, 256, 448, dtype=torch.float32)

        # 构建路径
        video_num = video_id.lower().replace('vid', 'video')  # VID01 -> video01
        frame_num = frame_name.split('.')[0]  # 000001.png -> 000001

        mask_path = os.path.join(
            Config.PSEUDO_MASK_ROOT,
            video_num,
            'cleaned_masks',
            f'{frame_num}_cleaned_masks.npy'
        )

        try:
            # 加载npy文件
            mask = np.load(mask_path)  # 可能是 [H, W] 或 [H, W, C] 或其他

            # 🔥 统一处理维度
            if mask.ndim == 3:
                # [H, W, C] -> 取第一个通道
                mask = mask[:, :, 0]
            elif mask.ndim != 2:
                # 其他奇怪的维度，返回全1
                print(f"⚠️ Unexpected mask shape {mask.shape} for {video_id}/{frame_name}, using all-ones")
                return torch.ones(1, 256, 448, dtype=torch.float32)

            # 现在 mask 是 [H, W]

            # 转为tensor并归一化到[0, 1]
            mask = torch.from_numpy(mask).float()
            if mask.max() > 1.0:
                mask = mask / 255.0

            # 🔥 正确的resize流程
            # [H, W] -> [1, 1, H, W] -> interpolate -> [1, 1, 256, 448] -> [1, 256, 448]
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            mask = torch.nn.functional.interpolate(
                mask,
                size=(256, 448),
                mode='nearest'
            )
            mask = mask.squeeze(0)  # [1, 256, 448]

            # 🔥 最终验证
            assert mask.shape == (1, 256, 448), f"Final mask shape {mask.shape} != (1, 256, 448)"

            return mask

        except FileNotFoundError:
            # 如果文件不存在，返回全1 mask
            return torch.zeros(1, 256, 448, dtype=torch.float32)
        except Exception as e:
            print(f"⚠️ Failed to load mask for {video_id}/{frame_name}: {e}")
            return torch.zeros(1, 256, 448, dtype=torch.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像
        image = Image.open(sample['image']).convert('RGB')
        image = self.transform(image)

        # 转换标签
        label = torch.tensor(sample['label'], dtype=torch.float32)

        # 🔥 加载伪标签
        pseudo_mask = self._load_pseudo_mask(sample['video_id'], sample['frame'])

        # 🔥 验证维度
        assert image.shape == (3, 256, 448), f"Image shape error: {image.shape}"
        assert label.shape == (100,), f"Label shape error: {label.shape}"
        assert pseudo_mask.shape == (
        1, 256, 448), f"Mask shape error: {pseudo_mask.shape} at {sample['video_id']}/{sample['frame']}"

        return image, label, sample['video_id'], pseudo_mask