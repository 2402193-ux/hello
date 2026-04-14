
import os
import torch


class Config:
    # ========== 路径配置 ==========
    DATA_ROOT = 'D:/download/CholecT50'
    LEMON_WEIGHTS = 'D:/LEMON-main/LemonFM.pth'
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = './logs'
    PSEUDO_MASK_ROOT = 'D:/MedSAM-main/cholect50_masks_results'
    USE_PSEUDO_MASK = True  # 是否使用伪标签
    PSEUDO_MASK_WEIGHT = 1.0  # 空间权重的强度 [0.5~1.5]
    @staticmethod
    def get_rdv_official_splits():
        """
        RDV论文官方视频划分（cholect50）
        来源：https://github.com/CAMMA-public/cholect45
        """
        train = [1, 15, 26, 40, 52, 65, 79, 2, 18, 27, 43, 56, 66, 92,
                 4, 22, 31, 47, 57, 68, 96, 5, 23, 35, 48, 60, 70, 103,
                 13, 25, 36, 49, 62, 75, 110]
        val = [8, 12, 29, 50, 78]
        test = [6, 51, 10, 73, 14, 74, 32, 80, 42, 111]

        def to_vid_format(nums):
            return ['VID{}'.format(str(v).zfill(2)) for v in nums]

        return to_vid_format(train), to_vid_format(val), to_vid_format(test)

    # ========== 视频划分 ==========
    @staticmethod
    def get_available_videos():
        """扫描实际存在的视频文件夹"""
        data_dir = os.path.join(Config.DATA_ROOT, 'data')
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        all_videos = []
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path) and item.startswith('VID'):
                all_videos.append(item)

        all_videos.sort()
        print(f"\n✓ Found {len(all_videos)} videos: {all_videos[:10]}...")

        if len(all_videos) < 10:
            raise ValueError(f"Only found {len(all_videos)} videos, expected ~50")

        return all_videos

    @staticmethod
    def split_train_val_test(all_videos):
        """按照CholecT50标准划分: 35 train / 5 val / 10 test"""
        n_total = len(all_videos)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.1)

        train_videos = all_videos[:n_train]
        val_videos = all_videos[n_train:n_train+n_val]
        test_videos = all_videos[n_train+n_val:]

        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_videos)} videos")
        print(f"  验证集: {len(val_videos)} videos")
        print(f"  测试集: {len(test_videos)} videos")

        return train_videos, val_videos, test_videos

    ALL_VIDEOS = None
    TRAIN_VIDEOS = None
    VAL_VIDEOS = None
    TEST_VIDEOS = None

    @staticmethod
    def initialize():
        """初始化视频列表"""
        Config.ALL_VIDEOS = Config.get_available_videos()
        Config.TRAIN_VIDEOS, Config.VAL_VIDEOS, Config.TEST_VIDEOS = \
            Config.get_rdv_official_splits()

    # ========== 模型配置 ==========
    FEATURE_DIM = 1536
    HIDDEN_DIM = 512
    NUM_TRIPLETS = 100
    NUM_INSTRUMENTS = 6
    NUM_VERBS = 10
    NUM_TARGETS = 15

    # ========== LoRA训练配置（优化版） ==========
    # 🔥 针对RTX 4080 SUPER (16GB)优化
    BATCH_SIZE = 16          # LoRA允许更大的batch（Stage 2会用到）
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4     # Stage 2的默认学习率
    WEIGHT_DECAY = 1e-4

    FREEZE_BACKBONE = False  # LoRA会自动处理
    DROPOUT = 0.3
    NUM_WORKERS = 0         # 🔥 增加worker数量，加快数据加载

    # ========== 混合精度训练（节省内存，加速训练） ==========
    USE_AMP = True           # 🔥 强烈推荐开启

    # ========== 梯度累积（模拟更大batch） ==========
    ACCUMULATION_STEPS = 4   # 🔥 有效batch = BATCH_SIZE * ACCUMULATION_STEPS = 32

    # ========== 梯度裁剪（防止梯度爆炸） ==========
    MAX_GRAD_NORM = 1.0      # 🔥 推荐值

    # ========== 图像配置 ==========
    IMG_SIZE = 224
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    # ========== 其他配置 ==========
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42

    @staticmethod
    def create_dirs():
        """创建必要的目录"""
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        print(f"✓ Directories created")


# ========== LoRA专用配置 ==========
class LoRAConfig:
    # ========== LoRA超参数 ==========
    LORA_RANK = 8           # 🔥 推荐值：8
    LORA_ALPHA = 16         # 🔥 通常是rank的2倍
    LORA_DROPOUT = 0.1      # 🔥 防止过拟合


    # ========== LoRA目标层 ==========
    LORA_TARGET_MODULES = ['pwconv1', 'pwconv2']  # ConvNeXt的关键层

    # ========== 阶段配置 ==========
    STAGE1_CONFIG = {
        'name': 'Linear Probing',
        'train_lora': False,
        'learning_rate': 1e-3,
        'batch_size': 32,
        'max_epochs': 50,
        'patience': 10,
        'weight_decay': 1e-4
    }

    STAGE2_CONFIG = {
        'name': 'LoRA Fine-tuning',
        'train_lora': True,
        'learning_rate': 1e-4,
        'batch_size': 4,  # ← 从8改为4
        'max_epochs': 50,
        'patience': 3,
        'weight_decay': 1e-4
    }


# ========== 根据GPU调整配置 ==========
class GPUOptimizedConfig:

    RTX_4080_SUPER = {
        'lora_rank': 8,
        'batch_size_stage1': 32,
        'batch_size_stage2': 16,
        'accumulation_steps': 4,
        'use_amp': True,
        'num_workers': 4,
        'expected_memory': '5-7GB',
        'expected_time': '20-30 hours'
    }



# ========== 快速配置选择器 ==========
def select_config_for_gpu(gpu_name='RTX 4080 SUPER'):

    configs = {
        'RTX 4080 SUPER': GPUOptimizedConfig.RTX_4080_SUPER,
    }

    config = configs.get(gpu_name, GPUOptimizedConfig.RTX_4080_SUPER)
    print(f"\n配置已选择: {gpu_name}")
    print(f"  LoRA Rank: {config['lora_rank']}")
    print(f"  Batch Size (Stage 2): {config['batch_size_stage2']}")
    print(f"  预期内存占用: {config['expected_memory']}")
    print(f"  预期训练时间: {config['expected_time']}")
    print()

    return config


# ========== 使用示例 ==========
if __name__ == '__main__':
    print("="*70)
    print(" "*20 + "配置信息")
    print("="*70)
    print("\n当前配置:")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Learning Rate: {Config.LEARNING_RATE}")
    print(f"  Use AMP: {Config.USE_AMP}")
    print(f"  Accumulation Steps: {Config.ACCUMULATION_STEPS}")
    print(f"  Num Workers: {Config.NUM_WORKERS}")
    print("\nLoRA配置:")
    print(f"  Rank: {LoRAConfig.LORA_RANK}")
    print(f"  Alpha: {LoRAConfig.LORA_ALPHA}")
    print(f"  Dropout: {LoRAConfig.LORA_DROPOUT}")
    print(f"  Target Modules: {LoRAConfig.LORA_TARGET_MODULES}")
    print("\nStage 1 配置:")
    for key, value in LoRAConfig.STAGE1_CONFIG.items():
        print(f"  {key}: {value}")
    print("\nStage 2 配置:")
    for key, value in LoRAConfig.STAGE2_CONFIG.items():
        print(f"  {key}: {value}")
    print("\n" + "="*70)
    gpu_config = select_config_for_gpu('RTX 4080 SUPER')
    print("="*70)
    print("预期训练参数量:")
    print("="*70)
    rank = LoRAConfig.LORA_RANK
    feature_dim = 1536
    params_per_layer = rank * (feature_dim + feature_dim)  # A和B矩阵
    estimated_lora_layers = 30
    total_lora_params = params_per_layer * estimated_lora_layers
    classifier_params = feature_dim * Config.NUM_TRIPLETS
    total_trainable = total_lora_params + classifier_params
    print(f"  LoRA参数: {total_lora_params:,} ({total_lora_params/1e6:.2f}M)")
    print(f"  分类头参数: {classifier_params:,} ({classifier_params/1e6:.2f}M)")
    print(f"  总可训练参数: {total_trainable:,} ({total_trainable/1e6:.2f}M)")
    print(f"  参数减少比例: {(1 - total_trainable/196e6)*100:.1f}%")
    print("\n" + "="*70)
    print("✓ 配置检查完成")
    print("="*70 + "\n")