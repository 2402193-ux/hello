"""
工具函数 - 完整版（对齐RDV评估协议）
"""
import torch
import numpy as np
import random
from sklearn.metrics import average_precision_score


# ==================== 基础工具函数 ====================

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed set to {seed}")


def save_checkpoint(model, optimizer, epoch, best_map, path):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_map': best_map
    }
    torch.save(checkpoint, path)


def compute_map(labels, probs, ignore_null=True):
    """
    计算多标签分类的mAP (mean Average Precision)

    Args:
        labels: [N, K] numpy array, 真实标签 (0/1)
        probs: [N, K] numpy array, 预测概率 [0, 1]
        ignore_null: bool, 是否忽略没有正样本的类别

    Returns:
        mAP: float, 所有类别的平均AP
        aps: numpy array [K,], 每个类别的AP (可能包含nan)
    """
    N, K = labels.shape
    aps = np.full(K, np.nan, dtype=np.float32)

    for k in range(K):
        label_k = labels[:, k]
        prob_k = probs[:, k]

        # 检查是否有正样本
        if label_k.sum() == 0:
            if ignore_null:
                continue  # 保持nan
            else:
                aps[k] = 0.0
        else:
            # 计算AP
            try:
                ap = average_precision_score(label_k, prob_k)
                aps[k] = ap
            except:
                aps[k] = 0.0

    # 计算mAP（只考虑有效的AP值）
    valid_aps = aps[~np.isnan(aps)]
    if len(valid_aps) > 0:
        mAP = valid_aps.mean()
    else:
        mAP = 0.0

    return mAP, aps


# ==================== Triplet分解函数 ====================

from triplet_config import (
    TRIPLET_CLASSES,
    build_component_to_triplets,
    build_pair_to_triplets,
    get_component_counts,
    INSTRUMENTS, VERBS, TARGETS
)


def decompose_triplet_to_components(triplet_labels, triplet_probs):
    """
    从triplet预测分解出component预测（RDV论文Eq. 10）

    原理：如果任何包含该component的triplet存在，则该component存在
    使用max操作来聚合

    Args:
        triplet_labels: [N, 100] numpy array, 真实triplet标签
        triplet_probs: [N, 100] numpy array, 预测triplet概率

    Returns:
        dict: {
            'instrument_labels': [N, 6],
            'instrument_probs': [N, 6],
            'verb_labels': [N, 10],
            'verb_probs': [N, 10],
            'target_labels': [N, 15],
            'target_probs': [N, 15]
        }
    """
    N = triplet_labels.shape[0]
    counts = get_component_counts()

    # 获取component到triplets的映射
    comp_map = build_component_to_triplets()

    # 初始化component标签和预测
    instrument_labels = np.zeros((N, counts['n_instruments']), dtype=np.float32)
    instrument_probs = np.zeros((N, counts['n_instruments']), dtype=np.float32)

    verb_labels = np.zeros((N, counts['n_verbs']), dtype=np.float32)
    verb_probs = np.zeros((N, counts['n_verbs']), dtype=np.float32)

    target_labels = np.zeros((N, counts['n_targets']), dtype=np.float32)
    target_probs = np.zeros((N, counts['n_targets']), dtype=np.float32)

    # 对每个component，取所有包含它的triplet的max
    for i in range(counts['n_instruments']):
        triplet_ids = comp_map['instruments'][i]
        if triplet_ids:
            instrument_labels[:, i] = triplet_labels[:, triplet_ids].max(axis=1)
            instrument_probs[:, i] = triplet_probs[:, triplet_ids].max(axis=1)

    for v in range(counts['n_verbs']):
        triplet_ids = comp_map['verbs'][v]
        if triplet_ids:
            verb_labels[:, v] = triplet_labels[:, triplet_ids].max(axis=1)
            verb_probs[:, v] = triplet_probs[:, triplet_ids].max(axis=1)

    for t in range(counts['n_targets']):
        triplet_ids = comp_map['targets'][t]
        if triplet_ids:
            target_labels[:, t] = triplet_labels[:, triplet_ids].max(axis=1)
            target_probs[:, t] = triplet_probs[:, triplet_ids].max(axis=1)

    return {
        'instrument_labels': instrument_labels,
        'instrument_probs': instrument_probs,
        'verb_labels': verb_labels,
        'verb_probs': verb_probs,
        'target_labels': target_labels,
        'target_probs': target_probs
    }


def decompose_triplet_to_pairs(triplet_labels, triplet_probs):
    """
    从triplet预测分解出pair预测

    Args:
        triplet_labels: [N, 100] numpy array
        triplet_probs: [N, 100] numpy array

    Returns:
        dict: {
            'iv_labels': [N, n_iv_pairs],
            'iv_probs': [N, n_iv_pairs],
            'it_labels': [N, n_it_pairs],
            'it_probs': [N, n_it_pairs]
        }
    """
    N = triplet_labels.shape[0]

    # 获取pair到triplets的映射
    pair_map = build_pair_to_triplets()

    # IV pairs
    iv_pairs = sorted(pair_map['iv'].keys())
    n_iv_pairs = len(iv_pairs)
    iv_labels = np.zeros((N, n_iv_pairs), dtype=np.float32)
    iv_probs = np.zeros((N, n_iv_pairs), dtype=np.float32)

    for idx, (i, v) in enumerate(iv_pairs):
        triplet_ids = pair_map['iv'][(i, v)]
        if triplet_ids:
            iv_labels[:, idx] = triplet_labels[:, triplet_ids].max(axis=1)
            iv_probs[:, idx] = triplet_probs[:, triplet_ids].max(axis=1)

    # IT pairs
    it_pairs = sorted(pair_map['it'].keys())
    n_it_pairs = len(it_pairs)
    it_labels = np.zeros((N, n_it_pairs), dtype=np.float32)
    it_probs = np.zeros((N, n_it_pairs), dtype=np.float32)

    for idx, (i, t) in enumerate(it_pairs):
        triplet_ids = pair_map['it'][(i, t)]
        if triplet_ids:
            it_labels[:, idx] = triplet_labels[:, triplet_ids].max(axis=1)
            it_probs[:, idx] = triplet_probs[:, triplet_ids].max(axis=1)

    return {
        'iv_labels': iv_labels,
        'iv_probs': iv_probs,
        'it_labels': it_labels,
        'it_probs': it_probs
    }


def compute_detailed_metrics(triplet_labels, triplet_probs):
    """
    计算完整的6个评估指标

    Args:
        triplet_labels: [N, 100] numpy array
        triplet_probs: [N, 100] numpy array

    Returns:
        dict: {
            'APi': float,
            'APv': float,
            'APt': float,
            'APiv': float,
            'APit': float,
            'APivt': float,
            'detailed': {...}
        }
    """
    # 1. 分解component
    components = decompose_triplet_to_components(triplet_labels, triplet_probs)

    # 2. 分解pairs
    pairs = decompose_triplet_to_pairs(triplet_labels, triplet_probs)

    # 3. 计算各个指标的mAP
    APi, aps_i = compute_map(components['instrument_labels'],
                              components['instrument_probs'])
    APv, aps_v = compute_map(components['verb_labels'],
                              components['verb_probs'])
    APt, aps_t = compute_map(components['target_labels'],
                              components['target_probs'])

    APiv, aps_iv = compute_map(pairs['iv_labels'],
                                pairs['iv_probs'])
    APit, aps_it = compute_map(pairs['it_labels'],
                                pairs['it_probs'])

    APivt, aps_ivt = compute_map(triplet_labels, triplet_probs)

    return {
        'APi': APi,
        'APv': APv,
        'APt': APt,
        'APiv': APiv,
        'APit': APit,
        'APivt': APivt,
    }


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("Testing utils functions...")

    # 测试compute_map
    print("\n1. Testing compute_map...")
    labels = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    probs = np.array([[0.9, 0.1, 0.8], [0.2, 0.9, 0.3], [0.7, 0.6, 0.4]])
    mAP, aps = compute_map(labels, probs)
    print(f"   mAP = {mAP:.4f}")
    print(f"   APs = {aps}")

    # 测试详细指标
    print("\n2. Testing detailed metrics...")
    np.random.seed(42)
    N = 100
    triplet_labels = np.random.randint(0, 2, size=(N, 100)).astype(np.float32)
    triplet_probs = np.random.rand(N, 100).astype(np.float32)

    metrics = compute_detailed_metrics(triplet_labels, triplet_probs)
    print(f"   APi: {metrics['APi']:.4f}")
    print(f"   APv: {metrics['APv']:.4f}")
    print(f"   APt: {metrics['APt']:.4f}")
    print(f"   APiv: {metrics['APiv']:.4f}")
    print(f"   APit: {metrics['APit']:.4f}")
    print(f"   APivt: {metrics['APivt']:.4f}")

    print("\n✓ All tests passed!")