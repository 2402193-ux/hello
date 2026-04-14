"""
Phase-Triplet 关联分析 - RDV官方顺序版
输出的triplet_id与triplet_config.py中的RDV官方顺序一致
"""

import os
import re
from collections import defaultdict, Counter
import csv

# ============ 配置路径 ============
CHOLECT50_PATH = r"D:\download\CholecT50"
CHOLEC80_PHASE_PATH = r"D:\EndoViT-main\datasets\Cholec80\phase_annotations"

# ============ Phase定义 ============
PHASE_NAME_TO_ID = {
    'preparation': 0,
    'calottriangledissection': 1,
    'clippingcutting': 2,
    'gallbladderdissection': 3,
    'gallbladderpackaging': 4,
    'cleaningcoagulation': 5,
    'gallbladderretraction': 6,
}

PHASE_NAMES = {
    0: 'Preparation',
    1: 'Calot Triangle Dissection',
    2: 'Clipping and Cutting',
    3: 'Gallbladder Dissection',
    4: 'Gallbladder Packaging',
    5: 'Cleaning and Coagulation',
    6: 'Gallbladder Retraction'
}

# ============ CholecT50原生编号系统 ============
# 这是数据集triplet标注文件使用的编号
CHOLECT50_INSTRUMENTS = {
    0: 'grasper', 1: 'bipolar', 2: 'hook',
    3: 'scissors', 4: 'clipper', 5: 'irrigator'
}

CHOLECT50_VERBS = {
    0: 'grasp', 1: 'retract', 2: 'dissect', 3: 'coagulate',
    4: 'clip', 5: 'cut', 6: 'aspirate', 7: 'irrigate',
    8: 'pack', 9: 'null_verb'
}

CHOLECT50_TARGETS = {
    0: 'gallbladder', 1: 'cystic_plate', 2: 'cystic_duct',
    3: 'cystic_artery', 4: 'cystic_pedicle', 5: 'blood_vessel',
    6: 'fluid', 7: 'abdominal_wall_cavity', 8: 'liver',
    9: 'adhesion', 10: 'omentum', 11: 'peritoneum',
    12: 'gut', 13: 'specimen_bag', 14: 'null_target'
}

# CholecT50原生triplet编号 -> (inst_id, verb_id, target_id)
CHOLECT50_TRIPLET_MAP = {
    0: (0, 0, 0), 1: (0, 0, 1), 2: (0, 0, 2), 3: (0, 0, 8), 4: (0, 0, 9),
    5: (0, 0, 10), 6: (0, 0, 11), 7: (0, 0, 13), 8: (0, 1, 0), 9: (0, 1, 1),
    10: (0, 1, 2), 11: (0, 1, 8), 12: (0, 1, 9), 13: (0, 1, 10), 14: (0, 1, 11),
    15: (0, 2, 0), 16: (0, 2, 1), 17: (0, 2, 2), 18: (0, 2, 9), 19: (0, 2, 10),
    20: (0, 2, 11), 21: (0, 8, 0), 22: (0, 8, 13), 23: (0, 9, 14), 24: (1, 2, 0),
    25: (1, 2, 1), 26: (1, 2, 2), 27: (1, 2, 3), 28: (1, 2, 4), 29: (1, 2, 9),
    30: (1, 2, 10), 31: (1, 2, 11), 32: (1, 3, 0), 33: (1, 3, 1), 34: (1, 3, 2),
    35: (1, 3, 3), 36: (1, 3, 4), 37: (1, 3, 5), 38: (1, 3, 8), 39: (1, 3, 9),
    40: (1, 3, 10), 41: (1, 3, 11), 42: (1, 1, 0), 43: (1, 1, 1), 44: (1, 1, 2),
    45: (1, 1, 8), 46: (1, 9, 14), 47: (2, 2, 0), 48: (2, 2, 1), 49: (2, 2, 2),
    50: (2, 2, 4), 51: (2, 2, 9), 52: (2, 2, 10), 53: (2, 2, 11), 54: (2, 3, 0),
    55: (2, 3, 1), 56: (2, 3, 2), 57: (2, 3, 3), 58: (2, 3, 4), 59: (2, 3, 5),
    60: (2, 3, 8), 61: (2, 3, 9), 62: (2, 3, 10), 63: (2, 3, 11), 64: (2, 1, 0),
    65: (2, 1, 1), 66: (2, 1, 8), 67: (2, 9, 14), 68: (3, 5, 2), 69: (3, 5, 3),
    70: (3, 5, 4), 71: (3, 5, 5), 72: (3, 5, 9), 73: (3, 5, 10), 74: (3, 5, 11),
    75: (3, 2, 9), 76: (3, 2, 10), 77: (3, 2, 11), 78: (3, 5, 0), 79: (3, 9, 14),
    80: (4, 4, 2), 81: (4, 4, 3), 82: (4, 4, 4), 83: (4, 4, 5), 84: (4, 9, 14),
    85: (5, 6, 6), 86: (5, 7, 7), 87: (5, 2, 1), 88: (5, 2, 0), 89: (5, 1, 8),
    90: (5, 2, 9), 91: (5, 2, 10), 92: (5, 9, 14), 93: (0, 0, 12), 94: (0, 1, 12),
    95: (1, 3, 12), 96: (2, 3, 12), 97: (3, 5, 12), 98: (0, 2, 8), 99: (1, 1, 9),
}

# ============ RDV官方triplet顺序（与triplet_config.py一致） ============
RDV_TRIPLET_CLASSES = [
    ('bipolar', 'coagulate', 'abdominal-wall/cavity'),  # 0
    ('bipolar', 'coagulate', 'blood-vessel'),           # 1
    ('bipolar', 'coagulate', 'cystic-artery'),          # 2
    ('bipolar', 'coagulate', 'cystic-duct'),            # 3
    ('bipolar', 'coagulate', 'cystic-pedicle'),         # 4
    ('bipolar', 'coagulate', 'cystic-plate'),           # 5
    ('bipolar', 'coagulate', 'gallbladder'),            # 6
    ('bipolar', 'coagulate', 'liver'),                  # 7
    ('bipolar', 'coagulate', 'omentum'),                # 8
    ('bipolar', 'coagulate', 'peritoneum'),             # 9
    ('bipolar', 'dissect', 'adhesion'),                 # 10
    ('bipolar', 'dissect', 'cystic-artery'),            # 11
    ('bipolar', 'dissect', 'cystic-duct'),              # 12
    ('bipolar', 'dissect', 'cystic-plate'),             # 13
    ('bipolar', 'dissect', 'gallbladder'),              # 14
    ('bipolar', 'dissect', 'omentum'),                  # 15
    ('bipolar', 'grasp', 'cystic-plate'),               # 16
    ('bipolar', 'grasp', 'liver'),                      # 17
    ('bipolar', 'grasp', 'specimen-bag'),               # 18
    ('bipolar', 'null-verb', 'null-target'),            # 19
    ('bipolar', 'retract', 'cystic-duct'),              # 20
    ('bipolar', 'retract', 'cystic-pedicle'),           # 21
    ('bipolar', 'retract', 'gallbladder'),              # 22
    ('bipolar', 'retract', 'liver'),                    # 23
    ('bipolar', 'retract', 'omentum'),                  # 24
    ('clipper', 'clip', 'blood-vessel'),                # 25
    ('clipper', 'clip', 'cystic-artery'),               # 26
    ('clipper', 'clip', 'cystic-duct'),                 # 27
    ('clipper', 'clip', 'cystic-pedicle'),              # 28
    ('clipper', 'clip', 'cystic-plate'),                # 29
    ('clipper', 'null-verb', 'null-target'),            # 30
    ('grasper', 'dissect', 'cystic-plate'),             # 31
    ('grasper', 'dissect', 'gallbladder'),              # 32
    ('grasper', 'dissect', 'omentum'),                  # 33
    ('grasper', 'grasp', 'cystic-artery'),              # 34
    ('grasper', 'grasp', 'cystic-duct'),                # 35
    ('grasper', 'grasp', 'cystic-pedicle'),             # 36
    ('grasper', 'grasp', 'cystic-plate'),               # 37
    ('grasper', 'grasp', 'gallbladder'),                # 38
    ('grasper', 'grasp', 'gut'),                        # 39
    ('grasper', 'grasp', 'liver'),                      # 40
    ('grasper', 'grasp', 'omentum'),                    # 41
    ('grasper', 'grasp', 'peritoneum'),                 # 42
    ('grasper', 'grasp', 'specimen-bag'),               # 43
    ('grasper', 'null-verb', 'null-target'),            # 44
    ('grasper', 'pack', 'gallbladder'),                 # 45
    ('grasper', 'retract', 'cystic-duct'),              # 46
    ('grasper', 'retract', 'cystic-pedicle'),           # 47
    ('grasper', 'retract', 'cystic-plate'),             # 48
    ('grasper', 'retract', 'gallbladder'),              # 49
    ('grasper', 'retract', 'gut'),                      # 50
    ('grasper', 'retract', 'liver'),                    # 51
    ('grasper', 'retract', 'omentum'),                  # 52
    ('grasper', 'retract', 'peritoneum'),               # 53
    ('hook', 'coagulate', 'blood-vessel'),              # 54
    ('hook', 'coagulate', 'cystic-artery'),             # 55
    ('hook', 'coagulate', 'cystic-duct'),               # 56
    ('hook', 'coagulate', 'cystic-pedicle'),            # 57
    ('hook', 'coagulate', 'cystic-plate'),              # 58
    ('hook', 'coagulate', 'gallbladder'),               # 59
    ('hook', 'coagulate', 'liver'),                     # 60
    ('hook', 'coagulate', 'omentum'),                   # 61
    ('hook', 'cut', 'blood-vessel'),                    # 62
    ('hook', 'cut', 'peritoneum'),                      # 63
    ('hook', 'dissect', 'blood-vessel'),                # 64
    ('hook', 'dissect', 'cystic-artery'),               # 65
    ('hook', 'dissect', 'cystic-duct'),                 # 66
    ('hook', 'dissect', 'cystic-plate'),                # 67
    ('hook', 'dissect', 'gallbladder'),                 # 68
    ('hook', 'dissect', 'omentum'),                     # 69
    ('hook', 'dissect', 'peritoneum'),                  # 70
    ('hook', 'null-verb', 'null-target'),               # 71
    ('hook', 'retract', 'gallbladder'),                 # 72
    ('hook', 'retract', 'liver'),                       # 73
    ('irrigator', 'aspirate', 'fluid'),                 # 74
    ('irrigator', 'dissect', 'cystic-duct'),            # 75
    ('irrigator', 'dissect', 'cystic-pedicle'),         # 76
    ('irrigator', 'dissect', 'cystic-plate'),           # 77
    ('irrigator', 'dissect', 'gallbladder'),            # 78
    ('irrigator', 'dissect', 'omentum'),                # 79
    ('irrigator', 'irrigate', 'abdominal-wall/cavity'), # 80
    ('irrigator', 'irrigate', 'cystic-pedicle'),        # 81
    ('irrigator', 'irrigate', 'liver'),                 # 82
    ('irrigator', 'null-verb', 'null-target'),          # 83
    ('irrigator', 'retract', 'gallbladder'),            # 84
    ('irrigator', 'retract', 'liver'),                  # 85
    ('irrigator', 'retract', 'omentum'),                # 86
    ('scissors', 'coagulate', 'omentum'),               # 87
    ('scissors', 'cut', 'adhesion'),                    # 88
    ('scissors', 'cut', 'blood-vessel'),                # 89
    ('scissors', 'cut', 'cystic-artery'),               # 90
    ('scissors', 'cut', 'cystic-duct'),                 # 91
    ('scissors', 'cut', 'cystic-plate'),                # 92
    ('scissors', 'cut', 'liver'),                       # 93
    ('scissors', 'cut', 'omentum'),                     # 94
    ('scissors', 'cut', 'peritoneum'),                  # 95
    ('scissors', 'dissect', 'cystic-plate'),            # 96
    ('scissors', 'dissect', 'gallbladder'),             # 97
    ('scissors', 'dissect', 'omentum'),                 # 98
    ('scissors', 'null-verb', 'null-target'),           # 99
]


def build_cholect50_to_rdv_mapping():
    """
    构建 CholecT50原生编号 -> RDV编号 的映射
    """
    # 先将CholecT50的triplet转换为标准化的名称
    def normalize(s):
        return s.lower().replace('_', '-').replace('abdominal-wall-cavity', 'abdominal-wall/cavity')

    cholect50_to_name = {}
    for tid, (i, v, t) in CHOLECT50_TRIPLET_MAP.items():
        inst = normalize(CHOLECT50_INSTRUMENTS[i])
        verb = normalize(CHOLECT50_VERBS[v])
        targ = normalize(CHOLECT50_TARGETS[t])
        cholect50_to_name[tid] = (inst, verb, targ)

    # 构建RDV名称到ID的映射
    rdv_name_to_id = {}
    for rdv_id, (inst, verb, targ) in enumerate(RDV_TRIPLET_CLASSES):
        rdv_name_to_id[(inst, verb, targ)] = rdv_id

    # 构建最终映射
    cholect50_to_rdv = {}
    unmapped = []

    for ct50_id, name in cholect50_to_name.items():
        if name in rdv_name_to_id:
            cholect50_to_rdv[ct50_id] = rdv_name_to_id[name]
        else:
            unmapped.append((ct50_id, name))

    return cholect50_to_rdv, unmapped


def phase_name_to_id(phase_name):
    """将phase文本名称转换为ID"""
    normalized = phase_name.lower().replace(' ', '').replace('-', '').replace('_', '')
    return PHASE_NAME_TO_ID.get(normalized, -1)


def load_triplet_annotations(video_id, cholect50_to_rdv):
    """
    加载三元组标注，并转换为RDV编号
    """
    triplet_folder = os.path.join(CHOLECT50_PATH, 'triplet')

    possible_names = [f"VID{video_id:02d}.txt", f"VID{video_id}.txt"]

    for name in possible_names:
        triplet_file = os.path.join(triplet_folder, name)
        if os.path.exists(triplet_file):
            frame_triplets = {}
            with open(triplet_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) > 1:
                        frame_id = int(parts[0])
                        triplet_labels = [int(x) for x in parts[1:]]

                        # 转换为RDV编号
                        active_rdv = []
                        for ct50_id, is_active in enumerate(triplet_labels):
                            if is_active == 1 and ct50_id in cholect50_to_rdv:
                                active_rdv.append(cholect50_to_rdv[ct50_id])

                        frame_triplets[frame_id] = active_rdv
            return frame_triplets
    return None


def load_phase_annotations(video_id):
    """加载phase标注"""
    phase_file = os.path.join(CHOLEC80_PHASE_PATH, f"video{video_id:02d}-phase.txt")

    if not os.path.exists(phase_file):
        return None

    frame_to_phase = {}

    with open(phase_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or (line_num == 0 and 'Frame' in line):
                continue

            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    frame_id = int(parts[0])
                    phase_id = phase_name_to_id(parts[1])
                    if phase_id >= 0:
                        frame_to_phase[frame_id] = phase_id
                except ValueError:
                    continue

    return frame_to_phase if frame_to_phase else None


def main():
    print("=" * 70)
    print("🏥 Phase-Triplet 分析 (RDV官方顺序)")
    print("=" * 70)

    # 构建编号映射
    cholect50_to_rdv, unmapped = build_cholect50_to_rdv_mapping()
    print(f"\n✅ 编号映射: {len(cholect50_to_rdv)}/100 triplet可映射到RDV")
    if unmapped:
        print(f"⚠️ 无法映射的triplet ({len(unmapped)}个):")
        for ct50_id, name in unmapped[:5]:
            print(f"   CholecT50 #{ct50_id}: {name}")

    # 检测视频
    triplet_folder = os.path.join(CHOLECT50_PATH, 'triplet')
    triplet_files = os.listdir(triplet_folder)
    cholect50_videos = []
    for f in triplet_files:
        match = re.search(r'(\d+)', f)
        if match:
            cholect50_videos.append(int(match.group(1)))
    cholect50_videos = sorted(cholect50_videos)

    phase_files = os.listdir(CHOLEC80_PHASE_PATH)
    cholec80_videos = []
    for f in phase_files:
        match = re.search(r'video(\d+)', f)
        if match:
            cholec80_videos.append(int(match.group(1)))
    cholec80_videos = sorted(set(cholec80_videos))

    common_videos = sorted(set(cholect50_videos) & set(cholec80_videos))
    print(f"\n✅ 可匹配的视频: {len(common_videos)}个")

    # 统计
    phase_triplet_counts = defaultdict(Counter)  # phase -> {rdv_triplet_id: count}
    phase_total_frames = Counter()

    for video_id in common_videos:
        triplet_ann = load_triplet_annotations(video_id, cholect50_to_rdv)
        if triplet_ann is None:
            continue

        phase_ann = load_phase_annotations(video_id)
        if phase_ann is None:
            continue

        for frame_id, rdv_triplets in triplet_ann.items():
            cholec80_frame = frame_id * 25

            if cholec80_frame in phase_ann:
                phase = phase_ann[cholec80_frame]
            elif phase_ann:
                closest = min(phase_ann.keys(), key=lambda x: abs(x - cholec80_frame))
                phase = phase_ann[closest]
            else:
                continue

            phase_total_frames[phase] += 1
            for rdv_tid in rdv_triplets:
                phase_triplet_counts[phase][rdv_tid] += 1

    print(f"\n✅ 统计完成")

    # 输出CSV
    csv_file = os.path.join(CHOLECT50_PATH, 'phase_triplet_prior_rdv.csv')
    phases = sorted(phase_triplet_counts.keys())

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['triplet_id', 'triplet_name'] + [f'phase_{p}' for p in phases])

        for rdv_id in range(100):
            inst, verb, targ = RDV_TRIPLET_CLASSES[rdv_id]
            triplet_name = f"<{inst}, {verb}, {targ}>"

            row = [rdv_id, triplet_name]
            for p in phases:
                total = phase_total_frames[p]
                count = phase_triplet_counts[p].get(rdv_id, 0)
                row.append(f"{count/total:.6f}" if total > 0 else "0")
            writer.writerow(row)

    print(f"\n💾 已保存: {csv_file}")

    # 验证输出
    print("\n" + "=" * 70)
    print("验证输出:")
    print("=" * 70)
    import pandas as pd
    df = pd.read_csv(csv_file)
    print(f"triplet_id=0: {df.iloc[0]['triplet_name']}")
    print(f"triplet_id=38: {df.iloc[38]['triplet_name']}")
    print(f"triplet_id=99: {df.iloc[99]['triplet_name']}")

    print("\n✅ 完成!")


if __name__ == "__main__":
    main()