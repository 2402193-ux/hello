"""
Phase-Triplet 关联分析 - 最终修复版
正确解析Cholec80的文本格式phase标注
"""

import os
import re
from collections import defaultdict, Counter
import csv

# ============ 配置路径 ============
CHOLECT50_PATH = r"D:\download\CholecT50"
CHOLEC80_PHASE_PATH = r"D:\EndoViT-main\datasets\Cholec80\phase_annotations"

# ============ Phase名称到ID的映射 ============
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
    0: 'Preparation (准备)',
    1: 'Calot Triangle Dissection (胆囊三角解剖)',
    2: 'Clipping and Cutting (夹闭与切断)',
    3: 'Gallbladder Dissection (胆囊分离)',
    4: 'Gallbladder Packaging (胆囊打包)',
    5: 'Cleaning and Coagulation (清洁与电凝)',
    6: 'Gallbladder Retraction (胆囊牵出)'
}

# ============ 类别定义 ============
INSTRUMENTS = {
    0: 'grasper', 1: 'bipolar', 2: 'hook', 
    3: 'scissors', 4: 'clipper', 5: 'irrigator'
}

VERBS = {
    0: 'grasp', 1: 'retract', 2: 'dissect', 3: 'coagulate',
    4: 'clip', 5: 'cut', 6: 'aspirate', 7: 'irrigate', 
    8: 'pack', 9: 'null_verb'
}

TARGETS = {
    0: 'gallbladder', 1: 'cystic_plate', 2: 'cystic_duct',
    3: 'cystic_artery', 4: 'cystic_pedicle', 5: 'blood_vessel',
    6: 'fluid', 7: 'abdominal_wall_cavity', 8: 'liver',
    9: 'adhesion', 10: 'omentum', 11: 'peritoneum',
    12: 'gut', 13: 'specimen_bag', 14: 'null_target'
}

TRIPLET_MAP = {
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

def triplet_id_to_name(tid):
    if tid in TRIPLET_MAP:
        i, v, t = TRIPLET_MAP[tid]
        return f"<{INSTRUMENTS[i]}, {VERBS[v]}, {TARGETS[t]}>"
    return f"<unknown_{tid}>"

def phase_name_to_id(phase_name):
    """将phase文本名称转换为ID"""
    # 移除空格、连字符，转小写
    normalized = phase_name.lower().replace(' ', '').replace('-', '').replace('_', '')
    return PHASE_NAME_TO_ID.get(normalized, -1)

def load_triplet_annotations(video_id):
    """加载三元组标注"""
    triplet_folder = os.path.join(CHOLECT50_PATH, 'triplet')
    
    possible_names = [
        f"VID{video_id:02d}.txt",
        f"VID{video_id}.txt",
    ]
    
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
                        active = [tid for tid, p in enumerate(triplet_labels) if p == 1]
                        frame_triplets[frame_id] = active
            return frame_triplets
    return None

def load_phase_annotations(video_id):
    """加载phase标注（文本格式）"""
    phase_file = os.path.join(CHOLEC80_PHASE_PATH, f"video{video_id:02d}-phase.txt")
    
    if not os.path.exists(phase_file):
        return None
    
    frame_to_phase = {}
    
    with open(phase_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # 跳过表头
            if line_num == 0 and 'Frame' in line:
                continue
            
            # 解析：Frame\tPhaseName
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    frame_id = int(parts[0])
                    phase_name = parts[1]
                    phase_id = phase_name_to_id(phase_name)
                    
                    if phase_id >= 0:
                        frame_to_phase[frame_id] = phase_id
                except ValueError:
                    continue
    
    return frame_to_phase if frame_to_phase else None

def detect_video_mapping():
    """检测视频对应关系"""
    print("=" * 70)
    print("🔍 检测CholecT50与Cholec80视频对应关系")
    print("=" * 70)
    
    # CholecT50视频
    triplet_folder = os.path.join(CHOLECT50_PATH, 'triplet')
    triplet_files = os.listdir(triplet_folder)
    
    cholect50_videos = []
    for f in triplet_files:
        match = re.search(r'(\d+)', f)
        if match:
            cholect50_videos.append(int(match.group(1)))
    cholect50_videos = sorted(cholect50_videos)
    
    print(f"\n📁 CholecT50 包含 {len(cholect50_videos)} 个视频:")
    print(f"   {cholect50_videos}")
    
    # Cholec80 phase文件
    phase_files = os.listdir(CHOLEC80_PHASE_PATH)
    cholec80_videos = []
    for f in phase_files:
        match = re.search(r'video(\d+)', f)
        if match:
            cholec80_videos.append(int(match.group(1)))
    cholec80_videos = sorted(set(cholec80_videos))
    
    print(f"\n📁 Cholec80 phase标注包含 {len(cholec80_videos)} 个视频")
    
    # 交集
    common = sorted(set(cholect50_videos) & set(cholec80_videos))
    print(f"\n✅ 可匹配的视频 ({len(common)} 个): {common}")
    
    missing = sorted(set(cholect50_videos) - set(cholec80_videos))
    if missing:
        print(f"⚠️ 无phase标注的视频: {missing}")
    
    return common

def main():
    print("🏥 CholecT50 Phase-Triplet 关联分析")
    print("=" * 70)
    
    # 检测视频
    common_videos = detect_video_mapping()
    
    if not common_videos:
        print("❌ 没有可匹配的视频")
        return
    
    # 统计结构
    phase_triplet_counts = defaultdict(Counter)
    phase_total_frames = Counter()
    triplet_phase_counts = defaultdict(Counter)
    
    success_count = 0
    
    print("\n" + "=" * 70)
    print("📊 加载数据")
    print("=" * 70)
    
    for video_id in common_videos:
        triplet_ann = load_triplet_annotations(video_id)
        if triplet_ann is None:
            continue
        
        phase_ann = load_phase_annotations(video_id)
        if phase_ann is None:
            print(f"   ⚠️ Video {video_id:02d}: phase解析失败")
            continue
        
        success_count += 1
        
        # 对齐帧号 (CholecT50是1fps, Cholec80是25fps)
        for frame_id, triplets in triplet_ann.items():
            cholec80_frame = frame_id * 25
            
            # 找最近的phase
            if cholec80_frame in phase_ann:
                phase = phase_ann[cholec80_frame]
            elif phase_ann:
                closest = min(phase_ann.keys(), key=lambda x: abs(x - cholec80_frame))
                phase = phase_ann[closest]
            else:
                continue
            
            phase_total_frames[phase] += 1
            for tid in triplets:
                phase_triplet_counts[phase][tid] += 1
                triplet_phase_counts[tid][phase] += 1
    
    print(f"\n✅ 成功加载 {success_count}/{len(common_videos)} 个视频")
    
    if success_count == 0:
        return
    
    # ============ 输出分析结果 ============
    
    print("\n" + "=" * 70)
    print("📊 各阶段三元组分布")
    print("=" * 70)
    
    for phase_id in sorted(phase_triplet_counts.keys()):
        phase_name = PHASE_NAMES.get(phase_id, f"Phase {phase_id}")
        triplets = phase_triplet_counts[phase_id]
        total = sum(triplets.values())
        frames = phase_total_frames[phase_id]
        
        print(f"\n{'='*65}")
        print(f"🏷️ Phase {phase_id}: {phase_name}")
        print(f"   帧数: {frames} | 三元组实例: {total} | 类别数: {len(triplets)}")
        print(f"\n   📌 Top 15 三元组:")
        for tid, count in triplets.most_common(15):
            pct = count / total * 100 if total > 0 else 0
            print(f"      {triplet_id_to_name(tid)}: {count} ({pct:.1f}%)")
    
    # 阶段特异性分析
    print("\n" + "=" * 70)
    print("🎯 阶段特异性三元组")
    print("=" * 70)
    
    all_triplets = set(range(100))
    
    for phase_id in sorted(phase_triplet_counts.keys()):
        phase_name = PHASE_NAMES.get(phase_id, f"Phase {phase_id}")
        present = set(phase_triplet_counts[phase_id].keys())
        absent = all_triplets - present
        
        print(f"\n{'='*65}")
        print(f"🏷️ Phase {phase_id}: {phase_name}")
        
        # 独特/主导三元组
        unique, dominant = [], []
        for tid in present:
            total = sum(triplet_phase_counts[tid].values())
            pc = triplet_phase_counts[tid][phase_id]
            ratio = pc / total if total > 0 else 0
            
            if len(triplet_phase_counts[tid]) == 1:
                unique.append((tid, pc))
            elif ratio > 0.5:
                dominant.append((tid, ratio, pc))
        
        if unique:
            print(f"\n   ✨ 独特三元组 (只在该阶段):")
            for tid, c in sorted(unique, key=lambda x: -x[1])[:8]:
                print(f"      {triplet_id_to_name(tid)}: {c}次")
        
        if dominant:
            print(f"\n   📌 主导三元组 (>50%在该阶段):")
            for tid, r, c in sorted(dominant, key=lambda x: -x[1])[:8]:
                print(f"      {triplet_id_to_name(tid)}: {r*100:.1f}%")
        
        # 不出现的
        meaningful_absent = [t for t in absent if 'null' not in triplet_id_to_name(t)]
        print(f"\n   🚫 从不出现 ({len(meaningful_absent)}个): ", end="")
        print(", ".join([str(t) for t in list(meaningful_absent)[:10]]), "...")
    
    # 保存结果
    print("\n" + "=" * 70)
    print("💾 保存结果")
    print("=" * 70)
    
    # CSV先验矩阵
    csv_file = os.path.join(CHOLECT50_PATH, 'phase_triplet_prior.csv')
    phases = sorted(phase_triplet_counts.keys())
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['triplet_id', 'triplet_name'] + [f'phase_{p}' for p in phases])
        
        for tid in range(100):
            row = [tid, triplet_id_to_name(tid)]
            for p in phases:
                total = phase_total_frames[p]
                count = phase_triplet_counts[p].get(tid, 0)
                row.append(f"{count/total:.6f}" if total > 0 else "0")
            writer.writerow(row)
    
    print(f"   ✅ 先验矩阵: {csv_file}")
    
    # 约束规则文本
    txt_file = os.path.join(CHOLECT50_PATH, 'phase_triplet_constraints.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("# Phase-Triplet 约束规则\n\n")
        
        for phase_id in sorted(phase_triplet_counts.keys()):
            phase_name = PHASE_NAMES.get(phase_id, f"Phase {phase_id}")
            present = set(phase_triplet_counts[phase_id].keys())
            absent = all_triplets - present
            meaningful_absent = [t for t in absent if 'null' not in triplet_id_to_name(t)]
            
            f.write(f"\n{'='*50}\n")
            f.write(f"Phase {phase_id}: {phase_name}\n")
            f.write(f"{'='*50}\n\n")
            
            f.write(f"出现的三元组 ({len(present)}个):\n")
            for tid in sorted(present):
                count = phase_triplet_counts[phase_id][tid]
                f.write(f"  {tid}: {triplet_id_to_name(tid)} ({count})\n")
            
            f.write(f"\n不出现的三元组 ({len(meaningful_absent)}个):\n")
            for tid in sorted(meaningful_absent):
                f.write(f"  {tid}: {triplet_id_to_name(tid)}\n")
    
    print(f"   ✅ 约束规则: {txt_file}")
    
    print("\n" + "=" * 70)
    print("✅ 分析完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()
