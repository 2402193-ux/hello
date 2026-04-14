"""
CholecT50 数据集分析脚本
分析三元组分布、阶段关联等信息
"""

import os
from collections import defaultdict, Counter
from pathlib import Path
import json

# 数据集路径 - 请根据实际情况修改
DATASET_PATH = r"D:\download\CholecT50"

# CholecT50 的标准类别定义
INSTRUMENTS = {
    0: 'grasper',
    1: 'bipolar', 
    2: 'hook',
    3: 'scissors',
    4: 'clipper',
    5: 'irrigator'
}

VERBS = {
    0: 'grasp',
    1: 'retract',
    2: 'dissect',
    3: 'coagulate',
    4: 'clip',
    5: 'cut',
    6: 'aspirate',
    7: 'irrigate',
    8: 'pack',
    9: 'null_verb'
}

TARGETS = {
    0: 'gallbladder',
    1: 'cystic_plate',
    2: 'cystic_duct',
    3: 'cystic_artery',
    4: 'cystic_pedicle',
    5: 'blood_vessel',
    6: 'fluid',
    7: 'abdominal_wall_cavity',
    8: 'liver',
    9: 'adhesion',
    10: 'omentum',
    11: 'peritoneum',
    12: 'gut',
    13: 'specimen_bag',
    14: 'null_target'
}

# 100个三元组类别的标准定义
TRIPLET_MAP = {
    0: (0, 0, 0),   # grasper, grasp, gallbladder
    1: (0, 0, 1),   # grasper, grasp, cystic_plate
    2: (0, 0, 2),   # grasper, grasp, cystic_duct
    3: (0, 0, 8),   # grasper, grasp, liver
    4: (0, 0, 9),   # grasper, grasp, adhesion
    5: (0, 0, 10),  # grasper, grasp, omentum
    6: (0, 0, 11),  # grasper, grasp, peritoneum
    7: (0, 0, 13),  # grasper, grasp, specimen_bag
    8: (0, 1, 0),   # grasper, retract, gallbladder
    9: (0, 1, 1),   # grasper, retract, cystic_plate
    10: (0, 1, 2),  # grasper, retract, cystic_duct
    11: (0, 1, 8),  # grasper, retract, liver
    12: (0, 1, 9),  # grasper, retract, adhesion
    13: (0, 1, 10), # grasper, retract, omentum
    14: (0, 1, 11), # grasper, retract, peritoneum
    15: (0, 2, 0),  # grasper, dissect, gallbladder
    16: (0, 2, 1),  # grasper, dissect, cystic_plate
    17: (0, 2, 2),  # grasper, dissect, cystic_duct
    18: (0, 2, 9),  # grasper, dissect, adhesion
    19: (0, 2, 10), # grasper, dissect, omentum
    20: (0, 2, 11), # grasper, dissect, peritoneum
    21: (0, 8, 0),  # grasper, pack, gallbladder
    22: (0, 8, 13), # grasper, pack, specimen_bag
    23: (0, 9, 14), # grasper, null_verb, null_target
    24: (1, 2, 0),  # bipolar, dissect, gallbladder
    25: (1, 2, 1),  # bipolar, dissect, cystic_plate
    26: (1, 2, 2),  # bipolar, dissect, cystic_duct
    27: (1, 2, 3),  # bipolar, dissect, cystic_artery
    28: (1, 2, 4),  # bipolar, dissect, cystic_pedicle
    29: (1, 2, 9),  # bipolar, dissect, adhesion
    30: (1, 2, 10), # bipolar, dissect, omentum
    31: (1, 2, 11), # bipolar, dissect, peritoneum
    32: (1, 3, 0),  # bipolar, coagulate, gallbladder
    33: (1, 3, 1),  # bipolar, coagulate, cystic_plate
    34: (1, 3, 2),  # bipolar, coagulate, cystic_duct
    35: (1, 3, 3),  # bipolar, coagulate, cystic_artery
    36: (1, 3, 4),  # bipolar, coagulate, cystic_pedicle
    37: (1, 3, 5),  # bipolar, coagulate, blood_vessel
    38: (1, 3, 8),  # bipolar, coagulate, liver
    39: (1, 3, 9),  # bipolar, coagulate, adhesion
    40: (1, 3, 10), # bipolar, coagulate, omentum
    41: (1, 3, 11), # bipolar, coagulate, peritoneum
    42: (1, 1, 0),  # bipolar, retract, gallbladder
    43: (1, 1, 1),  # bipolar, retract, cystic_plate
    44: (1, 1, 2),  # bipolar, retract, cystic_duct
    45: (1, 1, 8),  # bipolar, retract, liver
    46: (1, 9, 14), # bipolar, null_verb, null_target
    47: (2, 2, 0),  # hook, dissect, gallbladder
    48: (2, 2, 1),  # hook, dissect, cystic_plate
    49: (2, 2, 2),  # hook, dissect, cystic_duct
    50: (2, 2, 4),  # hook, dissect, cystic_pedicle
    51: (2, 2, 9),  # hook, dissect, adhesion
    52: (2, 2, 10), # hook, dissect, omentum
    53: (2, 2, 11), # hook, dissect, peritoneum
    54: (2, 3, 0),  # hook, coagulate, gallbladder
    55: (2, 3, 1),  # hook, coagulate, cystic_plate
    56: (2, 3, 2),  # hook, coagulate, cystic_duct
    57: (2, 3, 3),  # hook, coagulate, cystic_artery
    58: (2, 3, 4),  # hook, coagulate, cystic_pedicle
    59: (2, 3, 5),  # hook, coagulate, blood_vessel
    60: (2, 3, 8),  # hook, coagulate, liver
    61: (2, 3, 9),  # hook, coagulate, adhesion
    62: (2, 3, 10), # hook, coagulate, omentum
    63: (2, 3, 11), # hook, coagulate, peritoneum
    64: (2, 1, 0),  # hook, retract, gallbladder
    65: (2, 1, 1),  # hook, retract, cystic_plate
    66: (2, 1, 8),  # hook, retract, liver
    67: (2, 9, 14), # hook, null_verb, null_target
    68: (3, 5, 2),  # scissors, cut, cystic_duct
    69: (3, 5, 3),  # scissors, cut, cystic_artery
    70: (3, 5, 4),  # scissors, cut, cystic_pedicle
    71: (3, 5, 5),  # scissors, cut, blood_vessel
    72: (3, 5, 9),  # scissors, cut, adhesion
    73: (3, 5, 10), # scissors, cut, omentum
    74: (3, 5, 11), # scissors, cut, peritoneum
    75: (3, 2, 9),  # scissors, dissect, adhesion
    76: (3, 2, 10), # scissors, dissect, omentum
    77: (3, 2, 11), # scissors, dissect, peritoneum
    78: (3, 5, 0),  # scissors, cut, gallbladder
    79: (3, 9, 14), # scissors, null_verb, null_target
    80: (4, 4, 2),  # clipper, clip, cystic_duct
    81: (4, 4, 3),  # clipper, clip, cystic_artery
    82: (4, 4, 4),  # clipper, clip, cystic_pedicle
    83: (4, 4, 5),  # clipper, clip, blood_vessel
    84: (4, 9, 14), # clipper, null_verb, null_target
    85: (5, 6, 6),  # irrigator, aspirate, fluid
    86: (5, 7, 7),  # irrigator, irrigate, abdominal_wall_cavity
    87: (5, 2, 1),  # irrigator, dissect, cystic_plate
    88: (5, 2, 0),  # irrigator, dissect, gallbladder
    89: (5, 1, 8),  # irrigator, retract, liver
    90: (5, 2, 9),  # irrigator, dissect, adhesion
    91: (5, 2, 10), # irrigator, dissect, omentum
    92: (5, 9, 14), # irrigator, null_verb, null_target
    93: (0, 0, 12), # grasper, grasp, gut
    94: (0, 1, 12), # grasper, retract, gut
    95: (1, 3, 12), # bipolar, coagulate, gut
    96: (2, 3, 12), # hook, coagulate, gut
    97: (3, 5, 12), # scissors, cut, gut
    98: (0, 2, 8),  # grasper, dissect, liver
    99: (1, 1, 9),  # bipolar, retract, adhesion
}

def triplet_id_to_name(triplet_id):
    """将三元组ID转换为可读名称"""
    if triplet_id in TRIPLET_MAP:
        i, v, t = TRIPLET_MAP[triplet_id]
        return f"<{INSTRUMENTS[i]}, {VERBS[v]}, {TARGETS[t]}>"
    return f"<unknown_{triplet_id}>"

def analyze_dataset_structure():
    """分析数据集结构"""
    print("=" * 60)
    print("CholecT50 数据集结构分析")
    print("=" * 60)
    
    # 检查目录结构
    folders = ['instrument', 'labels', 'target', 'triplet', 'verb']
    
    for folder in folders:
        folder_path = os.path.join(DATASET_PATH, folder)
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            print(f"\n📁 {folder}/")
            print(f"   文件数量: {len(files)}")
            if files:
                # 读取第一个文件的前几行
                first_file = os.path.join(folder_path, sorted(files)[0])
                with open(first_file, 'r') as f:
                    lines = f.readlines()[:5]
                print(f"   示例文件: {sorted(files)[0]}")
                print(f"   总行数: {len(open(first_file).readlines())}")
                print(f"   前3行内容:")
                for line in lines[:3]:
                    print(f"      {line.strip()}")
        else:
            print(f"\n❌ {folder}/ 不存在")
    
    # 检查是否有phase标注
    phase_folder = os.path.join(DATASET_PATH, 'phase')
    if os.path.exists(phase_folder):
        print(f"\n✅ 发现 phase/ 文件夹 - 存在手术阶段标注!")
    else:
        print(f"\n⚠️  未发现 phase/ 文件夹")
        print("   CholecT50本身不包含phase标注，但可以从Cholec80获取")
        print("   Cholec80的phase标注路径通常为: cholec80/phase_annotations/")

def analyze_triplet_distribution():
    """分析三元组分布"""
    print("\n" + "=" * 60)
    print("三元组分布分析")
    print("=" * 60)
    
    triplet_folder = os.path.join(DATASET_PATH, 'triplet')
    if not os.path.exists(triplet_folder):
        print("❌ triplet文件夹不存在")
        return None
    
    # 统计所有三元组出现次数
    triplet_counts = Counter()
    video_triplets = defaultdict(set)  # 每个视频出现的三元组
    triplet_videos = defaultdict(set)  # 每个三元组出现在哪些视频
    
    files = sorted(os.listdir(triplet_folder))
    
    for file in files:
        video_id = file.replace('.txt', '')
        file_path = os.path.join(triplet_folder, file)
        
        with open(file_path, 'r') as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split(',')
                if len(parts) > 1:
                    # 跳过帧号，解析三元组标签
                    triplet_labels = [int(x) for x in parts[1:]]
                    for tid, present in enumerate(triplet_labels):
                        if present == 1:
                            triplet_counts[tid] += 1
                            video_triplets[video_id].add(tid)
                            triplet_videos[tid].add(video_id)
    
    # 按频率排序
    sorted_triplets = sorted(triplet_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 三元组类别总数: {len(triplet_counts)}")
    print(f"📊 三元组实例总数: {sum(triplet_counts.values())}")
    
    print("\n🔝 Top 20 高频三元组:")
    print("-" * 70)
    for tid, count in sorted_triplets[:20]:
        name = triplet_id_to_name(tid)
        video_count = len(triplet_videos[tid])
        print(f"   ID {tid:3d}: {name:45s} | 出现 {count:6d} 次 | {video_count:2d} 个视频")
    
    print("\n🔻 Top 20 低频三元组 (尾部类别):")
    print("-" * 70)
    for tid, count in sorted_triplets[-20:]:
        name = triplet_id_to_name(tid)
        video_count = len(triplet_videos[tid])
        print(f"   ID {tid:3d}: {name:45s} | 出现 {count:6d} 次 | {video_count:2d} 个视频")
    
    # 分析类别不平衡
    counts = list(triplet_counts.values())
    if counts:
        max_count = max(counts)
        min_count = min(counts)
        print(f"\n📈 类别不平衡分析:")
        print(f"   最高频: {max_count} 次")
        print(f"   最低频: {min_count} 次")
        print(f"   不平衡比: {max_count/min_count:.1f}:1")
    
    return triplet_counts, video_triplets, triplet_videos

def analyze_triplet_cooccurrence():
    """分析三元组共现关系 - 哪些三元组经常一起出现"""
    print("\n" + "=" * 60)
    print("三元组共现分析 (同一帧内)")
    print("=" * 60)
    
    triplet_folder = os.path.join(DATASET_PATH, 'triplet')
    if not os.path.exists(triplet_folder):
        return
    
    # 统计共现
    cooccurrence = defaultdict(Counter)
    
    files = sorted(os.listdir(triplet_folder))
    
    for file in files:
        file_path = os.path.join(triplet_folder, file)
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) > 1:
                    triplet_labels = [int(x) for x in parts[1:]]
                    active_triplets = [tid for tid, present in enumerate(triplet_labels) if present == 1]
                    
                    # 记录共现
                    for i, t1 in enumerate(active_triplets):
                        for t2 in active_triplets[i+1:]:
                            cooccurrence[t1][t2] += 1
                            cooccurrence[t2][t1] += 1
    
    # 找出高频共现对
    pairs = []
    seen = set()
    for t1, co_dict in cooccurrence.items():
        for t2, count in co_dict.items():
            if (t2, t1) not in seen:
                pairs.append((t1, t2, count))
                seen.add((t1, t2))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("\n🔗 Top 15 高频共现三元组对:")
    print("-" * 90)
    for t1, t2, count in pairs[:15]:
        name1 = triplet_id_to_name(t1)
        name2 = triplet_id_to_name(t2)
        print(f"   {name1}")
        print(f"   + {name2}")
        print(f"   共现 {count} 次\n")

def analyze_verb_target_combinations():
    """分析动作-目标的组合模式"""
    print("\n" + "=" * 60)
    print("动作-目标组合分析")
    print("=" * 60)
    
    # 基于三元组定义分析
    verb_target_combinations = defaultdict(set)
    instrument_verb_combinations = defaultdict(set)
    
    for tid, (i, v, t) in TRIPLET_MAP.items():
        verb_name = VERBS[v]
        target_name = TARGETS[t]
        instrument_name = INSTRUMENTS[i]
        
        verb_target_combinations[verb_name].add(target_name)
        instrument_verb_combinations[instrument_name].add(verb_name)
    
    print("\n🔧 器械-动作组合 (哪些器械可以执行哪些动作):")
    print("-" * 50)
    for inst, verbs in sorted(instrument_verb_combinations.items()):
        print(f"   {inst}: {', '.join(sorted(verbs))}")
    
    print("\n🎯 动作-目标组合 (哪些动作可以作用于哪些目标):")
    print("-" * 50)
    for verb, targets in sorted(verb_target_combinations.items()):
        print(f"   {verb}:")
        print(f"      {', '.join(sorted(targets))}")

def analyze_confusable_triplets():
    """分析易混淆的三元组对"""
    print("\n" + "=" * 60)
    print("易混淆三元组分析")
    print("=" * 60)
    
    # 找出只有一个组件不同的三元组对
    confusable_pairs = {
        'same_instrument_verb': [],  # 同器械同动作，不同目标
        'same_instrument_target': [],  # 同器械同目标，不同动作
        'same_verb_target': [],  # 同动作同目标，不同器械
    }
    
    triplet_ids = list(TRIPLET_MAP.keys())
    
    for i, tid1 in enumerate(triplet_ids):
        for tid2 in triplet_ids[i+1:]:
            i1, v1, t1 = TRIPLET_MAP[tid1]
            i2, v2, t2 = TRIPLET_MAP[tid2]
            
            # 计算差异数量
            diff = (i1 != i2) + (v1 != v2) + (t1 != t2)
            
            if diff == 1:  # 只有一个组件不同
                if i1 == i2 and v1 == v2:  # 同器械同动作
                    confusable_pairs['same_instrument_verb'].append((tid1, tid2))
                elif i1 == i2 and t1 == t2:  # 同器械同目标
                    confusable_pairs['same_instrument_target'].append((tid1, tid2))
                elif v1 == v2 and t1 == t2:  # 同动作同目标
                    confusable_pairs['same_verb_target'].append((tid1, tid2))
    
    print("\n⚠️ 同器械+同动作，仅目标不同 (视觉高度相似):")
    print("-" * 80)
    for tid1, tid2 in confusable_pairs['same_instrument_verb'][:10]:
        print(f"   {triplet_id_to_name(tid1)}")
        print(f"   vs {triplet_id_to_name(tid2)}\n")
    print(f"   ... 共 {len(confusable_pairs['same_instrument_verb'])} 对")
    
    print("\n⚠️ 同器械+同目标，仅动作不同 (最难区分!):")
    print("-" * 80)
    for tid1, tid2 in confusable_pairs['same_instrument_target'][:10]:
        print(f"   {triplet_id_to_name(tid1)}")
        print(f"   vs {triplet_id_to_name(tid2)}\n")
    print(f"   ... 共 {len(confusable_pairs['same_instrument_target'])} 对")

def generate_research_insights():
    """生成研究洞察"""
    print("\n" + "=" * 60)
    print("研究洞察与建议")
    print("=" * 60)
    
    print("""
📌 关于手术阶段(Phase)标注:
   
   CholecT50 本身不包含 phase 标注，但其50个视频来自 Cholec80 数据集。
   Cholec80 提供7个手术阶段的标注：
   
   Phase 1: Preparation (准备)
   Phase 2: Calot Triangle Dissection (胆囊三角解剖) 
   Phase 3: Clipping and Cutting (夹闭与切断)
   Phase 4: Gallbladder Dissection (胆囊分离)
   Phase 5: Gallbladder Packaging (胆囊打包)
   Phase 6: Cleaning and Coagulation (清洁与电凝)
   Phase 7: Gallbladder Retraction (胆囊牵出)

💡 获取Phase标注的方法:
   1. 下载 Cholec80 数据集 (http://camma.u-strasbg.fr/datasets)
   2. Phase标注在 cholec80/phase_annotations/ 目录下
   3. 文件格式: video{xx}-phase.txt, 每行是 "帧号\\t阶段ID"
   4. CholecT50的视频编号与Cholec80一致，可直接对应

🔬 Phase-Triplet关联分析的研究价值:
   
   1. 阶段特异性三元组:
      - Phase 2 (胆囊三角解剖): 主要是 dissect 动作
      - Phase 3 (夹闭切断): clipper+clip, scissors+cut 
      - Phase 4 (胆囊分离): hook/bipolar + coagulate
      
   2. 可以构建的先验知识:
      - 某阶段不可能出现的三元组 → 负样本约束
      - 阶段转换时的三元组变化模式 → 时序建模线索
      - 阶段内的三元组共现规律 → 关联约束

   3. 研究方向建议:
      - Phase-aware Triplet Recognition: 利用阶段信息指导三元组识别
      - Triplet-based Phase Recognition: 利用三元组特征推断手术阶段
      - Joint Phase-Triplet Modeling: 联合建模两个任务
""")

def main():
    print("🏥 CholecT50 数据集深度分析")
    print("=" * 60)
    
    # 1. 分析数据集结构
    analyze_dataset_structure()
    
    # 2. 分析三元组分布
    result = analyze_triplet_distribution()
    
    # 3. 分析共现关系
    analyze_triplet_cooccurrence()
    
    # 4. 分析组合模式
    analyze_verb_target_combinations()
    
    # 5. 分析易混淆三元组
    analyze_confusable_triplets()
    
    # 6. 研究洞察
    generate_research_insights()
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
