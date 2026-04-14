"""
CholecT50 Triplet配置 - 100个triplet类别定义
根据RDV论文Table 2构建
"""

# Component类别定义（与RDV论文Table 1对齐）
INSTRUMENTS = ['bipolar', 'clipper', 'grasper', 'hook', 'irrigator', 'scissors']
VERBS = ['aspirate', 'clip', 'coagulate', 'cut', 'dissect',
         'grasp', 'irrigate', 'null-verb', 'pack', 'retract']
TARGETS = ['abdominal-wall/cavity', 'adhesion', 'blood-vessel', 'cystic-artery',
           'cystic-duct', 'cystic-pedicle', 'cystic-plate', 'fluid', 'gallbladder',
           'gut', 'liver', 'null-target', 'omentum', 'peritoneum', 'specimen-bag']

# 100个Triplet类别定义（按RDV论文Table 2的顺序）
TRIPLET_CLASSES = [
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


# ==================== 映射构建函数 ====================

def build_triplet_to_components():
    """
    构建triplet到component的映射

    Returns:
        dict: triplet索引 -> (instrument_idx, verb_idx, target_idx)
    """
    triplet_to_comp = {}

    for triplet_idx, (inst, verb, targ) in enumerate(TRIPLET_CLASSES):
        i_idx = INSTRUMENTS.index(inst)
        v_idx = VERBS.index(verb)
        t_idx = TARGETS.index(targ)
        triplet_to_comp[triplet_idx] = (i_idx, v_idx, t_idx)

    return triplet_to_comp


def build_component_to_triplets():
    """
    构建component到triplet列表的映射（用于Eq. 10的分解）

    Returns:
        dict: {
            'instruments': {i_idx: [triplet_ids]},
            'verbs': {v_idx: [triplet_ids]},
            'targets': {t_idx: [triplet_ids]}
        }
    """
    n_instruments = len(INSTRUMENTS)
    n_verbs = len(VERBS)
    n_targets = len(TARGETS)

    # 初始化
    comp_to_triplets = {
        'instruments': {i: [] for i in range(n_instruments)},
        'verbs': {v: [] for v in range(n_verbs)},
        'targets': {t: [] for t in range(n_targets)}
    }

    # 遍历所有triplet
    for triplet_idx, (inst, verb, targ) in enumerate(TRIPLET_CLASSES):
        i_idx = INSTRUMENTS.index(inst)
        v_idx = VERBS.index(verb)
        t_idx = TARGETS.index(targ)

        comp_to_triplets['instruments'][i_idx].append(triplet_idx)
        comp_to_triplets['verbs'][v_idx].append(triplet_idx)
        comp_to_triplets['targets'][t_idx].append(triplet_idx)

    return comp_to_triplets


def build_pair_to_triplets():
    """
    构建pair到triplet列表的映射

    Returns:
        dict: {
            'iv': {(i_idx, v_idx): [triplet_ids]},
            'it': {(i_idx, t_idx): [triplet_ids]}
        }
    """
    pair_to_triplets = {
        'iv': {},
        'it': {}
    }

    for triplet_idx, (inst, verb, targ) in enumerate(TRIPLET_CLASSES):
        i_idx = INSTRUMENTS.index(inst)
        v_idx = VERBS.index(verb)
        t_idx = TARGETS.index(targ)

        # IV pair
        iv_key = (i_idx, v_idx)
        if iv_key not in pair_to_triplets['iv']:
            pair_to_triplets['iv'][iv_key] = []
        pair_to_triplets['iv'][iv_key].append(triplet_idx)

        # IT pair
        it_key = (i_idx, t_idx)
        if it_key not in pair_to_triplets['it']:
            pair_to_triplets['it'][it_key] = []
        pair_to_triplets['it'][it_key].append(triplet_idx)

    return pair_to_triplets


def get_component_counts():
    """返回component和pair的数量"""
    pair_map = build_pair_to_triplets()
    return {
        'n_instruments': len(INSTRUMENTS),
        'n_verbs': len(VERBS),
        'n_targets': len(TARGETS),
        'n_iv_pairs': len(pair_map['iv']),
        'n_it_pairs': len(pair_map['it']),
        'n_triplets': len(TRIPLET_CLASSES)
    }


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("Testing triplet_config...")

    # 1. 验证triplet数量
    print(f"\n1. Triplet count: {len(TRIPLET_CLASSES)}")
    assert len(TRIPLET_CLASSES) == 100, "Should have exactly 100 triplets!"

    # 2. 验证component数量
    counts = get_component_counts()
    print(f"\n2. Component counts:")
    print(f"   Instruments: {counts['n_instruments']}")
    print(f"   Verbs: {counts['n_verbs']}")
    print(f"   Targets: {counts['n_targets']}")
    print(f"   IV pairs: {counts['n_iv_pairs']}")
    print(f"   IT pairs: {counts['n_it_pairs']}")

    # 3. 验证映射
    print(f"\n3. Testing mappings...")
    comp_map = build_component_to_triplets()

    # 检查grasper出现在哪些triplet中
    grasper_idx = INSTRUMENTS.index('grasper')
    grasper_triplets = comp_map['instruments'][grasper_idx]
    print(f"   Grasper appears in {len(grasper_triplets)} triplets")
    print(f"   Example triplets: {grasper_triplets[:5]}")

    # 检查retract出现在哪些triplet中
    retract_idx = VERBS.index('retract')
    retract_triplets = comp_map['verbs'][retract_idx]
    print(f"   Retract appears in {len(retract_triplets)} triplets")

    # 检查gallbladder出现在哪些triplet中
    gb_idx = TARGETS.index('gallbladder')
    gb_triplets = comp_map['targets'][gb_idx]
    print(f"   Gallbladder appears in {len(gb_triplets)} triplets")

    # 4. 验证pair映射
    pair_map = build_pair_to_triplets()

    # (grasper, retract)这个pair
    gr_pair = (grasper_idx, retract_idx)
    gr_triplets = pair_map['iv'].get(gr_pair, [])
    print(f"\n4. (grasper, retract) pair has {len(gr_triplets)} triplets")

    # 打印几个示例triplet
    print(f"\n5. Example triplets:")
    for i in range(min(5, len(TRIPLET_CLASSES))):
        inst, verb, targ = TRIPLET_CLASSES[i]
        print(f"   {i}: ({inst}, {verb}, {targ})")

    print("\n✓ All tests passed!")