import os
import shutil
from collections import defaultdict
import random

# 设置随机种子以确保可复现性
random.seed(42)

# 定义文件路径
base_dir = 'currentAIG2'
aig_dir = os.path.join(base_dir, 'aig')
label_dir = os.path.join(base_dir, 'label')

train_dir = os.path.join(base_dir, 'train')
train_label_dir = os.path.join(base_dir, 'train_label')
test_dir = os.path.join(base_dir, 'test')
test_label_dir = os.path.join(base_dir, 'test_label')

# 创建所需的文件夹
os.makedirs(train_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# 读取所有aig文件并按电路名分组
file_groups = defaultdict(list)
for filename in os.listdir(aig_dir):
    if filename.endswith('.aig'):
        circuit_name = filename.split('_')[0]
        file_groups[circuit_name].append(filename)

# 对每个组划分90%训练集和10%测试集，并移动文件
for circuit_name, files in file_groups.items():
    random.shuffle(files)
    split_index = int(0.9 * len(files))
    train_files = files[:split_index]
    test_files = files[split_index:]
    
    # 移动训练文件
    for filename in train_files:
        shutil.copy(os.path.join(aig_dir, filename), os.path.join(train_dir, filename))
        # 移动对应的pkl文件
        pkl_filename = filename.replace('.aig', '.pkl')
        shutil.copy(os.path.join(label_dir, pkl_filename), os.path.join(train_label_dir, pkl_filename))
    
    # 移动测试文件
    for filename in test_files:
        shutil.copy(os.path.join(aig_dir, filename), os.path.join(test_dir, filename))
        # 移动对应的pkl文件
        pkl_filename = filename.replace('.aig', '.pkl')
        shutil.copy(os.path.join(label_dir, pkl_filename), os.path.join(test_label_dir, pkl_filename))

print("文件已成功分配和移动。")