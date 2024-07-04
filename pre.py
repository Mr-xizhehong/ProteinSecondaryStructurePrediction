import os
import lmdb
import pickle
import numpy as np
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torch.cuda
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from Bio.PDB import PDBParser

# 检查 GPU 是否可用
device = torch.device("cuda")

# 定义超参数
learning_rate = 0.003
epochs = 2000
batch_size = 64


# 读取数据的函数
def read_from_lmdb(lmdb_path):
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        data_dict = {}
        for key, value in cursor:
            key_str = key.decode('utf-8')
            value_obj = pickle.loads(value)
            data_dict[key_str] = value_obj
    env.close()
    return data_dict


def parse_pdb(file_path):
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure('protein', file_path)
    return structure


def extract_atom_coordinates(structure):
    atom_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_coords.append(atom.get_coord())
    return atom_coords


def read_ent_files_from_directory(directory):
    atom_coords_list = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".ent"):
            file_path = os.path.join(directory, filename)
            structure = parse_pdb(file_path)
            atom_coords = extract_atom_coordinates(structure)
            atom_coords_list.append(atom_coords)
            filenames.append(filename)
    return atom_coords_list, filenames


# 从LMDB读取数据集
fluorescence_train = read_from_lmdb("./dataset/fluorescence/fluorescence_train.lmdb")
stability_train = read_from_lmdb("./dataset/stability/stability_train.lmdb")
fluorescence_test = read_from_lmdb("./dataset/fluorescence/fluorescence_test.lmdb")
stability_test = read_from_lmdb("./dataset/stability/stability_test.lmdb")
fluorescence_valid = read_from_lmdb("./dataset/fluorescence/fluorescence_valid.lmdb")
stability_valid = read_from_lmdb("./dataset/stability/stability_valid.lmdb")

# 提取训练集的适应度分数(即label)  和  序列(即input)
log_fluorescence_train = [v["log_fluorescence"] for v in fluorescence_train.values() if isinstance(v, dict)]
primary_fluorescence_train = [v["primary"] for v in fluorescence_train.values() if isinstance(v, dict)]
stability_score_train = [v["stability_score"] for v in stability_train.values() if isinstance(v, dict)]
primary_stability_train = [v["primary"] for v in stability_train.values() if isinstance(v, dict)]

# 提取测试集的适应度分数(即label)  和  序列(即input)
log_fluorescence_test = [v["log_fluorescence"] for v in fluorescence_test.values() if isinstance(v, dict)]
primary_fluorescence_test = [v["primary"] for v in fluorescence_test.values() if isinstance(v, dict)]
stability_score_test = [v["stability_score"] for v in stability_test.values() if isinstance(v, dict)]
primary_stability_test = [v["primary"] for v in stability_test.values() if isinstance(v, dict)]

# 提取验证集的适应度分数(即label)  和  序列(即input)
log_fluorescence_valid = [v["log_fluorescence"] for v in fluorescence_valid.values() if isinstance(v, dict)]
primary_fluorescence_valid = [v["primary"] for v in fluorescence_valid.values() if isinstance(v, dict)]
stability_score_valid = [v["stability_score"] for v in stability_valid.values() if isinstance(v, dict)]
primary_stability_valid = [v["primary"] for v in stability_valid.values() if isinstance(v, dict)]

# 读取指定文件夹下的所有 .ent 文件的原子坐标
ent_directory = r"C:\Users\java~python\Desktop\P42212_archive-PDB"
atom_coords_list, filenames = read_ent_files_from_directory(ent_directory)

# 检查读取到的原子坐标列表
print(f"Number of .ent files read: {len(atom_coords_list)}")
print(f"Size of first element before flattening: {np.array(atom_coords_list[0]).shape}")

# 找到最长的坐标列表长度
max_length = max(len(coords) for coords in atom_coords_list) * 3

# 将每个原子坐标列表进行填充到相同长度
atom_coords_array = np.array(
    [np.pad(np.array(coords), ((0, max_length // 3 - len(coords)), (0, 0)), 'constant') for coords in atom_coords_list])

# 检查转换后的数组
print(f"Shape of atom_coords_array: {atom_coords_array.shape}")

# 转换为 PyTorch 张量
atom_coordinates_tensors = torch.tensor(atom_coords_array, dtype=torch.float32)
log_fluorescence_train_tensor = torch.tensor(log_fluorescence_train[:len(atom_coordinates_tensors)],
                                             dtype=torch.float32)

# 制作 DataLoader
dataset = TensorDataset(atom_coordinates_tensors, log_fluorescence_train_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 检查 DataLoader 中的张量维度
for data in dataloader:
    atom_coords_batch, log_fluorescence_batch = data
    print(f"Shape of atom_coords_batch: {atom_coords_batch.shape}")
    print(f"Shape of log_fluorescence_batch: {log_fluorescence_batch.shape}")
    break


# 简单的神经网络模型（加入卷积层）
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # 计算卷积和池化后的尺寸
        self._to_linear = None
        self.convs(torch.rand(1, 3, max_length))  # 运行一次 forward 来计算 _to_linear

        self.fc1 = torch.nn.Linear(self._to_linear, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def convs(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self._to_linear is None:
            self._to_linear = x.shape[1] * x.shape[2]
        return x

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整维度顺序以匹配 Conv1d 的输入
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN().to(device)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
model.train()
for epoch in range(epochs):
    for atom_coords_batch, log_fluorescence_batch in tqdm(dataloader):
        atom_coords_batch, log_fluorescence_batch = atom_coords_batch.to(device), log_fluorescence_batch.to(device)
        optimizer.zero_grad()
        outputs = model(atom_coords_batch)
        loss = criterion(outputs.squeeze(), log_fluorescence_batch)
        loss.backward()
        optimizer.step()

    # 每10次 epoch 输出一次信息
    if (epoch + 1) % 10 == 0:
        model.eval()
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for atom_coords_batch, log_fluorescence_batch in dataloader:
                atom_coords_batch, log_fluorescence_batch = atom_coords_batch.to(device), log_fluorescence_batch.to(
                    device)
                outputs = model(atom_coords_batch)
                all_outputs.extend(outputs.cpu().numpy().squeeze())
                all_targets.extend(log_fluorescence_batch.cpu().numpy())

        # 转换为 numpy 数组
        all_outputs = np.array(all_outputs)
        all_targets = np.array(all_targets)

        # 检查是否有 NaN 值
        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(f"Number of NaNs in all_outputs: {np.isnan(all_outputs).sum()}")
        print(f"Number of NaNs in all_targets: {np.isnan(all_targets).sum()}")

        # 计算 Spearman 相关系数
        spearman_corr, p_value = spearmanr(all_outputs, all_targets)
        print(f"Spearman Correlation: {spearman_corr}, p-value: {p_value}")

        # 输出预测值和实际值
        print(f"Predicted values: {all_outputs[:5]}")
        print(f"Actual values: {all_targets[:5]}")

        model.train()
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

# 最终评估
model.eval()
all_outputs = []
all_targets = []
with torch.no_grad():
    for atom_coords_batch, log_fluorescence_batch in dataloader:
        atom_coords_batch, log_fluorescence_batch = atom_coords_batch.to(device), log_fluorescence_batch.to(device)
        outputs = model(atom_coords_batch)
        all_outputs.extend(outputs.cpu().numpy().squeeze())
        all_targets.extend(log_fluorescence_batch.cpu().numpy())

# 转换为 numpy 数组
all_outputs = np.array(all_outputs)
all_targets = np.array(all_targets)

# 检查是否有 NaN 值
print(f"Number of NaNs in all_outputs: {np.isnan(all_outputs).sum()}")
print(f"Number of NaNs in all_targets: {np.isnan(all_targets).sum()}")

# 计算 Spearman 相关系数
spearman_corr, p_value = spearmanr(all_outputs, all_targets)
print(f"Spearman Correlation: {spearman_corr}, p-value: {p_value}")

# 对测试集进行评估
log_fluorescence_test_tensor = torch.tensor(log_fluorescence_test[:len(atom_coordinates_tensors)], dtype=torch.float32)

test_dataset = TensorDataset(atom_coordinates_tensors, log_fluorescence_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 使用训练好的模型进行预测并计算 Spearman 相关系数和 p-value
test_outputs = []
test_targets = []

with torch.no_grad():
    for atom_coords_batch, log_fluorescence_batch in tqdm(test_dataloader):
        atom_coords_batch, log_fluorescence_batch = atom_coords_batch.to(device), log_fluorescence_batch.to(device)
        outputs = model(atom_coords_batch)
        test_outputs.extend(outputs.cpu().numpy().squeeze())
        test_targets.extend(log_fluorescence_batch.cpu().numpy())

# 转换为 numpy 数组
test_outputs = np.array(test_outputs)
test_targets = np.array(test_targets)

# 检查是否有 NaN 值
print(f"Number of NaNs in test_outputs: {np.isnan(test_outputs).sum()}")
print(f"Number of NaNs in test_targets: {np.isnan(test_targets).sum()}")

# 计算 Spearman 相关系数和 p-value
spearman_corr_test, p_value_test = spearmanr(test_outputs, test_targets)
print(f"Spearman Correlation (Test): {spearman_corr_test}, p-value: {p_value_test}")

# import os
# import lmdb
# import pickle
# import numpy as np
# import torch
# from torch import optim
# from torch.utils.data import TensorDataset, DataLoader
# from tqdm import tqdm
# import torch.nn.functional as F
# import torch.cuda
# from matplotlib import pyplot as plt
# from scipy.stats import spearmanr
# from Bio.PDB import PDBParser
#
# # 检查 GPU 是否可用
# device = torch.device("cuda")
#
# # 定义超参数
# learning_rate = 0.003
# epochs = 2000
# batch_size = 64
#
# # 读取数据的函数
# def read_from_lmdb(lmdb_path):
#     env = lmdb.open(lmdb_path, readonly=True)
#     with env.begin() as txn:
#         cursor = txn.cursor()
#         data_dict = {}
#         for key, value in cursor:
#             key_str = key.decode('utf-8')
#             value_obj = pickle.loads(value)
#             data_dict[key_str] = value_obj
#     env.close()
#     return data_dict
#
# def parse_pdb(file_path):
#     parser = PDBParser(PERMISSIVE=1)
#     structure = parser.get_structure('protein', file_path)
#     return structure
#
# def extract_atom_coordinates(structure):
#     atom_coords = []
#     for model in structure:
#         for chain in model:
#             for residue in chain:
#                 for atom in residue:
#                     atom_coords.append(atom.get_coord())
#     return atom_coords
#
# def read_ent_files_from_directory(directory):
#     atom_coords_list = []
#     filenames = []
#     for filename in os.listdir(directory):
#         if filename.endswith(".ent"):
#             file_path = os.path.join(directory, filename)
#             structure = parse_pdb(file_path)
#             atom_coords = extract_atom_coordinates(structure)
#             atom_coords_list.append(atom_coords)
#             filenames.append(filename)
#     return atom_coords_list, filenames
#
# # 从LMDB读取数据集
# fluorescence_train = read_from_lmdb("./dataset/fluorescence/fluorescence_train.lmdb")
# stability_train = read_from_lmdb("./dataset/stability/stability_train.lmdb")
# fluorescence_test = read_from_lmdb("./dataset/fluorescence/fluorescence_test.lmdb")
# stability_test = read_from_lmdb("./dataset/stability/stability_test.lmdb")
# fluorescence_valid = read_from_lmdb("./dataset/fluorescence/fluorescence_valid.lmdb")
# stability_valid = read_from_lmdb("./dataset/stability/stability_valid.lmdb")
#
# # 提取训练集的适应度分数(即label)  和  序列(即input)
# log_fluorescence_train = [v["log_fluorescence"] for v in fluorescence_train.values() if isinstance(v, dict)]
# primary_fluorescence_train = [v["primary"] for v in fluorescence_train.values() if isinstance(v, dict)]
# stability_score_train = [v["stability_score"] for v in stability_train.values() if isinstance(v, dict)]
# primary_stability_train = [v["primary"] for v in stability_train.values() if isinstance(v, dict)]
#
# # 提取测试集的适应度分数(即label)  和  序列(即input)
# log_fluorescence_test = [v["log_fluorescence"] for v in fluorescence_test.values() if isinstance(v, dict)]
# primary_fluorescence_test = [v["primary"] for v in fluorescence_test.values() if isinstance(v, dict)]
# stability_score_test = [v["stability_score"] for v in stability_test.values() if isinstance(v, dict)]
# primary_stability_test = [v["primary"] for v in stability_test.values() if isinstance(v, dict)]
#
# # 提取验证集的适应度分数(即label)  和  序列(即input)
# log_fluorescence_valid = [v["log_fluorescence"] for v in fluorescence_valid.values() if isinstance(v, dict)]
# primary_fluorescence_valid = [v["primary"] for v in fluorescence_valid.values() if isinstance(v, dict)]
# stability_score_valid = [v["stability_score"] for v in stability_valid.values() if isinstance(v, dict)]
# primary_stability_valid = [v["primary"] for v in stability_valid.values() if isinstance(v, dict)]
#
# # 读取指定文件夹下的所有 .ent 文件的原子坐标
# ent_directory = r"C:\Users\java~python\Desktop\P42212_archive-PDB"
# atom_coords_list, filenames = read_ent_files_from_directory(ent_directory)
#
# # 检查读取到的原子坐标列表
# print(f"Number of .ent files read: {len(atom_coords_list)}")
# print(f"Size of first element before flattening: {np.array(atom_coords_list[0]).shape}")
#
# # 找到最长的坐标列表长度
# max_length = max(len(coords) for coords in atom_coords_list) * 3
#
# # 将每个原子坐标列表进行 flatten 操作并填充到相同长度
# atom_coords_array = np.array(
#     [np.pad(np.array(coords).flatten(), (0, max_length - len(coords) * 3)) for coords in atom_coords_list])
#
# # 检查转换后的数组
# print(f"Shape of atom_coords_array: {atom_coords_array.shape}")
#
# # 转换为 PyTorch 张量
# atom_coordinates_tensors = torch.tensor(atom_coords_array, dtype=torch.float32)
# log_fluorescence_train_tensor = torch.tensor(log_fluorescence_train[:len(atom_coordinates_tensors)],
#                                              dtype=torch.float32)
#
# # 制作 DataLoader
# dataset = TensorDataset(atom_coordinates_tensors, log_fluorescence_train_tensor)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#
# # 检查 DataLoader 中的张量维度
# for data in dataloader:
#     atom_coords_batch, log_fluorescence_batch = data
#     print(f"Shape of atom_coords_batch: {atom_coords_batch.shape}")
#     print(f"Shape of log_fluorescence_batch: {log_fluorescence_batch.shape}")
#     break
#
# # 简单的神经网络模型
# class SimpleNN(torch.nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc1 = torch.nn.Linear(max_length, 128)
#         self.fc2 = torch.nn.Linear(128, 1)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
# model = SimpleNN().to(device)
# criterion = torch.nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# # 训练模型
# model.train()
# for epoch in range(epochs):
#     for atom_coords_batch, log_fluorescence_batch in tqdm(dataloader):
#         atom_coords_batch, log_fluorescence_batch = atom_coords_batch.to(device), log_fluorescence_batch.to(device)
#         optimizer.zero_grad()
#         outputs = model(atom_coords_batch)
#         loss = criterion(outputs.squeeze(), log_fluorescence_batch)
#         loss.backward()
#         optimizer.step()
#
#     # 每10次 epoch 输出一次信息
#     if (epoch + 1) % 10 == 0:
#         model.eval()
#         all_outputs = []
#         all_targets = []
#         with torch.no_grad():
#             for atom_coords_batch, log_fluorescence_batch in dataloader:
#                 atom_coords_batch, log_fluorescence_batch = atom_coords_batch.to(device), log_fluorescence_batch.to(device)
#                 outputs = model(atom_coords_batch)
#                 all_outputs.extend(outputs.cpu().numpy().squeeze())
#                 all_targets.extend(log_fluorescence_batch.cpu().numpy())
#
#         # 转换为 numpy 数组
#         all_outputs = np.array(all_outputs)
#         all_targets = np.array(all_targets)
#
#         # 检查是否有 NaN 值
#         print(f"Epoch [{epoch + 1}/{epochs}]")
#         print(f"Number of NaNs in all_outputs: {np.isnan(all_outputs).sum()}")
#         print(f"Number of NaNs in all_targets: {np.isnan(all_targets).sum()}")
#
#         # 计算 Spearman 相关系数
#         spearman_corr, p_value = spearmanr(all_outputs, all_targets)
#         print(f"Spearman Correlation: {spearman_corr}, p-value: {p_value}")
#
#         # 输出预测值和实际值
#         print(f"Predicted values: {all_outputs[:5]}")
#         print(f"Actual values: {all_targets[:5]}")
#
#         model.train()
#     print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")
#
# # 最终评估
# model.eval()
# all_outputs = []
# all_targets = []
# with torch.no_grad():
#     for atom_coords_batch, log_fluorescence_batch in dataloader:
#         atom_coords_batch, log_fluorescence_batch = atom_coords_batch.to(device), log_fluorescence_batch.to(device)
#         outputs = model(atom_coords_batch)
#         all_outputs.extend(outputs.cpu().numpy().squeeze())
#         all_targets.extend(log_fluorescence_batch.cpu().numpy())
#
# # 转换为 numpy 数组
# all_outputs = np.array(all_outputs)
# all_targets = np.array(all_targets)
#
# # 检查是否有 NaN 值
# print(f"Number of NaNs in all_outputs: {np.isnan(all_outputs).sum()}")
# print(f"Number of NaNs in all_targets: {np.isnan(all_targets).sum()}")
#
# # 计算 Spearman 相关系数
# spearman_corr, p_value = spearmanr(all_outputs, all_targets)
# print(f"Spearman Correlation: {spearman_corr}, p-value: {p_value}")
#
# # 对测试集进行评估
# log_fluorescence_test_tensor = torch.tensor(log_fluorescence_test[:len(atom_coordinates_tensors)], dtype=torch.float32)
#
# test_dataset = TensorDataset(atom_coordinates_tensors, log_fluorescence_test_tensor)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#
# # 使用训练好的模型进行预测并计算 Spearman 相关系数和 p-value
# test_outputs = []
# test_targets = []
#
# with torch.no_grad():
#     for atom_coords_batch, log_fluorescence_batch in tqdm(test_dataloader):
#         atom_coords_batch, log_fluorescence_batch = atom_coords_batch.to(device), log_fluorescence_batch.to(device)
#         outputs = model(atom_coords_batch)
#         test_outputs.extend(outputs.cpu().numpy().squeeze())
#         test_targets.extend(log_fluorescence_batch.cpu().numpy())
#
# # 转换为 numpy 数组
# test_outputs = np.array(test_outputs)
# test_targets = np.array(test_targets)
#
# # 检查是否有 NaN 值
# print(f"Number of NaNs in test_outputs: {np.isnan(test_outputs).sum()}")
# print(f"Number of NaNs in test_targets: {np.isnan(test_targets).sum()}")
#
# # 计算 Spearman 相关系数和 p-value
# spearman_corr_test, p_value_test = spearmanr(test_outputs, test_targets)
# print(f"Spearman Correlation (Test): {spearman_corr_test}, p-value: {p_value_test}")


# import os
# import lmdb
# import pickle
# import numpy as np
# import torch
# from torch import optim
# from torch.utils.data import TensorDataset, DataLoader
# from tqdm import tqdm
# import torch.nn.functional as F
# import torch.cuda
# from matplotlib import pyplot as plt
# from scipy.stats import spearmanr
# from Bio.PDB import PDBParser
#
# # 检查 GPU 是否可用
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 定义超参数
# learning_rate = 0.001
# epochs = 100
# batch_size = 64
#
# # 读取数据的函数
# def read_from_lmdb(lmdb_path):
#     env = lmdb.open(lmdb_path, readonly=True)
#     with env.begin() as txn:
#         cursor = txn.cursor()
#         data_dict = {}
#         for key, value in cursor:
#             key_str = key.decode('utf-8')
#             value_obj = pickle.loads(value)
#             data_dict[key_str] = value_obj
#     env.close()
#     return data_dict
#
# def parse_pdb(file_path):
#     parser = PDBParser(PERMISSIVE=1)
#     structure = parser.get_structure('protein', file_path)
#     return structure
#
# def extract_atom_coordinates(structure):
#     atom_coords = []
#     for model in structure:
#         for chain in model:
#             for residue in chain:
#                 for atom in residue:
#                     atom_coords.append(atom.get_coord())
#     return atom_coords
#
# def read_ent_files_from_directory(directory):
#     atom_coords_list = []
#     filenames = []
#     for filename in os.listdir(directory):
#         if filename.endswith(".ent"):
#             file_path = os.path.join(directory, filename)
#             structure = parse_pdb(file_path)
#             atom_coords = extract_atom_coordinates(structure)
#             atom_coords_list.append(atom_coords)
#             filenames.append(filename)
#     return atom_coords_list, filenames
#
# # 从LMDB读取数据集
# fluorescence_train = read_from_lmdb("./dataset/fluorescence/fluorescence_train.lmdb")
# stability_train = read_from_lmdb("./dataset/stability/stability_train.lmdb")
#
# # 提取训练集的适应度分数(即label) 和 序列(即input)
# log_fluorescence_train = [v["log_fluorescence"] for v in fluorescence_train.values() if isinstance(v, dict)]
#
# # 读取指定文件夹下的所有 .ent 文件的原子坐标
# ent_directory = r"C:\Users\java~python\Desktop\P42212_archive-PDB"
# atom_coords_list, filenames = read_ent_files_from_directory(ent_directory)
#
# # 检查读取到的原子坐标列表
# print(f"Number of .ent files read: {len(atom_coords_list)}")
# print(f"Size of first element before flattening: {np.array(atom_coords_list[0]).shape}")
#
# # 找到最长的坐标列表长度
# max_length = max(len(coords) for coords in atom_coords_list) * 3
#
# # 将每个原子坐标列表进行 flatten 操作并填充到相同长度
# atom_coords_array = np.array([np.pad(np.array(coords).flatten(), (0, max_length - len(coords) * 3)) for coords in atom_coords_list])
#
# # 检查转换后的数组
# print(f"Shape of atom_coords_array: {atom_coords_array.shape}")
#
# # 转换为 PyTorch 张量
# atom_coordinates_tensors = torch.tensor(atom_coords_array, dtype=torch.float32)
#
# # 扩展 log_fluorescence_train
# expanded_log_fluorescence = torch.tensor(log_fluorescence_train * len(atom_coordinates_tensors), dtype=torch.float32)
#
# # 制作 DataLoader
# dataset = TensorDataset(atom_coordinates_tensors.repeat(len(log_fluorescence_train), 1), expanded_log_fluorescence)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#
# # 检查 DataLoader 中的张量维度
# for data in dataloader:
#     atom_coords_batch, log_fluorescence_batch = data
#     print(f"Shape of atom_coords_batch: {atom_coords_batch.shape}")
#     print(f"Shape of log_fluorescence_batch: {log_fluorescence_batch.shape}")
#     break
#
# # 简单的神经网络模型
# class SimpleNN(torch.nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc1 = torch.nn.Linear(max_length, 128)
#         self.fc2 = torch.nn.Linear(128, 1)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
# model = SimpleNN().to(device)
# criterion = torch.nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# # 训练模型
# model.train()
# for epoch in range(epochs):
#     for atom_coords_batch, log_fluorescence_batch in tqdm(dataloader):
#         atom_coords_batch, log_fluorescence_batch = atom_coords_batch.to(device), log_fluorescence_batch.to(device)
#         optimizer.zero_grad()
#         outputs = model(atom_coords_batch)
#         loss = criterion(outputs.squeeze(), log_fluorescence_batch)
#         loss.backward()
#         optimizer.step()
#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")
#
# # 预测并计算 Spearman 相关系数
# model.eval()
# all_outputs = []
# all_targets = []
# with torch.no_grad():
#     for atom_coords_batch, log_fluorescence_batch in dataloader:
#         atom_coords_batch, log_fluorescence_batch = atom_coords_batch.to(device), log_fluorescence_batch.to(device)
#         outputs = model(atom_coords_batch)
#         all_outputs.extend(outputs.cpu().numpy().squeeze())
#         all_targets.extend(log_fluorescence_batch.cpu().numpy())
#
# # 转换为 numpy 数组
# all_outputs = np.array(all_outputs)
# all_targets = np.array(all_targets)
#
# # 打印输出和目标
# print(f"all_outputs: {all_outputs[:10]}")  # 只打印前10个元素，方便查看
# print(f"all_targets: {all_targets[:10]}")  # 只打印前10个元素，方便查看
#
# # 检查是否有 NaN 值
# print(f"Number of NaNs in all_outputs: {np.isnan(all_outputs).sum()}")
# print(f"Number of NaNs in all_targets: {np.isnan(all_targets).sum()}")
#
# # 检查是否为常数值
# print(f"Variance in all_outputs: {np.var(all_outputs)}")
# print(f"Variance in all_targets: {np.var(all_targets)}")
#
# # 计算 Spearman 相关系数
# if np.var(all_outputs) == 0 or np.var(all_targets) == 0:
#     print("One of the arrays is constant, Spearman correlation is not defined.")
# else:
#     spearman_corr, p_value = spearmanr(all_outputs, all_targets)
#     print(f"Spearman Correlation: {spearman_corr}, p-value: {p_value}")
