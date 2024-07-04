import lmdb
import pickle
import numpy as np
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
import torch.cuda
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

import utils
from nnLayer import *
from utils import *

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# print(torch.cuda.is_availab le())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(torch.cuda.current_device()))

# 定义超参数
learning_rate = 0.001
epochs = 10
batch =  64
weightPath = 'model/FinalModel_cv1_copy.pkl'
classNum = 1
feaSize = 8623
filterNum = 128
contextSizeList = [1,9,81]
hiddenSize = 512
num_layers = 3
hiddenList = [2048]
map_location = torch.device('cpu')
device = "cpu"
k = 3
augmentation = 0.05
seq_max_length = 498

# print("这是数据读取模块-----------------------------------------------")
def read_from_lmdb(lmdb_path):
    # 打开LMDB数据库
    env = lmdb.open(lmdb_path, readonly=True)

    # 开始读取事务
    with env.begin() as txn:
        # 读取所有键值对
        cursor = txn.cursor()
        data_dict = {}
        for key, value in cursor:
            # 键需要解码为字符串
            key_str = key.decode('utf-8')
            # 值需要反序列化
            value_obj = pickle.loads(value)
            data_dict[key_str] = value_obj

    # 关闭LMDB数据库
    env.close()
    return data_dict


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


print(f"qwe:{primary_fluorescence_train[0]}")
print(f"ghjk:{primary_fluorescence_test[0]}")

#使用二维蛋白质预测代码中的序列映射列表保证参数一致
# dataClass = DataClass('data_seq_train.txt', 'data_sec_train.txt', k=3, validSize=0.3, minCount=10)
# fluorescenceItem2id = dataClass.seqItem2id
# id2fluorescenceItem = dataClass.id2seqItem
fluorescenceItem2id = {}
id2fluorescenceItem = []
with open('seq_item_mapping.txt', 'r') as f:
    for line in f:
        item, idx = line.strip().split('\t')
        fluorescenceItem2id[item] = int(idx)
        id2fluorescenceItem.append(item)

# 将所有序列扩展到最大长度并使用 <UNK> 进行填充
for i, seq in enumerate(primary_fluorescence_train):
    for q in range((seq_max_length - len(seq))):
           primary_fluorescence_train[i].append("<UNK>")

#制作dataloader
primary_fluorescence_train_encoder = [[fluorescenceItem2id.get(token) for token in sequence] for sequence in primary_fluorescence_train]
dataset = TensorDataset(torch.tensor(primary_fluorescence_train_encoder), torch.tensor(log_fluorescence_train))
dataloader = DataLoader(dataset, batch_size=batch, shuffle=True,drop_last=True)

#加载已经训练好的网络模型参数
stateDict = torch.load(weightPath, map_location=map_location)
seqItem2id,id2seqItem = stateDict['seqItem2id'],stateDict['id2seqItem']
# secItem2id,self.id2secItem = stateDict['secItem2id'],stateDict['id2secItem']
# self.trainIdList,self.validIdList = stateDict['trainIdList'],stateDict['validIdList']
# self.seqItem2id,self.id2seqItem = stateDict['seqItem2id'],stateDict['id2seqItem']
# self.secItem2id,self.id2secItem = stateDict['secItem2id'],stateDict['id2secItem']
textEmbedding = TextEmbedding(torch.zeros((len(id2seqItem),feaSize-8598), dtype=torch.float) ).to(device)
feaEmbedding = TextEmbedding(torch.zeros((len(id2seqItem),8598), dtype=torch.float), freeze=True, name='feaEmbedding' ).to(device)
textCNN = TextCNN(feaSize, contextSizeList, filterNum ).to(device)
textBiGRU = TextBiGRU(len(contextSizeList)*filterNum, hiddenSize, num_layers=num_layers).to(device)
fcLinear = MLP((len(contextSizeList)*filterNum+hiddenSize*2)*seq_max_length, classNum, hiddenList).to(device)
# fcLinear = nn.Linear((len(contextSizeList)*filterNum+hiddenSize*2)*seq_max_length,classNum).to(device)
moduleList = nn.ModuleList([textEmbedding,feaEmbedding,textCNN,textBiGRU,fcLinear])

for module in moduleList[:-1]:  # 遍历除了全连接层以外的所有层
    module.load_state_dict(stateDict[module.name])
    for param in module.parameters():
        param.requires_grad = False  # 冻结参数

# 定义损失函数和优化器
optimizer_fluorescence = optim.Adam(fcLinear.parameters(), learning_rate)
criteon = torch.nn.MSELoss()

# 存储每次epoch的平均spearman和最大spearman
epoch_results = []

for epoch in range(epochs):
    fcLinear.train()
    data_loader = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}')
    average_spearman = []
    max_spearman = -1
    max_p_value = None

    for batch_idx, (input, target) in enumerate(data_loader):
        input, target = input.to(device), target.to(device)  # 将数据移动到 GPU
        input = input.long()
        input = torch.cat([textEmbedding(input), feaEmbedding(input)], dim=2)  # => batchSize × seqLen × feaSize
        input_conved = textCNN(input)  # => batchSize × seqLen × scaleNum*filterNum
        input_BiGRUed = textBiGRU(input_conved, None)  # => batchSize × seqLen × hiddenSize*2
        input = torch.cat([input_conved, input_BiGRUed], dim=2)  # => batchSize × seqLen × (scaleNum*filterNum+hiddenSize*2)
        input = input.reshape(batch, -1)
        result = fcLinear(input)  # => batchSize × seqLen × classNum
        loss = criteon(result, target)
        optimizer_fluorescence.zero_grad()
        loss.backward()
        optimizer_fluorescence.step()

        result_np = result.detach().numpy()
        target_np = target.detach().numpy()
        result_np = np.reshape(result_np, (batch, -1))
        spearman_corr, p_value = spearmanr(result_np, target_np)

        average_spearman.append(spearman_corr)
        max_spearman = max(max_spearman, spearman_corr)


        if spearman_corr == max_spearman:
            max_p_value = p_value
            print(f"恭喜你！spearman系数又增加了，spearman:{spearman_corr},p_value为:{p_value}",end='\n')

    # 计算平均Spearman相关系数
    avg_spearman = np.mean(average_spearman)
    print(f"Epoch {epoch + 1}/{epochs} 平均Spearman相关系数: {avg_spearman}")

    # 打印最高的Spearman相关系数和对应的p-value
    print(f"Epoch {epoch + 1}/{epochs} 最高Spearman相关系数: {max_spearman}, 对应的p-value: {max_p_value}")

    # 将当前epoch得到的最大Spearman相关系数、平均Spearman相关系数和对应的p-value存储起来
    epoch_results.append((max_spearman, avg_spearman, max_p_value))






