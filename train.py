from utils import *
from DL_ClassifierModel import *
from SecondStructurePredictor import *
from metrics import *

# 初始化数据类
# dataClass = DataClass('data_seq_train.txt', 'data_sec_train.txt', k=3, validSize=0.3, minCount=10)
# # 词向量预训练
# dataClass.vectorize(method='char2vec', feaSize=25, sg=1)
# # onehot+理化特征获取
# dataClass.vectorize(method='feaEmbedding')
# # # 初始化模型对象
# model = FinalModel(classNum=dataClass.classNum, embedding=dataClass.vector['embedding'], feaEmbedding=dataClass.vector['feaEmbedding'],
#                    useFocalLoss=True, device=torch.device('cuda'))
# # # 开始训练
# model.cv_train( dataClass, trainSize=64, batchSize=64, epoch=10, stopRounds=1, earlyStop=1, saveRounds=1,
#                 savePath='model/FinalModel', lr=0.001, augmentation=0.1, kFold=3)
# # 预测, 得到的输出是一个N × L × C的矩阵，N为样例数，L为序列最大长度，C为类别数，即得到的是各序列各位置得到各类别上的概率。
# model = Predictor_final('model/FinalModel_cv1_699.pkl', device='cpu', map_location=torch.device('cpu'))
# model.predict('data_seq_train.txt', batchSize=64)
import torch
print(torch.__version__)
import torchvision
print(torchvision.__version__)
import gensim
print(gensim.__version__)





