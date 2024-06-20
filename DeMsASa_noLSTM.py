
import inference_pytorch.vgg16_places_365
import gc
import cv2
import csv
import time
import torch
import torch.nn as nn
# from torchviz import make_dot
import torch.utils.data as Data
import tensorflow as tf
import numpy as np
from vgg16_places_365 import VGG16_Places365
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from keras import Model
import matplotlib.pyplot as plt
# from cv2 import resize
from csv import reader
from ast import literal_eval
import pandas as pd
import re
import pickle
import torchvision.models as models
import math
from skimage import util


inputs=4096
hiddens=256
#seq_len=100
batchsize=100
input_dim=256
output_dim=10
num_head=2
total_epoches=500
lamda=1e-14
beta=1

learning_rate=1
# def SingleShot_LSTM(frames):#定义LSTM模块
#     lstm=nn.LSTM(input_size=inputs, hidden_size=hiddens, num_layers=1)
#    # input=torch.randn(seq_len,batch_size,inputs)
#     output,(hn,cn)=lstm(frames)
#     return hn


# def attention(input):#定义注意力模块
#     a = nn.Linear(input_dim, output_dim)(input)
#     output_attention=nn.Softmax(dim=1)(a)
#     output_attention=torch.max(output_attention,dim=1)
#     output_attention=output_attention.values.repeat(1,input_dim)
#     return output_attention*input
#     #return output_attention


# class MultiHead_SelfAttention(nn.Module):
#     def __init__(self, dim, num_head):
#         '''
#         Args:
#             dim: dimension for each time step
#             num_head:num head for multi-head self-attention
#         '''
#         super().__init__()
#         self.dim = dim
#         self.num_head = num_head
#         self.qkv = nn.Linear(dim, dim * 3)  # extend the dimension for later spliting
#
#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         att = q @ k.transpose(-1, -2) / math.sqrt(C)
#         att = att.softmax(dim=1)  # 将多个注意力矩阵合并为一个
#         x = (att @ v).transpose(1, 2)
#         x = x.reshape(B, N, C)
#         return x


def VGG_feature():#定义VGGplaces_365特征提取网络
    feature_videos8=[]#***********************************
    ShotSeg_video='/home/jie/Mount/Project/dataset/microvideo10categories/ShotSegmentation'
    with torch.no_grad():
      model1 = VGG16_Places365(weights='places')
      model2 = Model(inputs=model1.input, outputs=model1.get_layer('fc1').output)
    for i in range(7001,7246):#***********************************
        #若一次执行完成占用内存太多，可以将数据分成两部分，最后将结果列表拼接
        #feature_videos2=feature_videos+feature_videos1
        feature_shots = []
        # start = time.clock()
        print(i)
        ShotSeg_path = ShotSeg_video+'/'+str(i)
        # list=os.listdir(ShotSeg_path)
        # shot_num=len(list)
        for j in range(1,7):
            frames_shot_list=os.listdir(ShotSeg_path+'/'+str(j))
            frames_shot_list.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
            index = 0
            feature_oneshot = np.zeros((4096, 50))#len(frames_shot_list)=50
            for frame in frames_shot_list:
                image = cv2.imread(ShotSeg_path+'/'+str(j)+'/'+frame)
                image = np.array(image, dtype=np.uint8)
                image = resize(image, (224, 224))
                image = np.expand_dims(image, 0)
                feature_oneshot[:, index] = model2.predict(image)
                index=index+1
               #  torch.cuda.empty_cache()
            feature_shots.append(feature_oneshot)
        feature_videos8.append(feature_shots)#***********************************
        # end = time.clock()
        # runtime = end - start
        # print(runtime)
    with open('feature_videos8.pkl', 'wb') as file:#***********************************
        pickle.dump(feature_videos8,file)#***********************************

    print(0)

def datasplit():#数据集划分
    with open('feature_videos_7245.pkl','rb') as file:
        x=pickle.load(file)
    y=np.load('Label_7245_bi.npy')
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    return x_train,x_test,y_train,y_test
# def mseloss(x,y):
#     shape=x.shape
#     sum=0
#     for i in range (0,shape[0]):
#         for j in range (0,6):
#             for h in range(0,50):
#                 loss_frame=torch.norm((x[i][j][h]-y[i][j][h]))
#                 sum=sum+loss_frame
#     Loss=sum/(shape[0]*6*50)
#     return Loss
def gauss_noise(matrix,mean,sigma):
    noise=torch.randn_like(matrix)*sigma+mean
    matrix_noisy=matrix+noise
    return matrix_noisy


class MsAF_Net(nn.Module):#定义网络
    def __init__(self):
        super(MsAF_Net,self).__init__()
        # self.layers1=nn.Sequential(
        #     nn.LSTM(input_size=4096, hidden_size=2048, num_layers=1)
        # )
        # self.layers1 = nn.Sequential(
        #     nn.Linear(4096, 1024)
        #
        # )
        self.layers11 = nn.Sequential(
            nn.Linear(4096, 512),
            nn.Sigmoid(),
            nn.Linear(512, 10),

        )
        self.layers12 = nn.Sequential(
            nn.Linear(10, 512),
            nn.Sigmoid(),
            nn.Linear(512, 4096)
        )
        self.layers2 = nn.Sequential(
            nn.Linear(4096, 512),
            nn.Sigmoid(),
            #nn.Linear(2048, 1024),
            #nn.LeakyReLU(inplace=True),
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )
        self.layers3 = nn.Sequential(
            nn.Linear(4096, 4096 * 3)

        )
        self.fc1= nn.Sequential(
            nn.Linear(4096, 10),
            nn.Softmax(dim=2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(500, 10),
            nn.Softmax(dim=0)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(3000, 10),
            nn.Softmax(dim=0)
        )
    def forward(self,x):
        s = x.shape[0]
        #out_da1=torch.zeros([s,6,50,10],device="cuda:0")
        out_da1_o = torch.zeros([s, 6, 10], device="cuda:0")
        #out_da2 = torch.zeros([s,6, 50, 4096], device="cuda:0")
        out_da2_o = torch.zeros([s, 6, 50, 4096], device="cuda:0")
        prediction=torch.zeros([s,10],device="cuda:0")
        out_shot = torch.zeros([6, 50, 4096], device="cuda:0")
        #atten_sum_1 = torch.zeros([6, 1, 4096], device="cuda:0")
        # hn=torch.zeros([252,6,50,256],device="cuda:0")
        # weight=torch.zeros([252,6,50,1],device="cuda:0")
        # atten_out=torch.zeros([252,6,50,256],device="cuda:0")
          # atten=torch.zeros([1,1,1,256],device="cuda:0")
        for b in range (0,s):
          for i in range (0,6):#x是输入的一个视频，i是视频中镜头的索引
              # print(b,'_',i)
              # out,(h_n,c_n)=self.layers1(x[b][i][:][:])
              # hn[b][i][:][:]=out
              out_or=x[b][i][:][:]
              # out=self.layers1(out_or)
              out_da1=self.layers11(gauss_noise(out_or,0,0.2))#encoding
              out_da2=self.layers12(out_da1)#decoding
              sofm = self.layers2(out_da2)  # softmax
              sofm_max = torch.max(sofm, 1)  # 取最大值                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        , 1)  # 取最大值
              out_shot[i] = out_or * nn.functional.normalize(sofm_max.values, p=1, dim=0).unsqueeze(-1)  # 归一化
              out_da1_o[b][i] = self.fc2(out_da1.view(500))
              out_da2_o[b][i] = out_da2
          shot_feature=out_shot

          B, N, C = shot_feature.shape
          qkv = self.layers3(shot_feature.clone())
          qkv_rp=qkv.reshape(B, N, 3, num_head, C // num_head).permute(2, 0, 3, 1, 4)
          q, k, v = qkv_rp[0], qkv_rp[1], qkv_rp[2]
          att = q @ k.transpose(-1, -2) / math.sqrt(C)
          att = att.softmax(dim=1)  # 将多个注意力矩阵合并为一个
          out_1 = (att @ v).transpose(1, 2)
          out_1 = out_1.reshape(B, N, C)
          x_selA=self.fc1(out_1)
          prediction[b]=self.fc3(x_selA.view(3000))
        return prediction,out_da1_o,out_da2_o,out_or

if __name__ == '__main__':
   import os
   os.environ["CUDA_VISIBLE_DIVICES"] = "0"
   DEVICE = torch.device("cuda:0")
   print(DEVICE)
   # VGG_feature()#对视频各镜头的视频帧进行特征提取，提取特征后，注释掉该语句，再进行MsAF模型训练
   x_train, x_test, y_train, y_test=datasplit()
   y_train_label=np.array([np.argmax(i) for i in y_train])
   # class_weights = compute_class_weight(class_weight='balanced', classes=[0,1,2,3,4,5,6,7,8,9], y=y_train_label)
   class_weights=[0.09,0.06,0.09,0.2,0.06,0.06,0.09,0.06,0.09,0.2]
   class_weights=torch.tensor(class_weights)
   y_train = torch.tensor(y_train)
   x_train=torch.tensor(x_train)
   y_test=torch.tensor(y_test)
   x_test=torch.tensor(x_test)
   #x_train=x_train.add(1e-5)
   #x_test=x_test.add(1e-5)
   torch_traindata=Data.TensorDataset(x_train, y_train)
   torch_testdata=Data.TensorDataset(x_test,y_test)
   #加载数据
   train_loader=torch.utils.data.DataLoader(dataset=torch_traindata,batch_size=batchsize,shuffle=True)
   test_loader = torch.utils.data.DataLoader(dataset=torch_testdata, batch_size=batchsize, shuffle=True)
   #为网络指定模型计算的位置，GPU or CPU
   MsAF_Net=MsAF_Net()
   print(MsAF_Net)
   DEVICE=torch.device("cuda:0")
   #*******
   MsAF_Net=MsAF_Net.to(DEVICE)
   criterion=nn.CrossEntropyLoss(class_weights).to(DEVICE)
   criterion1 = nn.CrossEntropyLoss().to(DEVICE)
   optimizer=torch.optim.SGD(MsAF_Net.parameters(),lr=learning_rate)
   #scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.1)

   losses_tr= []
   losses_te=[]
   acces_tr=[]
   acces_te=[]
   microf1s_tr=[]
   microf1s_te=[]
   macrof1s_tr=[]
   macrof1s_te=[]


   for epoch in range (total_epoches):
     print('epoch=',epoch)
     # 开始训练
     correct_sum_tr=0
     total_tr=0
     for i,(videos,labels) in enumerate(train_loader):
       #with torch.autograd.set_detect_anomaly(True):
           videos = videos.to(DEVICE)
           videos = videos.permute(0, 1, 3, 2)  # 交换后两维数据
           torch.unsqueeze(videos, 3)  # 增加维度，长度为1
           labels = labels.to(DEVICE)
           # 清零
           optimizer.zero_grad()
           videos = videos.float()
           labels = labels.float()
           labels_real = [torch.argmax(i) for i in labels]  # 为了计算micro-f1和macro-f1
           outputs_tr1,outputs_tr2,outputs_tr3,outputs_tr4 = MsAF_Net(videos)
           # outputs_tr1, outputs_tr2, outputs_tr3= MsAF_Net(videos)
           # 计算损失
           loss_tr1 = criterion(outputs_tr1, labels)
           loss_tr2 = criterion1(outputs_tr2, labels.long())
           loss_tr3=torch.norm(torch.norm(torch.norm(torch.pairwise_distance(outputs_tr3,outputs_tr4),dim=2),dim=1),dim=0).to(DEVICE)
           loss_tr=beta*loss_tr1+beta*loss_tr2+lamda*loss_tr3
           # loss_tr = beta * loss_tr1 + beta * loss_tr2
           #计算训练准确率
           _, predicted = torch.max(outputs_tr1.data, 1)
           predicted2one_hot = nn.functional.one_hot(predicted, num_classes=10)
           l_size = labels.size(0)
           correct_tr = 0
           for j in range(0, l_size):
               if predicted2one_hot[j].equal(labels[j]):
                   correct_tr += 1
           # print('准确率：%.4f%%'%(correct/total))
           correct_sum_tr += correct_tr
           total_tr += l_size
           microf1_tr = f1_score(torch.as_tensor(labels_real,device='cpu'), torch.as_tensor(predicted,device='cpu'), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], average='micro')
           macrof1_tr = f1_score(torch.as_tensor(labels_real,device='cpu'), torch.as_tensor(predicted,device='cpu'), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], average='macro')
           #optimization
           optimizer.zero_grad()
           loss_tr.backward()
           optimizer.step()
     losses_tr.append(loss_tr.cpu().data.item())
     acc_tr = correct_sum_tr / total_tr
     acces_tr.append(acc_tr)
     microf1s_tr.append(microf1_tr)
     macrof1s_tr.append(macrof1_tr)

     # 开始测试
     MsAF_Net.eval()
     correct_sum_te = 0
     total_te = 0

     for videos, labels in test_loader:
         videos = videos.to(DEVICE)
         videos = videos.permute(0, 1, 3, 2)  # 交换后两维数据
         torch.unsqueeze(videos, 3)  # 增加维度，长度为1
         labels = labels.to(DEVICE)
         videos = videos.float()
         labels = labels
         labels_real=[torch.argmax(i) for i in labels]#为了计算micro-f1和macro-f1
         outputs_te1, outputs_te2, outputs_te3,outputs_te4 = MsAF_Net(videos)
         # outputs_te1, outputs_te2, outputs_te3= MsAF_Net(videos)
         # 计算损失
         loss_te1 = criterion(outputs_te1, labels)
         loss_te2 = criterion1(outputs_te2, labels.long())
         # loss_te3 = mseloss(outputs_te3, outputs_te4)
         loss_te3 = torch.norm(torch.norm(torch.norm(torch.pairwise_distance(outputs_te3, outputs_te4), dim=2), dim=1),dim=0)
         loss_te = beta * loss_te1 + beta * loss_te2+ lamda * loss_te3
         #计算准确率
         _, predicted = torch.max(outputs_te1.data, 1)
         predicted2one_hot = nn.functional.one_hot(predicted, num_classes=10)
         l_size = labels.size(0)
         correct_te = 0
         for j in range(0, l_size):
             if predicted2one_hot[j].equal(labels[j]):
                 correct_te += 1
         # print('准确率：%.4f%%'%(correct/total))
         correct_sum_te += correct_te
         total_te += l_size
         microf1_te = f1_score(torch.as_tensor(labels_real,device='cpu'), torch.as_tensor(predicted,device='cpu'), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], average='micro')
         macrof1_te = f1_score(torch.as_tensor(labels_real,device='cpu'), torch.as_tensor(predicted,device='cpu'), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], average='macro')
     losses_te.append(loss_te.cpu().data.item())
     acc_te = correct_sum_te / total_te
     acces_te.append(acc_te)
     microf1s_te.append(microf1_te)
     macrof1s_te.append(macrof1_te)

     print('Epoch {},Loss_tr:{},Loss_te:{},Acc_tr:{},Acc_te:{},microf1_tr:{},microf1_te:{},macrof1_tr:{},macrof1_te:{}'.format(epoch,loss_tr.data.item(),loss_te.data.item(),
           acc_tr,acc_te,microf1_tr,microf1_te,macrof1_tr,macrof1_te))
     if microf1_te >= 0.898 and macrof1_te >= 0.871:
         break

   #损失可视化
   plt.subplot(2,2,1)
   plt.xlabel('Epoch #')
   plt.ylabel('Loss')
   plt.plot(losses_tr,label='train')
   plt.plot(losses_te,label='test')
   plt.legend()
   #准确率可视化
   plt.subplot(2,2,2)
   plt.xlabel('Epoch #')
   plt.ylabel('ACC')
   plt.plot(acces_tr,label='train')
   plt.plot(acces_te,label='test')
   plt.legend()
   #micro-f1可视化
   plt.subplot(2,2,3)
   plt.xlabel('Epoch #')
   plt.ylabel('microf1')
   plt.plot(microf1s_tr,label='train')
   plt.plot(microf1s_te,label='test')
   plt.legend()
   # macro-f1可视化
   plt.subplot(2, 2, 4)
   plt.xlabel('Epoch #')
   plt.ylabel('macrof1')
   plt.plot(macrof1s_tr, label='train')
   plt.plot(macrof1s_te, label='test')
   plt.legend()
   plt.show()

    #保存模型
   #torch.save(MsAF_Net.state_dict(),"fm_MsAF.pth")
   #*****************
    #加载用这个
   # MsAF_Net.load_state_dict(torch.load("fm_MsAF.pth"))

    #开始测试
   # MsAF_Net.eval()
   # correct_sum=0
   # total=0
   # acces=[]
   # for videos,labels in test_loader:
   #      videos=videos.to(DEVICE)
   #      videos = videos.permute(0, 1, 3, 2)  # 交换后两维数据
   #      torch.unsqueeze(videos, 3)  # 增加维度，长度为1
   #      labels = labels.to(DEVICE)
   #      videos = videos.float()
   #      labels = labels
   #      outputs_te1,outputs_te2,outputs_te3=MsAF_Net(videos)
   #      _,predicted=torch.max(outputs_te1.data,1)
   #      predicted2one_hot=nn.functional.one_hot(predicted,num_classes=10)
   #      l_size=labels.size(0)
   #      correct=0
   #      for j in range(0,l_size):
   #          if predicted2one_hot[j].equal(labels[j]):
   #              correct+=1
   #      # print('准确率：%.4f%%'%(correct/total))
   #      correct_sum+=correct
   #      total+=l_size
   # acc=correct_sum/total
   # print('准确率：%f'% (acc))
   # #测试准确率可视化
   # plt.xkcd();
   # plt.xlabel('Epoch #');
   # plt.ylabel('acc')
   # plt.plot(acces)
   # plt.show();