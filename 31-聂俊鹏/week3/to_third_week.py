#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

"""

class TorModel(nn.Module):
    def __init__(self,vocab , char_dim , char_len):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab) , char_dim)
        self.pool = nn.AvgPool1d(char_len)
        self.linear = nn.Linear(char_dim, 7)  # 线性层
        self.classify = nn.RNN(char_dim,10 , bias= False , batch_first= True)
        self.loss = nn.CrossEntropyLoss()

    def forward(self , x , y = None):
        x1 = self.embedding(x)          # 20 * 6 * 10
        # print("经过embedding后：-->", x1.shape)
        # x2 = x1.transpose(1,2)
        # # print("经过转置后两位后：-->", x2.shape)
        # x3 = self.pool(x2)
        # # print("经过平均池化后：-->", x3.shape)
        # x4 = x3.squeeze()
        # # print("经过去尾后：-->", x4.shape)
        # y_pred = self.linear(x4)
        # # print("经过去尾后：-->", y_pred.shape)
        _, x2 = self.classify(x1)
        y_pred = x2.squeeze()
        if y is not None:
            return self.loss(y_pred , y)
        else:
            return y_pred



def build_vocab():
    vocab = json.load(open( "vocab.json", "r", encoding="utf8"))
    return vocab

def to_index(x):
    for index , char in enumerate(x):
        if char == 'f':
            return index
    return 6

def build_sample(vocab , char_len):
    x = [random.choice(list(vocab.keys())) for _ in range(char_len)]

    y = to_index(x )
    # print(x,"<---查看y-->",y)
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x , y

def build_dataset(vocab , batch_size ,char_len):
    dataset_x = []
    dataset_y = []
    for i in range(batch_size):
        x, y = build_sample(vocab, char_len)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def evaluate(model , vocab , char_len):
    model.eval()
    x , y = build_dataset(vocab , 200 , char_len)

    print("本次预测集中共有%d个下标为0，%d个下标为1，%d个下标为2，%d个下标为3，%d个下标为4，%d个下标为5，%d个字符串中没有f" % (sum(y == 0), sum(y == 1),sum(y == 2),sum(y == 3),sum(y == 4),sum(y == 5),sum(y == 6)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p ,y_t in zip(y_pred , y):
            # print(torch.argmax(y_p), "<==========>", y_p)
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d / %d, 正确率：%f" % (correct, 200, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    count_sum = 100      #训练的轮数
    batch_size = 10     #每轮训练的个数
    numberSum = 1000    #总样本数
    char_dim = 10       #样本的维度
    char_len = 6        #样本的长度
    linear_rate = 0.001 #学习率
    #创建字表
    vocab = build_vocab()
    print(vocab)
    #创建模型
    model = TorModel(vocab , char_dim , char_len)

    #优化器
    optim = torch.optim.Adam(model.parameters(),lr= linear_rate)
    log = []

    for i in range(count_sum):
        model.train()
        watch_log = []
        for j in range(numberSum // batch_size):
            x ,y = build_dataset(vocab , batch_size , char_len)
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_log.append(loss.item())
        print("=========================\n第%d轮,平均loss：%f"%(i +1 , np.mean(watch_log)))
        acc = evaluate(model , vocab , char_len)
        log.append([acc , np.mean(watch_log)])
    plt.plot(range(len(log)),[ l[0] for l in log] , label='acc')
    plt.plot(range(len(log)),[ l[1] for l in log] , label='loss')
    plt.show()

    #保存模型
    torch.save(model.state_dict(),"model.pth")

    return




if __name__ == "__main__":
    main()


