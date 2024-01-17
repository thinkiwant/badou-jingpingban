#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中a出现在文本中的位置，第一位为0，以此类推

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层   28*6
        
        #我们需要对整句话作池化，所以这里传递进字表的长度
        self.pool = nn.AvgPool1d(sentence_length)   #池化层     
        self.classify = nn.Linear(vector_dim, 6)     #线性层
        # self.activation = torch.sigmoid     #sigmoid归一化函数
        self.loss = nn.functional.cross_entropy  #loss函数采用均方差损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # print("++++++++++++++++++++",x)
        x = x.transpose(1, 2)                      #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        x = self.pool(x)                           #(batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        x = x.squeeze()                            #(batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        y_pred = self.classify(x) 
        # print("++++++++++++++++++++++",y_pred,y)                      #(batch_size, vector_dim) -> (batch_size, 1) 3*5 5*1 -> 3*1
        # y_pred = self.activation(x)                #(batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果
        

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号

    # 通常我们会在词表中预留一个位置放unk，我们不能保证我们的词表可以覆盖未来我们要预测的样本中的所有字符的。
    # 假设未来我们要预测的字符串里出现了一个字符，在ebedding层中没有对应的位置，则会报错，因为它找不到对应的向量来代表这个词
    # 两种做法：1.不在词表中的字符直接删除（会丢失信息，不推荐），2.设置一个未知的token，所有没见过的字符都统一取这个位置token对应的向量（推荐）
    
    vocab['unk'] = len(vocab) #26
    return vocab


#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list("bcdefghijklmnopqrstuvwxyz")) for i in range(sentence_length-1)]
    x.append('a')
    random.shuffle(x)
    y = x.index('a')
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x,y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    # print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1   #负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1   #正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)





def main():
    #配置参数
    epoch_num = 200        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 6         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.01 #学习率
    # 建立字表
    vocab = build_vocab()    #28个
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)   #   
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   

    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample, vocab, sentence_length)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):    
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",x,y)
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    #     acc = evaluate(model, vocab, train_sample)  # 测试本轮模型结果
    #     log.append([acc, float(np.mean(watch_loss))])
    #     print(log)
    # # 保存模型
    # torch.save(model.state_dict(), "model.pt")
    # # 画图
    # print(log)
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    # plt.legend()
    # plt.show()
    return


if __name__ == "__main__":
    main()