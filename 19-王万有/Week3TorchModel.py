# coding:utf8
import random
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
输入一个字符串，根据字符r所在位置进行分类
对比rnn和pooling做法'

"""
# class Torchmodel(nn.Module):
#     def __init__(self,vocab,vector_dim,sentence_length):
#         super(Torchmodel, self).__init__()
#         self.embedding = nn.Embedding(len(vocab), vector_dim)
#         self.pooling = nn.AvgPool1d(sentence_length)
#
#         self.classify = nn.Linear(vector_dim, sentence_length + 1)
#         self.loss = nn.functional.cross_entropy
#
#
#     def forward(self,x,y=None):
#         x = self.embedding(x)
#         x = x.transpose(1, 2)
#         x = self.pooling(x)
#         # squeeze方法去掉值为1的维度
#         x = x.squeeze()
#         y_pred = self.classify(x)
#
#         if y is not None:
#             return  self.loss(y_pred,y)
#         else:
#             return y_pred

class Torchmodel(nn.Module):
    def __init__(self,vocab,vector_dim,sentence_length):
        super(Torchmodel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # 添加RNN层
        self.classify = nn.Linear(vector_dim, sentence_length + 1)
        self.loss = nn.functional.cross_entropy

    def forward(self,x,y=None):
        x = self.embedding(x)
        x, _ = self.rnn(x)  # 使用RNN层处理输入(batch_size, sequence_length, vector_dim)
        x = x[:, -1, :]  # 取最后一个时间步的输出
        y_pred = self.classify(x)

        if y is not None:
            return  self.loss(y_pred,y)
        else:
            return y_pred


    #建立字典
def build_vocab():
    chars = 'abcdefghijklmnopqrstuvwxyz'
    vocab ={"pad":0}
    for index,char in enumerate(chars):
        vocab[char] = index+1
    vocab['unk'] = len(vocab)
    return vocab
vocab = build_vocab()
a = ['i', 'g', 'r', 'unk']
x = [vocab.get(word, vocab['unk']) for word in a]


#print([random.choice(list(vocab.keys()))for _ in range(3)])
# print(a)
# print(x)

#建立数组
def build_sample(vocab,sentence_length):

    x = random.sample(list(vocab.keys()),sentence_length)

    if 'r' in x:
        y = x.index('r')
    else:
        y = sentence_length

    x = [vocab.get(_,vocab['unk']) for _ in x ]

    return x,y

#sample  =  build_sample(vocab,4)
#print(sample)

def build_sample_dataset(num,vocab,sentence_length):
    x_data = []
    y_data = []
    for i in range(num):
        x,y =build_sample(vocab,sentence_length)
        x_data.append(x)
        y_data.append(y)

    return torch.LongTensor(x_data),torch.LongTensor(y_data)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = Torchmodel(vocab, char_dim, sentence_length)
    return model

def model_calcute(model,vocab,sentence_length):
    model.eval()
    x,y = build_sample_dataset(200,vocab,sentence_length)
    print("本次预测集中共有%d个样本" % (len(y)))
    correct,wrong =0,0
    with torch.no_grad():
        y_pred = model(x)
        for y_p,y_t in zip(y_pred,y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)



def main():
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 40       #每次训练样本个数
    train_sample = 1000    #每轮训练总共训练的样本总数
    char_dim = 50         #每个字的维度
    sentence_length = 10   #样本文本长度
    learning_rate = 0.001 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample/batch_size)):
            x,y = build_sample_dataset(batch_size,vocab,sentence_length)
            optim.zero_grad()
            loss= model(x,y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = model_calcute(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

        # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "mymodel1.pth")
    # 保存词表
    writer = open("myvocab1.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return
#使用训练好的模型做预测

def predict(model_path, vocab_path, input_strings):
    char_dim = 50  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    original_positions = []  # 存储原始位置的列表
    for input_string in input_strings:
        #输入字符串截断或填充，以确保其长度不超过 sentence_length
        input_string = (input_string + ' ' * sentence_length)[:sentence_length]
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  #将输入序列化
        if 'r' in input_string:
            original_positions.append(input_string.index('r'))  # 存储 'r' 的原始位置
        else:
            original_positions.append(sentence_length)  # 如果 'r' 不在字符串中，位置设为 sentence_length
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 原本类别：%s" % (input_string, int(torch.argmax(result[i])), original_positions[i])) #打印结果
if __name__ == "__main__":
    main()
    test_strings = ["wwybjr", "bjr", "abc", "789"]
    predict("mymodel.pth", "myvocab.json", test_strings)
