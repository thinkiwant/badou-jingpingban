#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class RNNModel(nn.Module):
    def __init__(self, vector_dim, hidden_size, vocab):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        self.layer1 = nn.RNN(vector_dim, hidden_size, bias=False, batch_first=True, nonlinearity='tanh')
        self.layer2 = nn.Linear(hidden_size, 1)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss

    def forward(self, x, y=None): # x.shape --> (batch_size, sentence_len)
        x = self.embedding(x) # 输入shape (batch_size, sentence_len) --> (batch_size, sentence_len, vector_dim)
        x = self.layer1(x)[1][0] # 输出 ([h0,h1,h2,h3...,ht],ht)元组，取最后时间步的结果即 output[1] -->（1, batch_size, hidden_size)
        x = self.layer2(x) # (1, batch_size, hidden_size) --> (1, batch_size, 1)
        # x = x.squeeze(0) # (1, batch_size, 1) --> (batch_size, 1)
        y_pred = self.activation(x) # (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #指定哪些字出现时为正样本
    if set("abc") & set(x):
        y = 1
    #指定字都未出现，则为负样本
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        tmp = zip(y_pred, y)
        for y_p, y_t in tmp:  #与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1   #负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1   #正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    epoch_num = 200
    batch_size = 64
    train_sample = 640
    char_dim = 20
    hidden_size = 25
    sentence_len = 6
    lr = 0.005

    vocab = build_vocab()
    model = RNNModel(char_dim, hidden_size, vocab)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_len)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_len)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    hidden_size = 25
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = RNNModel(char_dim, hidden_size, vocab)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i])) #打印结果



if __name__ == "__main__":
    # main()
    test_strings = ["fnvfee", "wzsdfg", "rqwdeg", "nakwww"]
    predict("model.pth", "vocab.json", test_strings)