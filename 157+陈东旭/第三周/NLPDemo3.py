'''
简单NLP多分类问题
生成长度为6的字符串, 字符a第一次出现在字符串位置即是类型: index 

'''

import random
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from torch.autograd import Variable
import json

class TorchModel(nn.Module):
    '''
    长度6的字符串中a出现的位置, 6分类问题
    '''
    def __init__(self, vector_dim, vocab, sentence_length, num_classes):
        super().__init__()
        #embedding向量化
        self.embedding = nn.Embedding(len(vocab), vector_dim)  # (batch_size, sentenc_length) -> (batch_size, sentenc_length, vector_dim)
        # # 池化层
        # self.pool = nn.AvgPool1d(sentence_length)
        self.hidden_dim = 10
        # 线性层
        # self.classify = nn.Linear(vector_dim, num_classes)
        self.rnn = nn.RNN(vector_dim, self.hidden_dim, 1, bias=False, batch_first=True)
        self.hidden2label = nn.Linear(self.hidden_dim, num_classes)
        #self.activation = nn.Softmax
        # 损失函数
        self.loss = nn.CrossEntropyLoss()  # 包含softmax





    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sentenc_length) -> (batch_size, sentenc_length, vector_dim) 20, 6, 20
        # h0 = Variable(torch.zeros(1, x.size(0), self.hidden_dim))
        output, h = self.rnn(x)
        h = h.squeeze()
        y_pred = self.hidden2label(h)

        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for i in range(len(chars)):
        vocab[chars[i]] = i + 1

    vocab['unk'] = len(vocab)
    return vocab

def build_sample(vocab, sentence_length):
    '''
    生成长度为sentence_length的字符串, 输出字符a第一次出现的位置: 索引
    :param vocab: 词表
    :param sentence_length: 字符串长度
    :return:
    '''
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    y = -1
    for index, word in enumerate(x):
        if word == 'a':
            y = index
            break
    if y == -1:
        y = random.randint(0, 5)
        x[y] = 'a'
    x = [vocab.get(word, vocab['unk']) for word in x]
    # print(x, y)
    return x, y

def build_datasets(vocab, sentence_length, sample_size):
    '''

    :param vocab:
    :param sentence_length:
    :param sample_size:
    :return:
    '''
    X = []
    Y = []
    for i in range(sample_size):
        x, y = build_sample(vocab, sentence_length)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)

#测试每轮的准确率
def evaluate(model: TorchModel, vocab, sentence_length):
    model.eval()
    test_X, test_Y = build_datasets(vocab, sentence_length, 100)
    y_sum = [0] * 6
    for y in test_Y:
        y_sum[y] = y_sum[y] + 1

    with torch.no_grad():
        y_pred = model(test_X)
        y_pred = torch.max(y_pred, dim=1)[1]
        acc = accuracy_score(y_true=test_Y, y_pred=y_pred)
        # p = precision_score(y_true=test_Y, y_pred=y_pred)
        # f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        # recall = recall_score(y_true=test_Y, y_pred=y_pred)

        # print("本次测试共生成%d个样本, 每个分类的样本数量为%s,  模型准确率: %f, "
        #       "模型精确率: %f, 模型f1: %f, 模型召回率: %f, "%(100, str(y_sum), acc, 0, 0, 0))
        print("本次测试共生成%d个样本, 每个分类的样本数量为%s,  模型准确率: %f " % (100, str(y_sum), acc))
        return acc





def main():
    vocab = build_vocab()  # 创建词表
    sentence_length = 6    # 确定字符串长度
    num_classes = 6
    train_sample_size = 5000  # 每次样本数量
    vector_dim = 20        # 字符embeeding维度
    epoch_num = 20     # 训练轮数
    batch_size = 20   # batch样本数
    logs = []  # 记录每轮的loss和准确率
    train_X, train_Y = build_datasets(vocab, sentence_length, train_sample_size)
    # 创建模型结构
    model = TorchModel(vector_dim, vocab, sentence_length, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample_size // batch_size):
            x = train_X[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_Y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())

        print("第%d轮的loss: %f"% (epoch, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        logs.append([np.mean(watch_loss), acc])

    print(logs)
    plt.plot(range(len(logs)), [log[1] for log in logs], label="acc")
    # plt.plot(range(len(logs)), [log[2] for log in logs], label="p")
    # plt.plot(range(len(logs)), [log[3] for log in logs], label="f1")
    # plt.plot(range(len(logs)), [log[4] for log in logs], label="recall")
    plt.plot(range(len(logs)), [log[0] for log in logs], label="loss")
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), 'rnn_model.pt')
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

def predic(model_path, vocab_path, sentence_length, input_strings):
    # 加载词表
    vocab = json.load(open(vocab_path, 'r', encoding='utf-8'))
    # 将字符串映射成数字
    X = []
    for input_string in input_strings:
        X.append([vocab[word] for word in input_string])

    vector_dim = 20
    num_classes = 6
    # 创建模型结构
    model = TorchModel(vector_dim, vocab, sentence_length, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        results = model(torch.LongTensor(X))
        for index, input_string in enumerate(input_strings):

            y, y_pred = torch.max(results, dim=1)
            # print(y, y_pred)
            print("输入：%s, 预测类别：%d" % (input_string, y_pred[index])) #打印结果






if __name__ == '__main__':
    # vocab = build_vocab()
    # sentence_length = 6
    # num_classes = 7
    # X, Y = build_datasets(vocab, sentence_length, 10)
    # print(X, Y)
    main()
    #input_strings = ["abcdef", "bcdefa", "bacdef", "bcadef"]
    #predic("rnn_model.pt", "vocab.json", 6, input_strings)



