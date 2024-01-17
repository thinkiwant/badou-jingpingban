# coding:utf8
import string

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

"""


class ClassifyModel(nn.Module):
    def __init__(self, dict_size, vector_dim):
        super(ClassifyModel, self).__init__()
        # 第一个参数为生成的词向量表的batch_size,第二个参数为每个词向量的维度
        self.embedding = nn.Embedding(dict_size, vector_dim)
        self.classify = nn.Linear(vector_dim, 1)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.functional.mse_loss  # loss函数采用均方差损失

    def forward(self, x, y=None):
        new_x = []
        for v in x:
            v = self.embedding(torch.LongTensor(v))
            # print(f'转成的词向量后，x的形状为:{v.shape}')
            # 根据句子的的长短使用不同的kenel_size进行动态池化
            # len(v)计算要放在transpose前边，以准确的求得句子的长短
            pool = nn.AvgPool1d(len(v))
            v = v.transpose(1, 0)
            v = pool(v).squeeze()
            new_x.append(v.tolist())

        # print(f'池化后的x为：{new_x}')
        y_pred = self.classify(torch.FloatTensor(new_x))
        y_pred = self.activation(y_pred)

        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


class DynamicRNNModel(nn.Module):
    def __init__(self, dict_size, vector_dim, output_dim):
        super(DynamicRNNModel, self).__init__()
        self.embedding = nn.Embedding(dict_size, vector_dim)
        self.rnn = nn.RNN(vector_dim, vector_dim * 2, bias=False, batch_first=False)  # 使用RNN层
        self.classify = nn.Linear(vector_dim * 2, output_dim)  # 线性层
        self.loss = nn.functional.cross_entropy  # loss函数采用均方差损失

    def forward(self, x, y=None):
        Y_pred = []
        # TODO
        # 这种操作会导致backward 报错：RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        # 具体原因不明确
        for v in x:
            v = self.embedding(torch.LongTensor(v))
            _, y_pred = self.rnn(v)
            y_pred = self.classify(y_pred).squeeze()
            Y_pred.append(y_pred.tolist())

        Y_pred = torch.FloatTensor(Y_pred)
        print(f'Y_pred shape is {Y_pred.shape}')
        print(f'y shape is {y.shape}')
        if y is not None:
            return self.loss(Y_pred, y)
        else:
            return y_pred


class StaticRNNModel(nn.Module):
    def __init__(self, dict_size, vector_dim, output_dim):
        super(StaticRNNModel, self).__init__()
        self.embedding = nn.Embedding(dict_size, vector_dim)
        self.rnn = nn.RNN(vector_dim, vector_dim * 2, bias=False, batch_first=True)  # 使用RNN层
        self.classify = nn.Linear(vector_dim * 2, output_dim)  # 线性层
        self.loss = nn.functional.cross_entropy  # loss函数采用均方差损失

    def forward(self, x, y=None):
        x = self.embedding(x)
        _, y_pred = self.rnn(x)
        y_pred = self.classify(y_pred).squeeze()
        # y_pred = self.activation(y_pred)

        # print(f'Y_pred shape is {y_pred.shape}')
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# class NlpModel(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super().__init__()
#         self.embedding = nn.Embedding(input_size, hidden_size)
#         # 一般认为hidden_size包含的信息要比单个input_size包含的信息多，所以一般hidden_size比input_size要大
#         # batch_first=True，用于声明模型第一位是batch_size,默认第一维不是batch_size
#         self.fc1 = nn.RNN(input_size, hidden_size, bias=False, batch_first=True)


class CharGen():
    def __init__(self):
        # 小写字母
        self.chars = string.ascii_letters
        self.char_dict = self.__generate_char_dict(self.chars)

    def __generate_char_dict(self, chars, start_index=0):
        char_dict = {}
        for i, value in enumerate(chars):
            char_dict[value] = i + start_index
        return char_dict

    # 生成一个含指定字符串的样本,account为样本大小，char为包含的字符，char_len为字符串的长度
    def generate_char_list(self, account, char, sentence_length=None):
        char_lib = self.chars.replace(char, "")
        char_lib_len = len(char_lib)
        X_char = []  # 字符串形式的样本
        X = []  # 字符表索引形式的样本
        Y = []  # 样本的标注结果
        input_size = sentence_length
        for _ in range(account):
            if sentence_length is None:
                # 设置句子的随机长度，需要大于1，小于等于字表长度
                input_size = random.randint(1, char_lib_len - 1)
            x = []  # 索引形式的单一样本
            # 方式如下：
            # 1、从字符串库截取一个字符串，当做基础字符串，基础字符串的长度等于char_len
            # 2、随机选取一个位置，将基础字符串的值替换为指定字符串

            # 计算字符串库截取的随机初始位置，初始位置不能过大，防止截取的总长度超过字符串库的最大长度
            start_index = random.randint(0, char_lib_len - 1 - input_size)
            # 根据起始位置加上字符串长度，生成指定的字符串
            chars = char_lib[start_index:start_index + input_size]
            # 设置一个随机的长度，用于将字符串的指定位置替换成指定的字符
            char_index = random.randint(0, input_size - 1)
            Y.append(char_index)
            # 将字符串指定的位置替换成目标字符
            new_string = chars[:char_index] + char + chars[char_index + 1:]
            X_char.append(new_string)

            for v in new_string:
                x.append(self.char_dict[v])
            X.append(x)
        return X_char, X, Y
        # return torch.FloatTensor(X),torch.FloatTensor(Y)

    # 根据字符串及词表生成词表序列
    def transform_to_index(self, chars):
        char_index = []
        for x in chars:
            char_index.append(self.char_dict[x])
        return char_index


def train_line_model():
    char_gen = CharGen()
    dict_size = len(CharGen().char_dict)  # 初始化字典表的长度
    vector_dim = 6  # 初始化词向量的长度
    train_sample = 500  # 总样本数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数

    learning_rate = 0.005  # 学习率

    X_char, X, Y = char_gen.generate_char_list(train_sample, 'b')
    # print(f'生成的chars列表为: {X_char}')
    # print(f'对应的字典表索引为：{X}')
    # print(f'样本标注结果为：{Y}')

    model = ClassifyModel(dict_size, vector_dim)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for i in range(train_sample // batch_size):
            X_batch = X[i * batch_size:(i + 1) * batch_size]
            Y_batch = Y[i * batch_size:(i + 1) * batch_size]
            optim.zero_grad()  # 梯度归零

            # tensor要求各个维度的大小都是相同的，而生成的句子长度是随机的，所以传入模型时，先不以tensor的形式传入，等形状统一后，在转为tensor
            loss = model(X_batch, torch.FloatTensor(Y_batch))  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print(f"模型类型为:{type(model)}，第{epoch + 1}轮平均loss:{np.mean(watch_loss)}")
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    # writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


def train_dynamic_sentence_length_rnn_model():
    print('动态模型训练=====================此模型存在问题，详见TODO')
    char_gen = CharGen()
    dict_size = len(CharGen().char_dict)  # 初始化字典表的长度
    flag = 'b'
    vector_dim = 6  # 初始化词向量的长度
    train_sample = 1000  # 总样本数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数

    learning_rate = 0.005  # 学习率

    X_char, X, Y = char_gen.generate_char_list(train_sample, 'b')
    # print(f'生成的chars列表为: {X_char}')
    # print(f'对应的字典表索引为：{X}')
    # print(f'样本标注结果为：{Y}')

    model = DynamicRNNModel(dict_size, vector_dim, dict_size)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for i in range(train_sample // batch_size):
            X_batch = X[i * batch_size:(i + 1) * batch_size]
            Y_batch = Y[i * batch_size:(i + 1) * batch_size]
            optim.zero_grad()  # 梯度归零

            # tensor要求各个维度的大小都是相同的，而生成的句子长度是随机的，所以传入模型时，先不以tensor的形式传入，等形状统一后，在转为tensor
            loss = model(X_batch, torch.LongTensor(Y_batch))  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print(f"模型类型为:{type(model)}，第{epoch + 1}轮平均loss:{np.mean(watch_loss)}")
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    # writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 训练固定长度语句的模型
def train_static_sentence_length_rnn_model():
    char_gen = CharGen()
    dict_size = len(CharGen().char_dict)  # 初始化字典表的长度
    flag = 'b'
    vector_dim = 6  # 初始化词向量的长度
    train_sample = 1000  # 总样本数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    num_class = 10  # 类别数
    sentence_length = num_class - 1  # 保证flag出现的位置在num_class的范围内

    learning_rate = 0.005  # 学习率

    X_char, X, Y = char_gen.generate_char_list(train_sample, flag, sentence_length)
    # print(f'生成的chars列表为: {X_char}')
    # print(f'对应的字典表索引为：{X}')
    # print(f'样本标注结果为：{Y}')

    model = StaticRNNModel(dict_size, vector_dim, num_class)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for i in range(train_sample // batch_size):
            X_batch = X[i * batch_size:(i + 1) * batch_size]
            Y_batch = Y[i * batch_size:(i + 1) * batch_size]
            optim.zero_grad()  # 梯度归零

            # tensor要求各个维度的大小都是相同的，而生成的句子长度是随机的，所以传入模型时，先不以tensor的形式传入，等形状统一后，在转为tensor
            loss = model(torch.LongTensor(X_batch), torch.LongTensor(Y_batch))  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print(f"模型类型为:{type(model)}，第{epoch + 1}轮平均loss:{np.mean(watch_loss)}")
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    # writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()

    # 创建测试集数据
    X_char, X, Y = char_gen.generate_char_list(10, flag, sentence_length)

    correct = 0;
    for char, dict, index in zip(X_char, torch.LongTensor(X), Y):
        y_pred = torch.argmax(model(dict))
        print(f'字符串{char},模型预测字符串{flag},出现的位置为：{y_pred},实际出现的位置为：{index}')
        if y_pred == index:
            correct += 1
    print(f'模型正确率为{correct / len(Y):.2%}')


def main():
    # 训练线性模型
    # train_line_model()

    # 训练预测静态长度文本中字符出现位置的RNN模型
    train_static_sentence_length_rnn_model()

    # 训练预测动态长度文本中字符出现位置的RNN模型
    train_dynamic_sentence_length_rnn_model()


if __name__ == "__main__":
    main()
