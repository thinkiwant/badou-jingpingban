# coding:utf-8
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json

"""
基于 pytorch的网络编写
实现一个网络完成一个简单的 nlp 任务
判断文本中是否有某些特定字符出现
"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  # embedding层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        self.classify = nn.Linear(vector_dim, sentence_length)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签， 返回 loss 值， 无真实标签， 返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)
        output, h = self.rnn(x)
        rnn_out = h.squeeze()  # 等同rnn_out = output[:, -1, :]
        y_pred = self.classify(rnn_out)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值损失
        else:
            return y_pred


# 为每个字符生成一个标号
# {"a":1,"b":2...}
# ab -> [1,2]
# 构建字符表
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {}
    for idx, char in enumerate(chars):
        vocab[char] = idx
    return vocab


# 随机生成一个样本
# 从所有字中选取 sentence_length个字
# 反之为负样本
def build_sample(vocab, sentence_length):
    # 随机从字表选取 sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length - 1)]
    x.append("a")
    random.shuffle(x)
    # 指定哪些字符出现时为正样本
    y = x.index("a")
    x = [vocab.get(word) for word in x]  # 将字转换为序号， 做 embedding
    return x, y


# 建立数据集
# 输入需要的样本数量
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 200个样本测试
    print(f"本次预测集中共有 {len(y)}个样本")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d，正确率:%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    bath_size = 20  # 每次训练样本数
    train_sample = 1000  # 每轮训练样本总数
    char_dim = 30  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.001  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # print("model.state_dict = ", model.state_dict())
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for bath in range(int(train_sample / bath_size)):
            x, y = build_dataset(bath_size, vocab, sentence_length)
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第 %d 轮平均 loss%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "rnn_model.pth")
    # 保存词表
    writer = open("rnn_vocab.json", "w", encoding="utf-8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 30  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf-8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print(f"输入：{input_string}， 预测类别：{torch.argmax(result[i])}, 概率值：{result[i]}")


if __name__ == '__main__':
    # main()
    test_str = ["adedgs", "ijadle", "lillka"]
    predict("rnn_model.pth", "rnn_vocab.json", test_str)