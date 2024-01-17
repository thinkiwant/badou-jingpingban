# 6分类，a出现在 文本长度为6的第几个位置 Pooling 或 RNN解决

import random
import string
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt

vocab = {
    "pad": 0,
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "e": 5,
    "f": 6,
    "g": 7,
    "h": 8,
    "i": 9,
    "j": 10,
    "k": 11,
    "l": 12,
    "m": 13,
    "n": 14,
    "o": 15,
    "p": 16,
    "q": 17,
    "r": 18,
    "s": 19,
    "t": 20,
    "u": 21,
    "v": 22,
    "w": 23,
    "x": 24,
    "y": 25,
    "z": 26,
    "unk": 27
}


# 创建样本
def str_to_sequence(string, vocab):
    return [vocab[s] for s in string]


def generate_string_with_one_a_and_index():
    position = random.randint(0, 5)
    letters = [random.choice(string.ascii_lowercase.replace('a', '')) for _ in range(6)]
    letters[position] = 'a'
    strlen = ''.join(letters)
    return str_to_sequence(strlen, vocab), position, strlen


def build_dataset(total_sample_num):
    pred = []
    target = []
    letters = []
    for i in range(total_sample_num):
        x, y, let = generate_string_with_one_a_and_index()
        pred.append(x)
        target.append(y)
        letters.append(let)

    return torch.FloatTensor(pred), torch.LongTensor(target), letters


# 生成10个这样的字符串及其 'a' 的位置
# strings_with_indices = [generate_string_with_one_a_and_index() for _ in range(10)]

# print(strings_with_indices[0][0], strings_with_indices[1])


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    with torch.no_grad():  # 禁止梯度计算，因为在测试阶段不需要
        inputs, targets, letters = build_dataset(test_sample_num)
        outputs = model(inputs)

    # 计算准确性
    _, predicted = torch.max(outputs, 1)  # 获取模型的预测类别
    print("预测值：", predicted)
    print("样本值：", targets)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    print(f"准确性: {accuracy * 100:.2f}%")
    return accuracy


class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TorchRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 6)  # 输出层
        self.loss = nn.CrossEntropyLoss()
        # self.activation = torch.softmax

    def forward(self, x):
        out, _ = self.rnn(x)
        # out = self.fc(out[:, -1])  # 只使用最后一个时间步的输出
        return out


def main():
    # 配置参数
    hidden_size = 20  # 隐藏维度 10
    epoch_num = 20  # 训练轮数
    batch_size = 200  # 每次训练样本个数
    train_sample = 4000  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 创建训练集，正常任务是读取训练集
    train_x, train_y, letters = build_dataset(train_sample)
    model = TorchRNN(6, hidden_size)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            input_data = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            target = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            # print(input_data)
            # print(target)
            output = model(input_data)
            # print(output)
            loss = model.loss(output, target)  # 计算loss
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        # acc =
        # (model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "ybRNNmodel.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

    return


# 使用训练好的模型做预测
def predict(model_path, input_vec, x, letters):
    print("开始测试")
    model = TorchRNN(6, 20)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        outputs = model(input_vec)

    # 计算准确性
    _, predicted = torch.max(outputs, 1)  # 获取模型的预测类别
    print("预测值：", predicted)
    print("样本值：", x)
    for vec, res, item in zip(input_vec, predicted, letters):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (item, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec, x, letters = build_dataset(5)
    predict("ybRNNmodel.pt", test_vec, x, letters)
