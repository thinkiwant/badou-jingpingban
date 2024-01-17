# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，向量中哪个标量最大就输出哪一维下标

"""


class TorchModel(nn.Module):
    def __init__(self, x_dim):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(x_dim, 5)  # 线性层
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, x_dim) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 模型
def build_model(x_dim):
    return TorchModel(x_dim)


# 优化器
def build_optimizer(model):
    learn_rate = 0.001  # 学习率
    return torch.optim.Adam(model.parameters(), lr=learn_rate)


# 训练
def training(model, model_path, optim):
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数

    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)

    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), model_path)
    return log


# 测试：测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 样本：随机生成一批样本，正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，根据每个向量中最大的标量同一下标构建Y
def build_sample():
    x = np.random.random(5)
    # 获取最大值的索引
    max_index = np.argmax(x)
    if max_index == 0:
        return x, 0
    elif max_index == 1:
        return x, 1
    elif max_index == 2:
        return x, 2
    elif max_index == 3:
        return x, 3
    else:
        return x, 4


# 画图
def plot(log):
    # print("log:", log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()


# 预测：使用训练好的模型做预测
def predict(x_dim, model_path, input_vec):
    model = build_model(x_dim)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print("model:", model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))  # 打印结果


def main():
    # 配置
    model_path = "model.pth"
    x_dim = 5  # 输入向量维度

    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.99871392],
                [0.1349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894]]

    # 模型
    model = build_model(x_dim)

    # 优化器
    optim = build_optimizer(model)

    # 训练
    log = training(model, model_path, optim)

    # 画图
    plot(log)

    # 预测
    predict(x_dim, model_path, test_vec)
    return


if __name__ == "__main__":
    main()
