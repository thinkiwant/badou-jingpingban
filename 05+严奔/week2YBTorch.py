import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from torch import optim

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本

"""


# 1. 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(5, 5)  # 5维输入，5个类别的输出
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.linear(x)


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 3)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.CrossEntropyLoss()  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        # y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            return x  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    # 生成一个包含 0 到 10 之间不同数值的1*5数组
    unique_values = np.random.choice(11, 5, replace=False)
    # 查找最大值的坐标
    max_coords = find_max_value_index(unique_values)
    return unique_values, max_coords


def find_max_value_index(array):
    if array.shape != (5,):
        return None  # 返回 None 如果输入的数组大小不符合要求
    # 找到数组中的最大值及其坐标
    max_value = np.max(array)
    max_coords = np.argwhere(array == max_value)[0][0]
    return max_coords


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    pred = []
    target = []
    for i in range(total_sample_num):
        x, y = build_sample()
        pred.append(x)
        target.append(y)
        # 将 target 转换为 NumPy 数组
        # target = np.array(target)

    return torch.FloatTensor(pred), torch.LongTensor(target)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    with torch.no_grad():  # 禁止梯度计算，因为在测试阶段不需要
        inputs, targets = build_dataset(test_sample_num)
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


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 1  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率

    model = MyModel()

    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            input_data = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            target = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            output = model(input_data)
            loss = model.ce_loss(output, target)  # 计算loss
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
    torch.save(model.state_dict(), "ybmodel.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = MyModel()
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        outputs = model(input_vec)

        # 计算准确性
    _, predicted = torch.max(outputs, 1)  # 获取模型的预测类别
    print("预测值：", predicted)
    print("样本值：", predicted)
    for vec, res in zip(input_vec, predicted):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    # main()
    test_vec,x = build_dataset(5)
    predict("ybmodel.pt", test_vec)
