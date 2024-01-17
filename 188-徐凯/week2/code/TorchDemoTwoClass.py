# 导入库
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于 pytorch 框架编写模型训练
实现一个自行构造的找规律（机器学习）任务
规律：x 是一个 5 维向量，如果第 1 个数 > 第 5 个数，则为正样本，反之为负样本
[5, 2, 4, 3, 1] 1
[1, 2, 3, 4, 5] 0
"""


# 1. 定义 torch 模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        # 创建线性层
        self.linear = nn.Linear(input_size, 1)
        # 创建 sigmoid 激活函数
        self.activation = nn.Sigmoid()
        # 创建 mse 损失函数
        self.loss = nn.MSELoss()

    # 定义前向运算函数
    def forward(self, x, y=None):
        """当输入真实标签，返回loss值;否则，返回预测值"""
        x = self.linear(x)  # (batch_size, input_size) --> (batch_size, 1)
        y_pred = self.activation(x)  # (batch_size, 1) --> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失

        return y_pred


# 2. 定义构建数据样本函数
# 生成一个样本，样本的生成方法，代表了我们要学习的规律
# 随机生成一个 5 维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    """
    构建数据样本
    :return: x, y
    """
    x = np.random.random(5)
    if x[0] > x[-1]:
        return x, 1
    else:
        return x, 0


# 3. 随机生成一批样本
# 正负样本随机生成
def build_dataset(total_sample_num):
    """
    随机生成一批样本
    :param total_sample_num: 样本数量
    :return: tensor类型的 X, Y
    """
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])

    X = np.array(X)
    Y = np.array(Y)

    return torch.Tensor(X), torch.Tensor(Y)


# 4. 定义验证评估函数
def evalute(model, val_dataset):
    """
    验证评估模型
    :param model: 模型
    :param val_dataset: 验证集
    :return: 准确率
    """
    # 选择验证模式
    model.eval()
    x, y = val_dataset
    print(f"本次预测集中共有{int(np.sum(y.numpy()))}个正样本，{int(len(val_dataset[0]) - np.sum(y.numpy()))}个负样本")
    correct, wrong = 0, 0

    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1

    acc = correct / (correct + wrong)
    print(f"正确预测个数：{correct}，正确率：{acc}")
    return acc


# 5. 使用训练好的模型进行预测
def predict(model_path, input_vec, input_size):
    """
    利用模型训练好的参数进行预测
    :param input_size: 输入向量维度
    :param model_path: 模型路径
    :param input_vec: 待预测向量
    :return: None
    """
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 验证模型
    with torch.no_grad():  # 不计算梯度
        result = model(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print(f"输入：{vec}，预测类别：{round(float(res))}，概率值：{res}")  # 打印结果


# 6. main 函数
def main():
    # 定义参数
    epochs = 50  # 训练论述
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 训练集样本数
    val_sample = 100  # 验证集样本数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率

    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    # 创建验证集（保证每次的验证集相同）
    val_dataset = build_dataset(val_sample)

    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 存放训练结果
    log = []

    # 训练过程
    for epoch in range(epochs):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            # 计算损失值
            loss = model(x, y)
            # 计算梯度
            loss.backward()
            # 更新权重
            optim.step()
            # 梯度归零
            optim.zero_grad()
            watch_loss.append(loss.item())

        print(f"============\n第{epoch + 1}轮平均loss：{np.mean(watch_loss)}")
        # 测试本轮模型结果
        acc = evalute(model, val_dataset)
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "../model/model_two_class.pt")
    # 画图
    # print(log)
    plt.plot(range(len(log)), [l[0] for l in log])  # 画 acc 曲线
    plt.plot(range(len(log)), [l[1] for l in log])  # 画 loss 曲线
    plt.legend(['acc', 'loss'])
    plt.title("train loss & acc")
    plt.xlabel("epoch")
    plt.show()

    return


if __name__ == "__main__":
    main()
    print("=" * 20, "模型预测", "=" * 20)
    test_vec = [[0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.19349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894]]
    predict("../model/model_two_class.pt", test_vec, 5)
