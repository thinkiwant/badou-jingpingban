# coding:utf8

'''
任务：5分类任务
数据生成规则：
1. 随机生成一个5维向量，每个维度的值在0到1之间。
2. 计算向量中所有值的和。
3. 根据和的大小将样本分为5个类别：
如果和小于1，则为类别0；
如果和在1到2之间，则为类别1；
如果和在2到3之间，则为类别2；
如果和在3到4之间，则为类别3；
如果和大于等于4，则为类别4；
'''

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 输出层改为num_classes个输出
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y)  # 如果有真实标签则返回损失
        else:
            return torch.softmax(x, dim=1)  # 否则返回softmax概率分布


def build_sample():
    x = np.random.random(5)
    y_sum = np.sum(x)
    if y_sum < 1:
        y = 0
    elif y_sum < 2:
        y = 1
    elif y_sum < 3:
        y = 2
    elif y_sum < 4:
        y = 3
    else:
        y = 4
    return x, y


def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 标签是长整型

# 评估函数
def evaluate(model, test_sample_num=100):
    model.eval()
    x, y = build_dataset(test_sample_num)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        predicted = torch.argmax(y_pred, 1)
        correct += (predicted == y).sum().item()
    acc = correct / test_sample_num
    print("正确预测个数：%d, 正确率：%f" % (correct, acc))
    return acc

# 主函数，训练和评估模型
def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    num_classes = 5
    learning_rate = 0.001

    model = TorchModel(input_size, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), "model.pt")
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))
    predicted = torch.argmax(result, 1)
    for vec, res in zip(input_vec, predicted):
        print("输入：%s, 预测类别：%d" % (vec, res))

if __name__ == "__main__":
    main()
    # 测试用例
    test_vec = [
        np.random.random(5),
        np.random.random(5),
        np.random.random(5),
        np.random.random(5)
    ]
    predict("model.pt", test_vec)
