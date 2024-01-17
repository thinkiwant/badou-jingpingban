import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import json


# 建立词典，diction的每一项为{w:id}
def build_diction():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    diction = {"pad": 0}
    for index, char in enumerate(chars):
        diction[char] = index + 1  # 每个字对应一个序号
    diction["unk"] = len(diction)  # 26
    return diction


# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 反之为负样本
def build_sample(diction, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    text = [random.choice(list(diction.keys())) for _ in range(sentence_length)]
    # 指定哪些字出现时为正样本
    if set("abc") & set(text):
        y = 1
    # 指定字都未出现，则为负样本
    else:
        y = 0
    x = [diction.get(word, diction["unk"]) for word in text]  # 将字符转换成其在字典中的索引
    # 创建句子的向量化表示，向量的尺寸为词典中词汇的个数，i位置上面的数值为第i个单词在sentence中出现的频率
    vector = np.zeros(len(diction))
    for l in x:
        vector[l] += 1
    return 1.0 * vector / len(x), y, text


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, diction, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y, _ = build_sample(diction, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.FloatTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立神经网络
# 模型定义
# 一个简单的前馈神经网络，第一层线性层，加一个非线性ReLU，第二层线性层，中间有10个隐含层神经元


# 输入维度为词典的大小
class TorchModel(nn.Module):
    def __init__(self, diction):
        super().__init__()
        self.linear1 = nn.Linear(len(diction), 10)
        self.activate1 = nn.ReLU()
        self.linear2 = nn.Linear(10, 2)
        self.activate2 = nn.LogSoftmax(dim=1)
        # 损失函数为交叉熵
        self.cost = torch.nn.NLLLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.activate1(self.linear1(x))
        y_pred = self.activate2(self.linear2(x))
        if y is not None:
            return self.cost(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred


def rightness(predictions, labels):
    """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，
    batch_size行num_classes列的矩阵，labels是数据之中的正确答案"""
    pred = torch.max(predictions.data, 1)[1]  # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    rights = pred.eq(
        labels.data.view_as(pred)
    ).sum()  # 将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return int(rights), len(labels)  # 返回正确的数量和这一次一共比较了多少元素


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 3000  # 每轮训练总共训练的样本总数
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表
    diction = build_diction()
    # 建立模型
    model = TorchModel(diction)
    # 选择优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(train_sample // batch_size):
            x, y = build_dataset(batch_size, diction, sentence_length)  # 构造一组训练样本
            optimizer.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重
            watch_loss.append(loss.item())
            # 每隔50步，跑一下校验数据集的数据，输出临时结果
            if batch % 50 == 0:
                val_losses = []
                x, y = build_dataset(500, diction, sentence_length)  # 构造一组校验样本
                predict = model(x)
                # 调用rightness函数计算准确度
                right = rightness(predict, y)
                loss = model(x, y)
                val_losses.append(loss.data.numpy())

                # 将校验集合上面的平均准确度计算出来
                right_ratio = 1.0 * np.sum(right[0]) / np.sum(right[1])
                print(
                    "第{}轮，训练损失：{:.2f}, 校验损失：{:.2f}, 校验准确率: {:.2f}".format(
                        epoch, np.mean(watch_loss), np.mean(val_losses), right_ratio
                    )
                )
        log.append([right_ratio, np.mean(watch_loss)])
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(diction, ensure_ascii=False, indent=2))
    writer.close()


# 使用训练好的模型做预测
def predict(model_path, dict_path):
    diction = json.load(open(dict_path, "r", encoding="utf8"))  # 加载字符表
    model = TorchModel(diction)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    with torch.no_grad():
        for _ in range(6):
            x, y, text = build_sample(diction, sentence_length=5)
            pred = model(torch.tensor(x, dtype=torch.float).unsqueeze(0))
            print("输入：%s, 预测类别：%d, 实际标签：%d" % (text, torch.max(pred, 1)[1], y))  # 打印结果


if __name__ == "__main__":
    # main()
    predict("model.pth", "vocab.json")
