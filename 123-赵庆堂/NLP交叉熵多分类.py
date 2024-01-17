import random
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot

# 创建字符集列表,0位是pad,末位unk(未知字符占用)
def build_char_dict():
    chars = 'abcdefgh'
    char_dict = {'pad':0}
    # 通过enumerate遍历string的索引和值
    for index, char in enumerate(chars):
        char_dict[char] = index + 1
    # 由于该字典value从0开始,所以真实长度是value+1,可以通过真实长度巧妙的将新key插入字典
    char_dict['unk'] = len(char_dict)
    return char_dict

# 创建一个数据样本
# 6分类样本,判断样本集中'a'元素所在位置,并返回'a'的真实索引,若'a'元素不存在返回6
def build_one_data(char_dict, sentence_length):
    # 通过random.sample使得每次不重复抽取列表中的元素
    char_list = random.sample(list(char_dict), sentence_length)
    # 通过random.choice使得每次抽取列表的一个元素,会产生重复数据
    # for i in range(sentence_length):
    #     char = random.choice(list(char_dict))
    #     char_list.append(char)
    if 'a' in char_list:
        label = char_list.index('a')
    else:
        label = 5
    # 将取出字符在字典中所对应的值取出,为了embedding层计算
    char_value_list = [char_dict.get(char) for char in char_list]
    return char_value_list, label

# 创建数据样本集
def build_data_set(data_number, char_dict, sentence_length):
    char_set = []
    label_set = []
    for i in range(data_number):
        char_value_list, label = build_one_data(char_dict, sentence_length)
        char_set.append(char_value_list)
        label_set.append(label)
    return torch.LongTensor(char_set), torch.LongTensor(label_set)

# 定义
class TorchRNN(nn.Module):
    def __init__(self, char_dim, hidden_size, char_dict):
        super(TorchRNN, self).__init__()
        self.layer = nn.Embedding(len(char_dict), char_dim)
        self.linear_RNN = nn.RNN(char_dim, hidden_size, bias=True, batch_first=True)
        self.linear_sort = nn.Linear(hidden_size, 6)
        self.loss = nn.functional.cross_entropy
# 计算loss值
    def forward(self, char, label = None):
        embedding_out = self.layer(char)
        char_tenor, char_result = self.linear_RNN(embedding_out)
        char_sort = self.linear_sort(char_result)
        char_expect = char_sort.squeeze()
        if label is not None:
            return self.loss(char_expect, label)
        else:
            return char_expect

def  evaluate(model):
    model.eval()      #模型进入测试模式
    data_number = 100    #样本数量
    sentence_length = 5
    char, label = build_data_set(data_number, build_char_dict(), sentence_length)
    # torch.sum()是求和用的,在其中加上条件判断,用他来计算每个样本的总和
    count_result_0 = torch.sum(label == 0)
    count_result_1 = torch.sum(label == 1)
    count_result_2 = torch.sum(label == 2)
    count_result_3 = torch.sum(label == 3)
    count_result_4 = torch.sum(label == 4)
    count_result_5 = torch.sum(label == 5)
    print('''
    本次预测集中有
    0号样本:{}个
    1号样本:{}个
    2号样本:{}个
    3号样本:{}个
    4号样本:{}个
    '''.format(count_result_0, count_result_1, count_result_2, count_result_3, count_result_4, count_result_5))
    correct, wrong = 0, 0
    with torch.no_grad():                                            # with代表声明状态,这里使用.no_grad是不进行反向传播(即不会计算梯度,也不会有梯度下降过程)
        char_expect = model(char)                                          # 预测期望x值
        for char_exp, label_true in zip(char_expect, label):
            if torch.argmax(char_exp) == int(label_true):             #torch.argmax()是取出该tensor最大值索引用的,torch.max()就是取出该tensor最大值索引用的
                correct += 1
            else:
                wrong += 1
    print('''
    正确预测个数:{}
    错误预测个数:{}
    正确率:{}%
    '''.format(correct, wrong, correct * 100 / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    epoch_num = 20                  #训练轮数
    batch_size = 10                  #每次训练样本个数
    every_train_sample_num = 5000    #每轮训练样本个数
    char_dim = 8                     #每个字符的维度
    hidden_size = 16
    char_dict = build_char_dict()
    sentence_length = 5              #每句话的长度
    learn_rate = 0.0001               #学习率
    # 建立模型
    model = TorchRNN(char_dim, hidden_size, char_dict)
    # 选择Adam的优化器(SGD高级版)优化器作用：梯度下降找loss极小值
    # .parameters()是可训练的所有参数(就是线性层中y=wx+b中的w与b)
    optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate)
    log = []                                                               #日志空list
    # 创建由build_dateset函数调用的训练集
    train_x, train_y = build_data_set(every_train_sample_num, char_dict, sentence_length)               #创建训练集次数的total_sample_num入参被'每轮训练样本个数'赋值
    # 训练过程
    for epoch in range(epoch_num):
        model = model.train()                                              #将模型设置为训练状态
        watch_loss = []
        for batch_index in range(every_train_sample_num // batch_size):    #'//'代表整除,即每轮需要循环5000/10=500次
            # train_x与train_y都是tensor数据类型,可以用类似list索引取值的方式对每个batch_size取一次值(x,y也是tensor类型)
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y_label = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y_label)          #计算loss值
            loss.backward()                   #梯度计算
            optimizer.step()                  #更新优化器权重
            optimizer.zero_grad()             #梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)                 # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
        # 保存模型
    torch.save(model.state_dict(), "model_2.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model_nlp.pt")
    # 保存词表
    writer = open("char_dict.json", "w", encoding="utf8")
    writer.write(json.dumps(char_dict, ensure_ascii=False, indent=2))
    writer.close()
    return

# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 8
    hidden_size = 16
    char_dict = json.load(open(vocab_path, "r", encoding="utf8"))
    # vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = TorchRNN(char_dim, hidden_size, char_dict)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([char_dict[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, torch.argmax(result[i]), result[i])) #打印结果



if __name__ == "__main__":
    main()
    test_strings = ["abcde", "bacde", "cbade", "bcdae", "bcdea", "bcdef"]
    predict("model_nlp.pt", "char_dict.json", test_strings)
