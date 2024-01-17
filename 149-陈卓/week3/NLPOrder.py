import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import random
import matplotlib.pyplot as plt

"""
判断某个字符在某个字符串内是否存在， 如果存在返回该字符出现的顺序（从1开始）， 如果不存在返回0
"""

# 构建数据
# 加载词表(从词表中随机选词)
with open('vocab.json') as vocab:
    vocab = json.load(vocab)
    
# 定义函数，从词表中随机抽取字符
# 固定字符长度为8
def extract_str(vocab):
    key_list = list(vocab.keys())[1:10]
    selected_keys = random.sample(key_list, 8)
    return ''.join(selected_keys)

# 定义函数，构建样本
def set_str(nums):
    str_list = []
    
    i = 0
    while i < nums:
        strings = extract_str(vocab)
        str_list.append(strings)        
        i += 1
    
    return str_list

# 定义函数把字符串转为索引
def vocab_to_index(string, vocab):
    return [vocab[s] for s in string]

# 定义函数，构建标签
# 判断字母c
def set_label(str_list):
    label_list = []
    
    for char in str_list:
        label = char.find('c') + 1
        label_list.append(label)
    
    return label_list

# 构建一个具有400个样本的数据集
x_train = set_str(400)
print(f"训练输入的字符串样本为：{x_train}")
y_train = set_label(x_train)
for i in range(len(x_train)):
    x_train[i] = vocab_to_index(x_train[i], vocab)
print(f"训练字符串转换成索引后是：{x_train}")
print(f"训练的标签是:{y_train}")

# 训练数据转换为张量
train_featrues_tensor = torch.tensor(x_train)
train_label_tensor = torch.tensor(y_train)

print(train_featrues_tensor)
print(train_label_tensor)

# 创建dataset
train_set = TensorDataset(train_featrues_tensor, train_label_tensor)
train_data = DataLoader(dataset=train_set, batch_size=20, shuffle=True)


# 再构建一个具有20个样本的测试集
x_test = set_str(20)
print(f"用于测试的字符串是：{x_test}")
y_test = set_label(x_test)
x_test_index = []
for i in range(len(x_test)):
    x_index = vocab_to_index(x_test[i], vocab)
    x_test_index.append(x_index)
print(f"用于测试的标签是：{y_test}")

# 测试数据转换为张量
test_featrues_tensor = torch.tensor(x_test_index)
test_label_tensor = torch.tensor(y_test)


# 定义模型结构
class OrderModelLinear(nn.Module):
    """
    视为一个分类问题，0~8类，不存在返回0，存在返回1~8位置
    embedding层 -> RNN层 -> Linear层 -> softmax激活函数
    """
      
    # 构造函数入参：str_size词表中所有字符的种类, str_dim每个字符用几个维度表示, 
    # max_length每个字符串最大长度
    def __init__(self, str_size, str_dim, max_length, rnn_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(str_size, str_dim) # embedding层
        # batch * max_length * str_dim

        self.rnn = nn.RNN(str_dim, rnn_hidden_size, bias = True, batch_first=True) # RNN
        # 输出batch*max_len*rnn_hiden_size
        
        # 线性层处理成9分类问题
        self.linear = nn.Linear(rnn_hidden_size, max_length + 1)
        # 输出batch*1*max_length+1
        
        # 使用torch.nn.CrossEntropyLoss已经内置了softmax,模型直接输出线性层的结果
        
    def forward(self, x):
        # batch * max_len * str_dim
        x = self.embedding(x)
        
        # 要把RNN的输出分解出来，才能得到每步的输出
        # batch* max_len *str_dim -> batch*max_len*rnn_hiden
        x = self.rnn(x)[0][:, -1, :] 
        
        y_pred = self.linear(x)
        # batch*max_len*9
        
        return y_pred
    

# 定义一个评估函数用于判断
def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        # 输出logit对应的类别，(每行最大值对应的索引)
        _, pred_classes = torch.max(y_pred, dim=1)
        pred_classes.tolist()
        y.tolist()
        
        # 与真实的标签对比
        correct = 0
        wrong = 0
        for i in range(len(pred_classes)):
            if pred_classes[i] == y[i]:
                correct += 1
            else:
                wrong += 1
                
        print(f"正确预测的个数是{correct}, 正确率是{correct / (correct + wrong)}")
        return correct / (correct + wrong)      


# 模型训练

str_size = 28
str_dim = 6
max_length = 8
rnn_hidden_size = 12

order_model = OrderModelLinear(str_size, str_dim, max_length, rnn_hidden_size)

# 定义优化器和loss
optim = torch.optim.Adam(order_model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

losses = []
log = []

for i in range(50):
    train_loss = 0
    order_model.train()
    for tdata, tlabel in train_data:
        y_pred = order_model(tdata)
        # y_pred形状是50*8*9 tlabel的形状是 50
        loss = criterion(y_pred, tlabel)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss = train_loss + loss.item()
        
    losses.append(train_loss / len(train_data))
    acc = evaluate(order_model, tdata, tlabel)
    # 损失的理解是：每个batch计算一次平均损失，用于反向传播
    # 下面输出的是每轮，各个batch计算的平均损失的平均值
    print("第{}轮训练的损失为{}".format(i, losses[i]))
    # 把每轮训练的准确率和平均损失记录下来，用于画图
    log.append([acc, train_loss / len(train_data)]) 


# 画图直观显示每轮训练的成果
plt.plot(range(len(log)), [l[0] for l in log], label="acurace")
plt.plot(range(len(log)), [l[1] for l in log], label="loss")
plt.legend()
plt.show()


# 使用测试集进行模型预测，并评估准确率
def test(model, x, x_str):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        # 输出logit对应的类别，(每行最大值对应的索引)
        _, pred_classes = torch.max(y_pred, dim=1)
        pred_classes.tolist()
        
        
        # 输出预测结果        
        for i in range(len(pred_classes)):
            if pred_classes[i] == 0:
                print(f"'c'在字符串{x_str[i]}中没有出现")
            else:
                print(f"'c'在字符串{x_str[i]}中位于第{pred_classes[i]}位")
                
test(order_model, test_featrues_tensor, x_test)