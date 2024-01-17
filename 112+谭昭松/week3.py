
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import json
#import matplotlib.pyplot as plt

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x, y=None):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        output = output[:, -1, :]
        y_pred = self.fc(output)
        if y is not None:
            return self.loss(y_pred, y.squeeze().long())
        else:
            return y_pred

#构建词表
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab

# 建立数据集，无a的时候标记为0，a的位置标志为对应的数字
def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    a_positions = [i for i, char in enumerate(x) if char == 'a']
    y = max(a_positions) if a_positions else 0
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 统计误差，loss和accurary
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    #print(x, y)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)

        y_pred_class = torch.argmax(y_pred, dim=1)

        correct += torch.sum(y_pred_class == y).item()
        wrong += torch.sum(y_pred_class != y).item()

    print("正确预测个数: %d, 正确率: %f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def build_model(input_size, hidden_size, output_size):
    model = RNNModel(input_size, hidden_size, output_size)
    return model

def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.005
    vocab = build_vocab()
    model = build_model(len(vocab), char_dim, sentence_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optimizer.zero_grad()
            y_pred = model(x, y)
            loss = y_pred  # loss is returned directly from the forward method
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print("=========\nEpoch %d - 平均 Loss: %f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
    # plt.plot(range(len(log)), [l[0] for l in log], label="Accuracy")
    # plt.plot(range(len(log)), [l[1] for l in log], label="Loss")
    # plt.legend()
    # plt.show()   #画图内核崩溃？报错内核已终止？
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# # 使用训练好的模型做预测
def predict(model_path, vocab_path, test_strings):
    
    vocab_path = "vocab.json"
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(len(vocab), char_dim, sentence_length)
    model_path = "model.pth"
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in test_strings:
        x.append([vocab[char] for char in input_string])
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
        probabilities = F.softmax(result, dim=1)
    for i, input_string in enumerate(test_strings):
        predicted_class = torch.argmax(result[i]).item()
        probability = probabilities[i][predicted_class].item()
        print("输入：%s, 预测类别：%d" % (input_string, predicted_class))

if __name__ == "__main__":
    main()
    test_strings = ["fnvfe", "asdfg", "radeg", "nkwaw","nkwba","fnvfe", "zsdfg", "qwdeg", "nakww"]
    predict("model.pth", "vocab.json", test_strings)

