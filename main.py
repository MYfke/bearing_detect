import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchsummary import summary

from torchvision import datasets
from torchvision import transforms

class_num = 10
batch_size = 32


def convert2onehot(data, lenth):
    hot = np.zeros((lenth, class_num))
    for i in range(lenth - 1):
        index = data[i]
        hot[int(i), int(index)] = 1
    return hot


class BearingDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=",", dtype=np.float32, encoding="UTF-8-sig")
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[1:, 1:-1])
        self.y_data = convert2onehot(torch.from_numpy(xy[1:, [-1]]), self.len)

    def __getitem__(self, index):
        return self.x_data[index - 1], self.y_data[index - 1]

    def __len__(self):
        return self.len


train_dataset = BearingDataset("Bear_data/train.csv")
test_dataset = BearingDataset("Bear_data/test_data.csv")
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=0)


# mnist_train = datasets.MNIST(root='./dataset', train=True, transform=transforms.ToTensor(), download=True)
# mnist_test = datasets.MNIST(root='./dataset', train=False, transform=transforms.ToTensor(), download=True)
# train_loader = DataLoader(dataset=mnist_train, batch_size=32, shuffle=True)
# test_loader = DataLoader(dataset=mnist_test, batch_size=32, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 16, kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv1d(16, 16, kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv1d(16, 64, kernel_size=3, stride=2)
        self.conv4 = torch.nn.Conv1d(64, 64, kernel_size=3, stride=2)
        self.conv5 = torch.nn.Conv1d(64, 256, kernel_size=3, stride=2)
        self.conv6 = torch.nn.Conv1d(256, 256, kernel_size=3, stride=2)
        self.conv7 = torch.nn.Conv1d(256, 512, kernel_size=3, stride=2)
        self.conv8 = torch.nn.Conv1d(512, 512, kernel_size=3, stride=2)

        self.pooling = torch.nn.MaxPool1d(2)
        # self.pooling2 = torch.nn.MaxPool1d(5)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pooling(self.relu(self.conv2(self.relu(self.conv1(x)))))
        x = self.pooling(self.relu(self.conv4(self.relu(self.conv3(x)))))
        x = self.pooling(self.relu(self.conv6(self.relu(self.conv5(x)))))
        x = self.pooling(self.relu(self.conv8(self.relu(self.conv7(x)))))
        # x = self.pooling2(self.relu(self.conv7(x)))
        # x = self.pooling(self.relu(self.conv4(self.conv3(x))))
        # x = self.pooling(self.relu(self.conv6(self.conv5(x))))
        # x = self.pooling(self.relu(self.conv8(self.conv7(x))))

        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(summary(model, input_size=(1, 6000), device='cuda'))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def train():
    for batch_index, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs = inputs[:, np.newaxis, :]
        inputs, target = inputs.to(device), target.to(device)  # 迁移至GPU
        # 前向传播
        outputs = model(inputs)
        # 反向传播
        optimizer.zero_grad()  # 优化器清零
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()  # 更新

        if batch_index % 10 == 0:
            print("训练次数: {}, Loss: {}".format(batch_index, loss.item()))


def test():
    correct = 0  # 正确的样本
    total = 0  # 总共的样本
    with torch.no_grad():  # 不计算梯度
        for data in test_loader:
            inputs, target = data
            inputs = inputs[:, np.newaxis, :]
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            target = target.argmax(1)
            predicted = outputs.argmax(1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('测试集的正确率为: {}%  [{}/{}]'.format(100 * correct / total, correct, total))
    return correct / total


if __name__ == '__main__':
    epoch_list = []
    acc_list = []
    best_acc = 0
    for epoch in range(100):
        print("----------------第{}轮训练开始----------------".format(epoch + 1))
        train()
        acc = test()
        if best_acc < acc:
            torch.save(model, "model.pth")
            print("模型保存文件更新")
            best_acc = acc

        epoch_list.append(epoch)
        acc_list.append(acc)

    plt.plot(epoch_list, acc_list)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
