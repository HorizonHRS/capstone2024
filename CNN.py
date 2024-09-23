import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np

# 调整图像的大小（可以考虑使用标准尺寸，如224x224）
image_size = (160, 90)

# 图像转换和标准化，使用ImageNet的标准化参数
transform = transforms.Compose([
    transforms.Resize(image_size),    # 调整图像大小
    transforms.RandomHorizontalFlip(), # 数据增强，随机水平翻转
    transforms.ToTensor(),            # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 使用ImageNet的均值和标准差
                         std=[0.229, 0.224, 0.225])
])

# 数据路径
data_dir = r'D:\capstone2024\Landscape-Dataset'

# 加载数据集
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 将数据集分为训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 定义卷积神经网络模型，添加dropout防止过拟合
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 输入通道3（RGB图像），输出通道16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # dropout层防止过拟合
        self.dropout = nn.Dropout(0.5)

        # 计算全连接层的输入大小
        self.fc1_input_size = self._get_fc1_input_size()
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)   # 输出类别

    def _get_fc1_input_size(self):
        with torch.no_grad():
            # 创建一个dummy输入来通过网络，计算特征图的尺寸
            dummy_input = torch.zeros(1, 3, image_size[0], image_size[1])
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 经过conv1和pool1
        x = self.pool(F.relu(self.conv2(x)))  # 经过conv2和pool2

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层，加入dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 使用dropout防止过拟合
        x = self.fc2(x)
        return x

# 假设有4个不同的风景类别
num_classes = 4
model = CNNModel(num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 添加学习率调度器，每5个epoch后将学习率减少一半
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# 早停策略相关变量
best_val_acc = 0.0
patience = 3  # 如果在3个epoch内验证集表现没有提升，停止训练
patience_counter = 0

# 检查是否可以使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练轮数
num_epochs = 20  # 增加epoch以观察过拟合

# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}')

    # 验证模型
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    print(f'Validation Accuracy: {val_acc:.2f}%, Validation Loss: {avg_val_loss:.4f}')

    # 更新学习率调度器
    scheduler.step()

    # Early Stopping 检查
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0  # 如果验证集准确率提高，重置计数器
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("早停策略激活，停止训练")
            break

print("训练完成，最佳验证集准确率:", best_val_acc)
