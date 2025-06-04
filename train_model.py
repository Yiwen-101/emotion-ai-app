raise RuntimeError("✅ 如果你看到这个，说明脚本确实执行了")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 配置
DATA_DIR = './data'
MODEL_PATH = 'emotion_model.pt'
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001

# 图像预处理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# 模型结构
class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)

# 主程序
if __name__ == "__main__":
    print("🚀 正在运行主函数", flush=True)

    # 加载数据
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    print("📦 已经加载数据！", flush=True)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 开始训练
    print("🏋️ 正在训练模型...")
    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"📘 Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}", flush=True)

    # 评估模型
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"🎯 测试准确率: {100 * correct / total:.2f}%", flush=True)

    # 保存模型
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ 模型保存成功：{MODEL_PATH}", flush=True)
