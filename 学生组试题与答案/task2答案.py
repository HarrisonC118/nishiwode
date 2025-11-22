# ==================== 1. 导入库和配置 ====================
import os, cv2, torch, pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ==================== 2. 路径和超参数配置 ====================
ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "processed"
TRAIN_TXT = SPLIT_DIR / "rec_gt_train.txt"
VAL_TXT = SPLIT_DIR / "rec_gt_test.txt"
SAVE_PATH = ROOT / "best_barcode.pth"
LOG_CSV = ROOT / "train_log.csv"
IMG_H, IMG_W = 80, 80
BATCH_SIZE = 32
EPOCHS = 25
LR = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 3. 类别编码解码工具 ====================
# 条码数量类别：1-1000的数字（简化版）
num_classes = 1000

def encode_label(text):
    """将条码数量文本转换为类别索引"""
    try:
        num = int(text)
        if 1 <= num <= 1000:
            return num - 1  # 索引从0开始，所以减1
        else:
            return 0  # 默认返回类别0
    except:
        return 0

def decode_label(idx):
    """将类别索引转换为条码数量文本"""
    return str(idx + 1)  # 索引从0开始，所以加1

def validate_number(text):
    """验证条码数量范围"""
    try:
        num = int(text)
        return 1 <= num <= 1000
    except:
        return False

# ==================== 4. 数据集类定义 ====================
class BarcodeDataset(Dataset):
    """条码数量数据集类，用于加载和预处理图片数据"""
    
    def __init__(self, list_txt: Path):
        # TODO: 数据集封装 - 实现__init__方法
        # 要求：读取文本文件，解析图片路径和标签，存储到self.samples列表中
        self.samples = []
        with open(list_txt, "r", encoding="utf-8") as f:
            for line in f:
                p, t = line.rstrip("\n").split("\t", 1)
                self.samples.append((p, t))
    
    def __len__(self): 
        # TODO: 数据集封装 - 实现__len__方法
        # 要求：返回数据集样本数量
        return len(self.samples)
    
    def __getitem__(self, idx):
        # TODO: 数据集封装 - 实现__getitem__方法
        # 要求：根据索引返回图片tensor和标签tensor
        p, t = self.samples[idx]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.resize(img, (IMG_W, IMG_H))
        import numpy as np
        import torch as _t
        img = (img / 255.0).astype("float32")
        img = _t.from_numpy(img).unsqueeze(0)
        target = _t.tensor(encode_label(t), dtype=_t.long)
        return img, target

# ==================== 5. 模型架构定义 ====================
class DepthwiseBlock(nn.Module):
    """深度可分离卷积块：用于特征提取"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class BarcodeCNN(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        # TODO: 模型搭建 - 构建CNN卷积层
        # 要求：实现深度可分离卷积层(16/32/64/128) + 池化层(2×2) + 全连接层(128→10000000000000)
        self.cnn = nn.Sequential(
            DepthwiseBlock(1, 16, 3, 1, 1),
            DepthwiseBlock(16, 32, 3, 1, 1),
            DepthwiseBlock(32, 64, 3, 1, 1),
            DepthwiseBlock(64, 128, 3, 1, 1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, nclass)
    
    def forward(self, x):
        # TODO: 前向传播 - 实现特征提取和分类
        # 要求：实现深度可分离卷积特征提取、全局平均池化、全连接分类
        x = self.cnn(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def calculate_accuracy(outputs, targets):
    """计算分类准确率"""
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    import torch as _t
    with _t.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            total_loss += float(loss.item()) * imgs.size(0)
            total += imgs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
    val_loss = total_loss / max(1, total)
    val_acc = correct / max(1, total)
    return val_acc, val_loss

def train_model():
    # TODO: 训练函数 - 实现train_model函数
    # 要求：实现损失计算、批量训练、tqdm展示进度，确保有输出
    train_ds, val_ds = BarcodeDataset(TRAIN_TXT), BarcodeDataset(VAL_TXT)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    model = BarcodeCNN(nclass=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.7)
    best_acc = -1.0
    logs = []
    
    from tqdm import tqdm
    for epoch in tqdm(range(1, EPOCHS + 1), desc="Training"):
        model.train()
        for imgs, targets in train_loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        debug = (epoch == 1 or epoch % 5 == 0)
        val_acc, val_loss = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch}: val Acc = {val_acc:.4f} | val Loss = {val_loss:.4f}")
        logs.append({"epoch": epoch, "val_acc": val_acc, "val_loss": val_loss})
        if val_acc > best_acc:
            best_acc = val_acc
        # TODO: 模型保存 - 保存训练结果
        # 要求：将训练结果保存为.pth/.pt/.onnx/.pkl格式，确保文件保存成功
        torch.save(model.state_dict(), SAVE_PATH)
        scheduler.step()
    pd.DataFrame(logs).to_csv(LOG_CSV, index=False)

if __name__ == "__main__":
    train_model()





