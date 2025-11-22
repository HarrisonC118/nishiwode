# ==================== 1. 导入库和配置 ====================
import os
import cv2
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ==================== 2. 路径和超参数配置 ====================
ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "processed"
TRAIN_TXT = SPLIT_DIR / "rec_gt_train.txt"
VAL_TXT = SPLIT_DIR / "rec_gt_test.txt"
SAVE_PATH = ROOT / "best_size.pth"
LOG_CSV = ROOT / "train_log.csv"
IMG_H, IMG_W = 32, 96
BATCH_SIZE = 32
EPOCHS = 40
LR = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 3. 字符编码解码工具 ====================
alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ() -./"
char2idx = {c: i+1 for i, c in enumerate(alphabet)}
idx2char = {i+1: c for i, c in enumerate(alphabet)}

def encode(text): 
    """将文本转换为字符索引列表"""
    return [char2idx[c] for c in text if c in char2idx]

def decode_greedy(indices):
    """CTC解码：将索引序列转换为文本，去除重复和空白"""
    res, prev = [], None
    for i in indices:
        if i != 0 and i != prev:
            res.append(idx2char.get(i, ""))
            prev = i
    return "".join(res)

# ==================== 4. 数据集类定义 ====================
class SizeDataset(Dataset):
    """尺码规格数据集类，用于加载和预处理图片数据"""
    
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
        # 要求：根据索引返回图片tensor、标签tensor和标签长度
        p, t = self.samples[idx]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.resize(img, (IMG_W, IMG_H))
        import numpy as np
        import torch as _t
        img = (img / 255.0).astype("float32")
        img = _t.from_numpy(img).unsqueeze(0)
        target = _t.tensor(encode(t), dtype=_t.int32)
        return img, target, len(target)

def collate_fn(batch):
    """批处理函数：将不同长度的序列打包成批次"""
    import torch as _t
    imgs, targets, lengths = zip(*batch)
    imgs = _t.stack(imgs, 0)
    targets = _t.cat([t for t in targets])
    target_lengths = _t.tensor([l for l in lengths], dtype=_t.int32)
    return imgs, targets, target_lengths

# ==================== 5. 模型架构定义 ====================
class AttentionBlock(nn.Module):
    """注意力块：用于增强特征提取能力"""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch//4, 1)
        self.conv2 = nn.Conv2d(ch//4, ch, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        att = self.conv1(x)
        att = self.conv2(att)
        att = self.sigmoid(att)
        return x * att

class SizeCRNN(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        # TODO: 模型搭建 - 构建CNN卷积层
        # 要求：实现卷积层(32/64/128/256) + 池化层(2×2) + 全连接层(256→58)
        self.cnn = nn.Sequential(
            nn.Conv2d(1,32,3,1,1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2,1)),
            nn.Conv2d(128,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2,1)),
            nn.Conv2d(256,256,2,1,0), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.att_blocks = nn.Sequential(*[AttentionBlock(256) for _ in range(2)])
        self.rnn = nn.LSTM(256,128,num_layers=2,bidirectional=True,batch_first=True,dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128*2, nclass)
    
    def forward(self, x):
        # TODO: 前向传播 - 实现特征提取和序列识别
        # 要求：实现卷积特征提取、注意力机制、LSTM序列建模、全连接分类
        x = self.cnn(x)
        x = self.att_blocks(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x.permute(1, 0, 2)

def sequence_confidence(log_probs_TBC):
    probs = log_probs_TBC.exp().detach()
    best = probs.max(dim=2).values
    return best.mean(dim=0)
def average_precision(y_true, y_score):
    import numpy as np
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = (y_true == 1).astype(float)
    fp = (y_true == 0).astype(float)
    tp_cum = tp.cumsum()
    fp_cum = fp.cumsum()
    recall = tp_cum / max(1, tp.sum())
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    ap, prev_r = 0.0, 0.0
    for p, r in zip(precision, recall):
        ap += p * max(r - prev_r, 0)
        prev_r = r
    return float(ap)
def ctc_beam_decode(log_probs_TBC, beam_width=3):
    import numpy as np
    T, B, C = log_probs_TBC.size()
    res = []
    for b in range(B):
        beams = [('', 0.0)]
        for t in range(T):
            probs = log_probs_TBC[t, b].exp().cpu().numpy()
            new = {} 
            for pref,score in beams:
                for i,p in enumerate(probs):
                    if i==0:
                        new[pref] = max(new.get(pref,-1e9), score + float(np.log(p+1e-12)))
                    else:
                        ch = idx2char.get(i,'')
                        key = pref if (len(pref)>0 and pref[-1]==ch) else pref+ch
                        new[key] = max(new.get(key,-1e9), score + float(np.log(p+1e-12)))
            beams = sorted(new.items(), key=lambda x:x[1], reverse=True)[:beam_width]
        res.append(beams[0][0] if beams else '')
    return res
def normalize_size(s: str) -> str:
    """尺码规格标准化"""
    s = s.upper().strip()
    if s in ['ONE SIZE', 'ONESIZE']: return 'ONE SIZE'
    if s == 'NA': return 'N/A'
    return s
def evaluate(model, loader, criterion, use_beam=False, debug=False):
    model.eval()
    total_loss = 0.0
    total = 0
    y_true_list = []
    y_score_list = []
    debug_count = 0
    import torch as _t
    import numpy as np
    with _t.no_grad():
        for imgs, targets, target_lengths in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            T, B, C = logits.size()
            logits_cpu = logits.log_softmax(2).cpu()
            targets_cpu = targets.cpu()
            input_lengths = _t.full((B,), T, dtype=_t.int32)
            target_lengths_cpu = target_lengths
            loss = criterion(logits_cpu, targets_cpu, input_lengths, target_lengths_cpu)
            total_loss += float(loss.item()) * B
            total += B
            logp = logits.log_softmax(2)
            seq_idx = logp.argmax(2)
            preds = ctc_beam_decode(logp, 3) if use_beam else [decode_greedy(seq_idx[:, b].cpu().tolist()) for b in range(B)]
            conf = sequence_confidence(logp).cpu().numpy()
            start = 0
            gt_texts = []
            for L in target_lengths_cpu:
                ids = targets_cpu[start:start+L].tolist()
                gt_texts.append("".join(idx2char.get(i, '') for i in ids))
                start += L
            preds = [normalize_size(p) for p in preds]
            gt_texts = [normalize_size(g) for g in gt_texts]
            for p, g, c in zip(preds, gt_texts, conf):
                y_true_list.append(1 if p == g else 0)
                y_score_list.append(c)
                if debug and debug_count < 5:
                    print(f"  [Sample] GT: '{g}' | Pred: '{p}' | Match: {p == g}")
                    debug_count += 1
    if not y_true_list:
        return 0.0, 0.0
    import numpy as np
    val_loss = total_loss / max(1, total)
    ap = average_precision(np.array(y_true_list), np.array(y_score_list))
    return ap, val_loss
def train_model():
    # TODO: 训练函数 - 实现train_model函数
    # 要求：实现损失计算、批量训练、tqdm展示进度，确保有输出
    train_ds = SizeDataset(TRAIN_TXT)
    val_ds = SizeDataset(VAL_TXT)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    model = SizeCRNN(nclass=len(alphabet)+1).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    best_ap = -1.0
    logs = []
    
    from tqdm import tqdm
    for epoch in tqdm(range(1, EPOCHS+1), desc="Training"):
        model.train()
        for imgs, targets, target_lengths in train_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            T, B, C = logits.size()
            import torch as _t
            logits_cpu = logits.log_softmax(2).cpu()
            targets_cpu = targets.cpu()
            input_lengths = _t.full((B,), T, dtype=_t.int32)
            target_lengths_cpu = target_lengths
            loss = criterion(logits_cpu, targets_cpu, input_lengths, target_lengths_cpu)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        debug = (epoch == 1 or epoch % 8 == 0)
        val_ap, val_loss = evaluate(model, val_loader, criterion, use_beam=False, debug=debug)
        print(f"Epoch {epoch}: val AP = {val_ap:.4f} | val Loss = {val_loss:.4f}")
        logs.append({"epoch": epoch, "val_ap": val_ap, "val_loss": val_loss})
        if val_ap > best_ap:
            best_ap = val_ap
        # TODO: 模型保存 - 保存训练结果
        # 要求：将训练结果保存为.pth/.pt/.onnx/.pkl格式，确保文件保存成功
        torch.save(model.state_dict(), SAVE_PATH)
        scheduler.step(val_loss)
    pd.DataFrame(logs).to_csv(LOG_CSV, index=False)

if __name__ == "__main__":
    train_model()





