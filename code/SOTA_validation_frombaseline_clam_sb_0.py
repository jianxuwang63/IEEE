import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import glob
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import get_cosine_schedule_with_warmup, ViTModel, ViTConfig
from tqdm import tqdm  # 🔥 新增：用于显示动态进度条

import albumentations as A
from albumentations.pytorch import ToTensorV2

# 引入你的工具库
from utils import seed_everything, PathologyMILDataset, mil_collate_fn

# =========================================================
# 🔒 环境与 GPU 强制设定
# =========================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


# =========================================================
# 1. 配置文件
# =========================================================
class Config:
    seed = 2026
    pos_path = r"D:/data_of_wjx_lyf/yes"
    neg_path = r"D:/data_of_wjx_lyf/no"

    # 纯离线的本地权重路径
    pretrained_weight_path = r"phikon_base.pth"

    # 🔥 关键切换：'baseline', 'clam_sb', 'dsmil', 'dtfd'
    arch_type = 'clam_sb'

    save_dir = f"results_{arch_type}_5fold_cv"

    k_folds = 5
    batch_size = 8
    accumulation_steps = 4
    learning_rate = 1e-5
    weight_decay = 0.05
    label_smoothing = 0.1
    dropout_rate = 0.2
    warmup_ratio = 0.1
    epochs = 15

    num_workers = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs(Config.save_dir, exist_ok=True)


# =========================================================
# 2. 图像增强逻辑
# =========================================================
def get_strong_train_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_transforms(mode='valid'):
    if mode == 'valid':
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    return get_strong_train_transforms()


# =========================================================
# 3. 满血版 SOTA 模型架构集成 (纯本地离线 + 半精度安全版)
# =========================================================
class FullSOTA_MIL(nn.Module):
    def __init__(self, arch_type='baseline', init_weights_path=None):
        super(FullSOTA_MIL, self).__init__()
        self.arch_type = arch_type
        self.feature_dim = 768

        # 1. 手动配置 Phikon 空壳架构，切断网络
        config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=224,
            patch_size=16,
            num_channels=3
        )
        self.vit = ViTModel(config)

        # 2. 执行离线权重加载
        if init_weights_path and os.path.exists(init_weights_path):
            self._smart_load(init_weights_path)
        else:
            print(f"⚠️ 警告: 未找到本地权重 {init_weights_path}，使用随机初始化！")

        # 3. 初始化对应的聚合头
        if arch_type == 'baseline':
            self.aggregator = nn.Sequential(
                nn.Linear(self.feature_dim, 256), nn.ReLU(), nn.Linear(256, 2)
            )
        elif arch_type == 'clam_sb':
            self.attention_V = nn.Sequential(nn.Linear(self.feature_dim, 256), nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(self.feature_dim, 256), nn.Sigmoid())
            self.attention_weights = nn.Linear(256, 1)
            self.instance_classifier = nn.Linear(self.feature_dim, 2)
            self.bag_classifier = nn.Linear(self.feature_dim, 2)
        elif arch_type == 'dsmil':
            self.i_classifier = nn.Linear(self.feature_dim, 2)
            self.q_proj = nn.Linear(self.feature_dim, 128)
            self.v_proj = nn.Linear(self.feature_dim, 128)
            self.b_classifier = nn.Linear(self.feature_dim * 2, 2)
        elif arch_type == 'dtfd':
            self.num_pseudo_bags = 5
            self.tier1_attention = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.Tanh(), nn.Linear(128, 1))
            self.tier1_classifier = nn.Linear(self.feature_dim, 2)
            self.tier2_attention = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.Tanh(), nn.Linear(128, 1))
            self.tier2_classifier = nn.Linear(self.feature_dim, 2)

    def _smart_load(self, path):
        """本地离线权重智能加载"""
        print(f"🔄 正在从本地注入 Phikon 权重: {path}...")
        try:
            state_dict = torch.load(path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

            model_dict = self.vit.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if
                               k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.vit.load_state_dict(model_dict, strict=False)
            print(f"✅ 成功匹配并加载了 {len(pretrained_dict)} 个 Phikon 参数字典键值！")
        except Exception as e:
            print(f"❌ 权重注入失败，报错详情: {e}")

    def forward(self, x, mask=None):
        b, n, c, h, w = x.shape
        x = x.view(b * n, c, h, w)
        feat = self.vit(pixel_values=x).last_hidden_state[:, 0, :].view(b, n, -1)

        aux_dict = {}
        if self.arch_type == 'clam_sb':
            A_V, A_U = self.attention_V(feat), self.attention_U(feat)
            A = self.attention_weights(A_V * A_U).transpose(2, 1)
            # 🔥 修复为 -1e4 以支持 float16
            if mask is not None: A = A.masked_fill(~mask.unsqueeze(1), -1e4)
            A_soft = torch.softmax(A, dim=2)
            M = torch.bmm(A_soft, feat).squeeze(1)
            aux_dict.update({'A': A_soft.squeeze(1), 'inst_logits': self.instance_classifier(feat)})
            return self.bag_classifier(M), aux_dict

        elif self.arch_type == 'dsmil':
            inst_logits = self.i_classifier(feat)
            pos_scores = inst_logits[:, :, 1]
            if mask is not None: pos_scores = pos_scores.masked_fill(~mask, -1e4)  # 🔥 修复为 -1e4
            max_idx = torch.argmax(pos_scores, dim=1)
            max_f = feat[torch.arange(b), max_idx, :].unsqueeze(1)
            q, v = self.q_proj(max_f), self.v_proj(feat)
            attn = torch.bmm(q, v.transpose(2, 1))
            if mask is not None: attn = attn.masked_fill(~mask.unsqueeze(1), -1e4)  # 🔥 修复为 -1e4
            attn_soft = torch.softmax(attn, dim=2)
            M_final = torch.cat([max_f.squeeze(1), torch.bmm(attn_soft, feat).squeeze(1)], dim=1)
            aux_dict['max_inst_logits'] = inst_logits[torch.arange(b), max_idx, :]
            return self.b_classifier(M_final), aux_dict

        elif self.arch_type == 'dtfd':
            t1_list, t2_list = [], []
            for i in range(b):
                f_i = feat[i][torch.randperm(n)]
                chunk = max(1, n // self.num_pseudo_bags)
                p_fs = []
                for j in range(self.num_pseudo_bags):
                    sub = f_i[j * chunk: (j + 1) * chunk]
                    a = torch.softmax(self.tier1_attention(sub).T, dim=1)
                    p_m = torch.mm(a, sub)
                    p_fs.append(p_m);
                    t1_list.append(self.tier1_classifier(p_m))
                p_fs = torch.cat(p_fs, 0)
                a2 = torch.softmax(self.tier2_attention(p_fs).T, dim=1)
                t2_list.append(torch.mm(a2, p_fs))
            aux_dict['tier1_logits'] = torch.cat(t1_list, 0)
            return self.tier2_classifier(torch.cat(t2_list, 0)), aux_dict

        else:
            return self.aggregator(torch.mean(feat, 1)), aux_dict


# =========================================================
# 4. 辅助函数：SOTA 专属损失与训练
# =========================================================
def compute_sota_loss(logits, labels, aux_dict, arch_type, criterion):
    bag_loss = criterion(logits, labels)
    if arch_type == 'clam_sb':
        A, inst_logits, inst_loss = aux_dict['A'], aux_dict['inst_logits'], 0
        for i in range(labels.size(0)):
            idx = torch.argmax(A[i])
            inst_loss += criterion(inst_logits[i, idx].unsqueeze(0), labels[i].unsqueeze(0))
        return 0.7 * bag_loss + 0.3 * (inst_loss / labels.size(0))
    elif arch_type == 'dsmil':
        return 0.5 * bag_loss + 0.5 * criterion(aux_dict['max_inst_logits'], labels)
    elif arch_type == 'dtfd':
        t1_logits = aux_dict['tier1_logits']
        return bag_loss + 0.5 * criterion(t1_logits, labels.repeat_interleave(5))
    return bag_loss


def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    # 🔥 加入 tqdm 动态进度条
    pbar = tqdm(enumerate(loader), total=len(loader), desc="   Training", leave=False, ncols=100)

    for batch_idx, (images, labels, mask) in pbar:
        images, labels, mask = images.to(Config.device), labels.to(Config.device), mask.to(Config.device)
        with torch.cuda.amp.autocast():
            logits, aux = model(images, mask)
            loss = compute_sota_loss(logits, labels, aux, Config.arch_type, criterion) / Config.accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % Config.accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        current_loss = loss.item() * Config.accumulation_steps
        running_loss += current_loss

        # 实时更新 Loss 显示
        pbar.set_postfix({'Loss': f'{current_loss:.4f}'})

    return running_loss / len(loader)


def validate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch_data in loader:
            images, labels, mask = batch_data[0].to(Config.device), batch_data[1].to(Config.device), batch_data[-1].to(
                Config.device)
            with torch.cuda.amp.autocast():
                logits, _ = model(images, mask)
                probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)


# =========================================================
# 🎨 绘图函数
# =========================================================
def plot_cv_results(cv_results):
    plt.figure(figsize=(8, 8))
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []
    for i, res in enumerate(cv_results):
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_score'])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i + 1} (AUC = {roc_auc:.3f})')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})', lw=3,
             alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    plt.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), color='grey',
                     alpha=.2)
    plt.plot([0, 1], [0, 1], '--', lw=2, color='r')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(Config.save_dir, "CV_Mean_ROC.pdf"))
    plt.close()


# =========================================================
# 🚀 主程序
# =========================================================
def main():
    seed_everything(Config.seed)
    pos_files = sorted(glob.glob(os.path.join(Config.pos_path, '*')))
    neg_files = sorted(glob.glob(os.path.join(Config.neg_path, '*')))
    X, y = np.array(pos_files + neg_files), np.array([1] * len(pos_files) + [0] * len(neg_files))

    skf = StratifiedKFold(n_splits=Config.k_folds, shuffle=True, random_state=Config.seed)
    cv_results, fold_metrics = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n🔄 FOLD {fold + 1}/{Config.k_folds} | Arch: {Config.arch_type}")
        train_ds = PathologyMILDataset(X[train_idx].tolist(), y[train_idx].tolist(),
                                       transform=get_strong_train_transforms())
        val_ds = PathologyMILDataset(X[val_idx].tolist(), y[val_idx].tolist(), transform=get_transforms('valid'))

        train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, collate_fn=mil_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False, collate_fn=mil_collate_fn)

        model = FullSOTA_MIL(arch_type=Config.arch_type, init_weights_path=Config.pretrained_weight_path).to(
            Config.device)
        optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        total_steps = len(train_loader) // Config.accumulation_steps * Config.epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * Config.warmup_ratio), total_steps)
        criterion = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(Config.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler)
            # 🔥 每一轮都打印结果，让你随时掌控进度
            print(f"   [Ep {epoch + 1}/{Config.epochs}] T-Loss: {train_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(Config.save_dir, f"model_fold_{fold + 1}.pth"))

        val_labels, val_probs = validate(model, val_loader)
        auc_s = roc_auc_score(val_labels, val_probs)
        acc = ((val_probs > 0.5).astype(int) == val_labels).mean()

        print(f"✅ Fold {fold + 1} Result: AUC={auc_s:.4f} | ACC={acc:.4f}")
        cv_results.append({'y_true': val_labels, 'y_score': val_probs})
        fold_metrics.append({'Fold': fold + 1, 'AUC': auc_s, 'ACC': acc})

    df = pd.DataFrame(fold_metrics)
    print("\n", df)
    print(f"\nMean AUC: {df['AUC'].mean():.4f} ± {df['AUC'].std():.4f}")
    df.to_csv(os.path.join(Config.save_dir, "CV_Detailed_Metrics.csv"), index=False)
    plot_cv_results(cv_results)


if __name__ == "__main__":
    main()