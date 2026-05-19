import os
import gc
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold  # 引入分层K折
import matplotlib

# 🔥 核心修复：添加 glob 库
import glob

# 🌟 设置无头模式
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import ViTModel, ViTConfig
from transformers import get_cosine_schedule_with_warmup
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 引用 utils
from utils import seed_everything, PathologyMILDataset, get_transforms

# 🔥 指定 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings("ignore")

# 🌟 SCI 绘图风格设置
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'


class Config:
    seed = 2026
    pos_path = r"D:/data_of_wjx_lyf/yes"
    neg_path = r"D:/data_of_wjx_lyf/no"
    pretrained_weight_path = r"E:\WJX\2026.1.27\phikon_base.pth"

    # 📂 结果保存目录
    save_dir = "results_cost_sensitive_5fold_cv"

    costs_to_run = [5, 10, 15, 20]
    n_folds = 5  # 🌟 5折交叉验证

    # 🌟 严格对齐 Baseline
    batch_size = 8
    accumulation_steps = 4
    epochs = 15

    learning_rate = 1e-5
    weight_decay = 0.05
    label_smoothing = 0.1
    dropout_rate = 0.2
    warmup_ratio = 0.1

    num_workers = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


os.makedirs(Config.save_dir, exist_ok=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# =========================================================
# 🎨 核心绘图函数库 (支持 5 折聚合)
# =========================================================
def save_plot(filename):
    plt.savefig(os.path.join(Config.save_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(Config.save_dir, f"{filename}.pdf"), format='pdf', bbox_inches='tight')
    plt.close()


def plot_aggregate_confusion_matrix(all_labels, all_preds, cost):
    """绘制 5 折累计混淆矩阵"""
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14, "weight": "bold"})
    plt.title(f'Aggregate Confusion Matrix (Cost={cost}, 5-Fold)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    save_plot(f"CM_Aggregate_Cost_{cost}")


def plot_prob_density_aggregate(all_labels, all_probs, cost):
    """绘制 5 折累计概率密度"""
    plt.figure(figsize=(8, 6))
    sns.kdeplot(all_probs[all_labels == 0], shade=True, color="blue", label="Normal", alpha=0.3, clip=(0, 1))
    sns.kdeplot(all_probs[all_labels == 1], shade=True, color="red", label="Cancer", alpha=0.3, clip=(0, 1))
    plt.title(f'Aggregate Probability Density (Cost={cost})')
    plt.xlabel('Predicted Probability of Cancer')
    plt.ylabel('Density')
    plt.legend()
    save_plot(f"Density_Aggregate_Cost_{cost}")


def plot_comparative_roc_cv(cost_metrics):
    """绘制不同 Cost 下 5 折平均 ROC"""
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("viridis", len(cost_metrics))

    for idx, (cost, metrics) in enumerate(cost_metrics.items()):
        # metrics['all_labels'] 和 metrics['all_probs'] 包含了所有 5 折的数据
        fpr, tpr, _ = roc_curve(metrics['all_labels'], metrics['all_probs'])
        roc_auc = auc(fpr, tpr)

        # 计算平均 FNR
        avg_fnr = metrics['mean_fnr']

        plt.plot(fpr, tpr, lw=2, color=colors[idx],
                 label=f'Cost {cost} (AUC={roc_auc:.4f}, Mean FNR={avg_fnr:.2%})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Comparative ROC Curves (5-Fold Aggregate)')
    plt.legend(loc="lower right", fontsize=11)
    save_plot("Comparative_ROC_5Fold")


def plot_tradeoff_bar_cv(results_df):
    """绘制带有误差棒的 Trade-off 图"""
    plt.figure(figsize=(10, 6))

    # 聚合数据
    agg = results_df.groupby('Cost Setting').agg({
        'FNR': ['mean', 'std'],
        'ACC': ['mean', 'std']
    }).reset_index()

    costs = agg['Cost Setting'].astype(str)
    fnr_mean = agg[('FNR', 'mean')]
    fnr_std = agg[('FNR', 'std')]
    acc_mean = agg[('ACC', 'mean')]
    acc_std = agg[('ACC', 'std')]

    x = np.arange(len(costs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, fnr_mean, width, yerr=fnr_std, label='Mean FNR', color='#d62728', alpha=0.8,
                    capsize=5)
    rects2 = ax.bar(x + width / 2, acc_mean, width, yerr=acc_std, label='Mean Accuracy', color='#1f77b4', alpha=0.8,
                    capsize=5)

    ax.set_ylabel('Score')
    ax.set_xlabel('Cost Weight')
    ax.set_title('5-Fold Trade-off Analysis: FNR vs Accuracy (Mean ± Std)')
    ax.set_xticks(x)
    ax.set_xticklabels(costs)
    ax.legend()
    ax.set_ylim(0, 1.1)

    save_plot("Tradeoff_Bar_5Fold_CV")


# =========================================================
# 🔄 常规组件 (Collate, Transforms, Model, Train)
# =========================================================
def mil_collate_fn(batch):
    images_list, labels = zip(*batch)
    c, h, w = images_list[0].shape[1:]
    lengths = [img.shape[0] for img in images_list]
    max_n = max(lengths)
    batch_size = len(images_list)
    padded_images = torch.zeros(batch_size, max_n, c, h, w)
    mask = torch.zeros(batch_size, max_n, dtype=torch.bool)
    for i, img in enumerate(images_list):
        n = lengths[i]
        padded_images[i, :n] = img
        mask[i, :n] = True
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_images, labels, mask


def get_strong_train_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()
    ])


class PhikonMIL(nn.Module):
    def __init__(self, init_weights_path=None):
        super(PhikonMIL, self).__init__()
        config = ViTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,
                           image_size=224, patch_size=16, num_channels=3)
        self.vit = ViTModel(config)
        self.feature_dim = 768
        self.attention_V = nn.Sequential(nn.Linear(self.feature_dim, 256), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.feature_dim, 256), nn.Sigmoid())
        self.attention_weights = nn.Linear(256, 1)
        self.classifier = nn.Sequential(nn.Dropout(p=Config.dropout_rate), nn.Linear(self.feature_dim, 2))
        if init_weights_path: self._smart_load(init_weights_path)

    def _smart_load(self, path):
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
        except Exception as e:
            pass  # silent load

    def forward(self, x, mask=None):
        if isinstance(x, (tuple, list)): x = x[0]
        b, n, c, h, w = x.size()
        x = x.view(b * n, c, h, w)
        outputs = self.vit(pixel_values=x)
        features = outputs.last_hidden_state[:, 0, :]
        features = features.view(b, n, self.feature_dim)
        A_V = self.attention_V(features)
        A_U = self.attention_U(features)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 2, 1)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)
            A = A.masked_fill(~mask_expanded, -1e4)
        A = torch.softmax(A, dim=2)
        M = torch.bmm(A, features).squeeze(1)
        logits = self.classifier(M)
        return logits, A


def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, epoch_idx):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    num_batches = len(loader)

    for batch_idx, (images, labels, mask) in enumerate(loader):
        images, labels, mask = images.to(Config.device), labels.to(Config.device), mask.to(Config.device)
        with torch.cuda.amp.autocast():
            logits, _ = model(images, mask)
            loss = criterion(logits, labels)
            loss = loss / Config.accumulation_steps
        scaler.scale(loss).backward()
        if (batch_idx + 1) % Config.accumulation_steps == 0 or (batch_idx + 1) == num_batches:
            scaler.step(optimizer);
            scaler.update();
            scheduler.step();
            optimizer.zero_grad();
            torch.cuda.empty_cache()

        running_loss += loss.item() * Config.accumulation_steps

        # 简化日志输出，只打印少量
        if batch_idx % 50 == 0:
            print(f"\r   Batch {batch_idx}/{num_batches}", end="")

    return running_loss / num_batches


def validate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 3:
                images, labels, mask = batch_data
            else:
                images, labels, mask = batch_data[0], batch_data[1], batch_data[-1]
            images, labels, mask = images.to(Config.device), labels.to(Config.device), mask.to(Config.device)
            with torch.cuda.amp.autocast():
                logits, _ = model(images, mask)
                probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)


# =========================================================
# 🚀 主程序 (Cost Loop -> 5-Fold Loop)
# =========================================================
def main():
    seed_everything(Config.seed)
    print(f"🚀 Starting 5-Fold Cost-Sensitive Experiments...")

    # 1. 准备所有数据
    pos_files = glob.glob(os.path.join(Config.pos_path, '*'))
    neg_files = glob.glob(os.path.join(Config.neg_path, '*'))
    X = np.array(pos_files + neg_files)
    y = np.array([1] * len(pos_files) + [0] * len(neg_files))

    # 定义 5 折
    skf = StratifiedKFold(n_splits=Config.n_folds, shuffle=True, random_state=Config.seed)

    # 存储所有结果
    fold_results = []
    cost_metrics = {}  # 存储每个Cost的聚合数据用于绘图

    # 循环 Cost
    for cost in Config.costs_to_run:
        print(f"\n{'=' * 60}")
        print(f"🧪 COST SETTING: {cost} (Running 5 Folds)")
        print(f"{'=' * 60}")

        # 存储当前 Cost 下 5 折的所有预测值 (用于聚合绘图)
        current_cost_labels = []
        current_cost_probs = []
        current_cost_preds = []
        fnr_list = []  # 记录每折的 FNR 以计算平均值

        # 循环 Folds
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n   >>> Fold {fold + 1}/{Config.n_folds} ...")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_ds = PathologyMILDataset(X_train.tolist(), y_train.tolist(), transform=get_strong_train_transforms())
            val_ds = PathologyMILDataset(X_val.tolist(), y_val.tolist(), transform=get_transforms('valid'))

            train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True,
                                      num_workers=Config.num_workers, pin_memory=True, collate_fn=mil_collate_fn)
            val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False,
                                    num_workers=Config.num_workers, pin_memory=True, collate_fn=mil_collate_fn)

            # 模型初始化
            model = PhikonMIL(init_weights_path=Config.pretrained_weight_path).to(Config.device)
            optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
            total_steps = len(train_loader) // Config.accumulation_steps * Config.epochs
            scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * Config.warmup_ratio), total_steps)
            scaler = torch.cuda.amp.GradScaler()

            # Cost-Sensitive Loss
            class_weights = torch.tensor([1.0, float(cost)], device=Config.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=Config.label_smoothing)

            # 训练 (跑满 15 Epoch，不早停)
            for epoch in range(Config.epochs):
                loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, epoch + 1)

            # 验证 (取最后一个 Epoch)
            val_labels, val_probs = validate(model, val_loader)
            preds = (val_probs > 0.5).astype(int)

            # 收集这一折的数据
            current_cost_labels.extend(val_labels)
            current_cost_probs.extend(val_probs)
            current_cost_preds.extend(preds)

            # 计算这一折的指标
            tn, fp, fn, tp = confusion_matrix(val_labels, preds).ravel()
            fnr = fn / (fn + tp + 1e-7)
            acc = (tp + tn) / (tp + tn + fp + fn + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            try:
                auc_score = roc_auc_score(val_labels, val_probs)
            except:
                auc_score = 0.0

            fnr_list.append(fnr)

            # 记录到总表
            fold_results.append({
                "Cost Setting": cost,
                "Fold": fold + 1,
                "AUC": auc_score,
                "FNR": fnr,
                "ACC": acc,
                "Recall": recall
            })

            print(f"       [Fold {fold + 1} Result] FNR: {fnr:.4f} | ACC: {acc:.4f}")

            model_name = f"model_cost_{cost}_fold_{fold + 1}.pth"
            save_path = os.path.join(Config.save_dir, model_name)
            torch.save(model.state_dict(), save_path)
            print(f"       ✅ [Model Saved] Weights saved to: {model_name}")
            # 清理显存
            del model, optimizer, scaler, scheduler, criterion
            torch.cuda.empty_cache()
            gc.collect()

        # --- 5 折结束后的聚合操作 ---
        all_labels_np = np.array(current_cost_labels)
        all_probs_np = np.array(current_cost_probs)
        all_preds_np = np.array(current_cost_preds)

        # 1. 绘制聚合混淆矩阵
        plot_aggregate_confusion_matrix(all_labels_np, all_preds_np, cost)

        # 2. 绘制聚合概率密度
        plot_prob_density_aggregate(all_labels_np, all_probs_np, cost)

        # 3. 保存聚合信息供对比图使用
        cost_metrics[cost] = {
            'all_labels': all_labels_np,
            'all_probs': all_probs_np,
            'mean_fnr': np.mean(fnr_list)
        }

    # =========================================================
    # 📊 最终对比分析 (Cross-Cost Comparison)
    # =========================================================
    print(f"\n>>> 📊 Generating 5-Fold Comparative Analysis...")
    df = pd.DataFrame(fold_results)

    # 1. 绘制带有误差棒的 Trade-off 图
    plot_tradeoff_bar_cv(df)

    # 2. 绘制 5 折平均 ROC 对比图
    plot_comparative_roc_cv(cost_metrics)

    # 3. 保存详细数据
    df.to_excel(os.path.join(Config.save_dir, "Final_5Fold_Metrics.xlsx"), index=False)

    # 4. 打印聚合统计表
    print("\n=== Aggregated Results (Mean ± Std) ===")
    agg_df = df.groupby('Cost Setting').agg({
        'FNR': ['mean', 'std'],
        'ACC': ['mean', 'std'],
        'AUC': ['mean', 'std']
    })
    print(agg_df)

    print(f"✅ All 5-Fold Experiments Complete! Check folder: {Config.save_dir}")


if __name__ == "__main__":
    main()