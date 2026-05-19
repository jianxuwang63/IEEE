import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
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
from transformers import get_cosine_schedule_with_warmup

import albumentations as A
from albumentations.pytorch import ToTensorV2

# 引入 utils
from utils import seed_everything, PathologyMILDataset, get_transforms, PhikonMIL, mil_collate_fn

# =========================================================
# 🔒 关键修改：强制覆盖 GPU 设置
# =========================================================
# 虽然 utils 里可能设置了 "1,0"，但我们在 PyTorch 正式使用显卡前
# 把它强制改为 "0"，这样这份代码就只会跑在物理 GPU 0 上，
# 与跑在 GPU 1 上的其他实验互不干扰。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


class Config:
    seed = 2026
    pos_path = r"D:/data_of_wjx_lyf/yes"
    neg_path = r"D:/data_of_wjx_lyf/no"
    pretrained_weight_path = r"phikon_base.pth"

    # 📂 结果保存目录
    save_dir = "results_baseline_5fold_cv"

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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# =========================================================
# 🚂 基础训练函数
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for batch_idx, (images, labels, mask) in enumerate(loader):
        images, labels, mask = images.to(Config.device), labels.to(Config.device), mask.to(Config.device)
        with torch.cuda.amp.autocast():
            logits, _ = model(images, mask)
            loss = criterion(logits, labels) / Config.accumulation_steps
        scaler.scale(loss).backward()
        if (batch_idx + 1) % Config.accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.step(optimizer);
            scaler.update();
            scheduler.step();
            optimizer.zero_grad();
            torch.cuda.empty_cache()
        running_loss += loss.item() * Config.accumulation_steps
    return running_loss / len(loader)


def validate(model, loader, criterion):
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


def get_strong_train_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()
    ])


# =========================================================
# 🎨 高级绘图：平均 ROC 和 PR 曲线 (带阴影)
# =========================================================
def plot_cv_results(cv_results):
    print(f"\nGenerating Aggregate CV Plots (PDF) in '{Config.save_dir}'...")

    # --- 1. Mean ROC Curve ---
    plt.figure(figsize=(8, 8))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

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
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (5-Fold CV)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(Config.save_dir, "CV_Mean_ROC.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(Config.save_dir, "CV_Mean_ROC.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 2. Confusion Matrix (Aggregate) ---
    all_y_true = np.concatenate([res['y_true'] for res in cv_results])
    all_y_score = np.concatenate([res['y_score'] for res in cv_results])
    all_y_pred = (all_y_score > 0.5).astype(int)

    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14, "weight": "bold"})
    plt.title('Aggregate Confusion Matrix (All Folds)')
    plt.savefig(os.path.join(Config.save_dir, "CV_Aggregate_CM.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(Config.save_dir, "CV_Aggregate_CM.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Plots Saved.")


# =========================================================
# 🚀 主程序
# =========================================================
def main():
    seed_everything(Config.seed)

    # 1. 准备全量数据
    pos_files = sorted(glob.glob(os.path.join(Config.pos_path, '*')))
    neg_files = sorted(glob.glob(os.path.join(Config.neg_path, '*')))
    X = np.array(pos_files + neg_files)
    y = np.array([1] * len(pos_files) + [0] * len(neg_files))

    print(f"Total Samples: {len(X)} (Pos: {len(pos_files)}, Neg: {len(neg_files)})")
    print(f"Starting {Config.k_folds}-Fold Cross-Validation (GPU 0 Only)...")

    # 2. 定义 K-Fold
    skf = StratifiedKFold(n_splits=Config.k_folds, shuffle=True, random_state=Config.seed)

    cv_results = []
    fold_metrics = []

    # 3. 循环 5 折
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n" + "=" * 40)
        print(f"🔄 FOLD {fold + 1}/{Config.k_folds}")
        print("=" * 40)

        # 数据集划分
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = PathologyMILDataset(X_train.tolist(), y_train.tolist(), transform=get_strong_train_transforms())
        val_ds = PathologyMILDataset(X_val.tolist(), y_val.tolist(), transform=get_transforms('valid'))

        train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True,
                                  num_workers=Config.num_workers, pin_memory=True, collate_fn=mil_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False,
                                num_workers=Config.num_workers, pin_memory=True, collate_fn=mil_collate_fn)

        # 初始化模型 (每一折都重新初始化！)
        model = PhikonMIL(init_weights_path=Config.pretrained_weight_path).to(Config.device)
        optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        total_steps = len(train_loader) // Config.accumulation_steps * Config.epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * Config.warmup_ratio), total_steps)
        criterion = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing)
        scaler = torch.cuda.amp.GradScaler()

        # 训练 15 轮
        for epoch in range(Config.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler)
            if (epoch + 1) % 5 == 0:
                print(f"   [Fold {fold + 1} Ep {epoch + 1}] T-Loss: {train_loss:.4f}")

        # 保存本折模型 (Last Epoch)
        fold_model_path = os.path.join(Config.save_dir, f"model_fold_{fold + 1}.pth")
        torch.save(model.state_dict(), fold_model_path)

        # 验证本折
        val_labels, val_probs = validate(model, val_loader, criterion)

        preds = (val_probs > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(val_labels, preds).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        auc_score = roc_auc_score(val_labels, val_probs)
        fnr = fn / (fn + tp + 1e-8)

        print(f"✅ Fold {fold + 1} Result: AUC={auc_score:.4f} | ACC={acc:.4f} | FNR={fnr:.4f}")

        cv_results.append({
            'fold': fold + 1,
            'y_true': val_labels,
            'y_score': val_probs
        })
        fold_metrics.append({
            'Fold': fold + 1,
            'AUC': auc_score, 'ACC': acc, 'FNR': fnr, 'Recall': tp / (tp + fn + 1e-8)
        })

    # =========================================================
    # 📊 汇总报告
    # =========================================================
    print("\n" + "=" * 60)
    print("📊 5-FOLD CV SUMMARY")
    print("=" * 60)

    df = pd.DataFrame(fold_metrics)
    print(df)

    print("-" * 60)
    print(f"Mean AUC: {df['AUC'].mean():.4f} ± {df['AUC'].std():.4f}")
    print(f"Mean ACC: {df['ACC'].mean():.4f} ± {df['ACC'].std():.4f}")
    print(f"Mean FNR: {df['FNR'].mean():.4f} ± {df['FNR'].std():.4f}")

    df.to_csv(os.path.join(Config.save_dir, "CV_Detailed_Metrics.csv"), index=False)
    plot_cv_results(cv_results)


if __name__ == "__main__":
    main()