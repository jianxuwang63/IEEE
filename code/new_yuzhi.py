import os
import gc
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import glob
import pandas as pd
# 🔥 确保导入 roc_auc_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup
from transformers import ViTModel, ViTConfig

import matplotlib

# 🔥 1. 必须开启无头模式
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 假设 Dataset 还在 utils 里
from utils import seed_everything, PathologyMILDataset

warnings.filterwarnings("ignore")


class Config:
    seed = 2026
    pos_path = r"D:/data_of_wjx_lyf/yes"
    neg_path = r"D:/data_of_wjx_lyf/no"
    pretrained_weight_path = r"/2026.3.31/phikon_base.pth"

    # 📂 目录名：最终修正版 (含平均图)
    save_dir = "results_kdd_Threshold_Final_With_AvgPlot"

    batch_size = 8
    accumulation_steps = 4
    epochs = 15
    learning_rate = 1e-5
    weight_decay = 0.05
    label_smoothing = 0.1
    dropout_rate = 0.2
    warmup_ratio = 0.1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = 0
    n_runs = 10


os.makedirs(Config.save_dir, exist_ok=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =========================================================
# 🎨 KDD 矢量图 (Vector Graphics) 严格 9pt 配置
# =========================================================
sns.set_theme(style="whitegrid", context="paper")

plt.rcParams.update({
    # --- 字体设置 ---
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,  # 🔥 核心：确保嵌入 TrueType 矢量字体
    'ps.fonttype': 42,

    # --- 🌟 字号 (严格修正为 9pt) ---
    'font.size': 9,  # 全局基础字号
    'axes.labelsize': 9,  # 轴标签
    'axes.titlesize': 9,  # 标题
    'xtick.labelsize': 9,  # 刻度
    'ytick.labelsize': 9,  # 刻度
    'legend.fontsize': 9,  # 图例

    # --- 分辨率 ---
    'figure.dpi': 300,
    'savefig.dpi': 300,

    # --- 线条 ---
    'lines.linewidth': 1.2,
    'axes.linewidth': 1.0,
})


# =========================================================
# 🛠️ 核心工具函数
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


def get_transforms(mode='train'):
    if mode == 'train':
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])


# =========================================================
# 🧠 模型定义
# =========================================================
class PhikonMIL(nn.Module):
    def __init__(self, init_weights_path=None):
        super(PhikonMIL, self).__init__()
        config = ViTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                           intermediate_size=3072, image_size=224, patch_size=16, num_channels=3)
        self.vit = ViTModel(config)
        self.feature_dim = 768
        self.attention_V = nn.Sequential(nn.Linear(self.feature_dim, 256), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.feature_dim, 256), nn.Sigmoid())
        self.attention_weights = nn.Linear(256, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=Config.dropout_rate),
            nn.Linear(self.feature_dim, 2)
        )
        if init_weights_path:
            self._smart_load(init_weights_path)

    def _smart_load(self, path):
        try:
            state_dict = torch.load(path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            model_dict = self.vit.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items()
                               if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.vit.load_state_dict(model_dict, strict=False)
        except Exception as e:
            print(f"   ❌ Weight load warning: {e}")

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


# =========================================================
# 🎨 矢量图绘制函数
# =========================================================
def save_vector_plot(filename):
    plt.tight_layout(pad=0.3)
    pdf_path = os.path.join(Config.save_dir, f"{filename}.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(Config.save_dir, f"{filename}_preview.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      🖼️  Saved Vector PDF: {pdf_path}")


def plot_kde_vector_double_column(test_labels, test_probs, thresholds_dict, run_id):
    """单次实验绘图：Positive / Negative"""
    plt.figure(figsize=(7.0, 3.5))

    # 1. 绘制分布 (Label 严格改为 Negative / Positive)
    sns.kdeplot(test_probs[test_labels == 0], shade=True, color="#1f77b4", label="Negative", alpha=0.2,
                clip=(0, 1), bw_adjust=0.6)
    sns.kdeplot(test_probs[test_labels == 1], shade=True, color="#d62728", label="Positive", alpha=0.2,
                clip=(0, 1), bw_adjust=0.6)

    # 2. 绘制阈值线
    line_colors = ['#2ca02c', '#ff7f0e', '#9467bd']
    sorted_items = sorted(thresholds_dict.items(), key=lambda x: x[1])
    y_min, y_max = plt.ylim()

    for i, (name, th) in enumerate(sorted_items):
        preds = (test_probs >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_labels, preds, labels=[0, 1]).ravel()
        actual_fnr = fn / (fn + tp + 1e-8)
        col = line_colors[i % len(line_colors)]

        plt.axvline(th, color=col, linestyle='--', linewidth=1.2)

        text_y_pos = y_max * (0.90 - i * 0.15)
        plt.text(th + 0.01, text_y_pos,
                 f"FNR: {actual_fnr:.1%}",
                 color=col, fontsize=9, fontweight='bold',
                 ha='left', va='center',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.9))

    plt.title(f"Distribution of Prediction Probabilities (Run {run_id})", pad=10)
    plt.xlabel("Prediction Probability (Risk Score)")
    plt.ylabel("Density")
    plt.xlim(0, 1.0)
    plt.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    save_vector_plot(f"Run_{run_id}_Thresholds_Wide")


# 🔥 新增：绘制所有 Run 平均分布的函数
def plot_average_threshold_distribution(all_labels, all_probs, mean_thresholds, mean_fnrs):
    """
    [Average Analysis Plot]
    聚合所有 10 次 Run 的数据绘制分布，并标出平均阈值
    """
    plt.figure(figsize=(7.0, 3.5))

    # 1. 绘制聚合分布 (Aggregated Distribution)
    sns.kdeplot(all_probs[all_labels == 0], shade=True, color="#1f77b4", label="Negative (Aggregated)", alpha=0.2,
                clip=(0, 1), bw_adjust=0.6)
    sns.kdeplot(all_probs[all_labels == 1], shade=True, color="#d62728", label="Positive (Aggregated)", alpha=0.2,
                clip=(0, 1), bw_adjust=0.6)

    line_colors = ['#2ca02c', '#ff7f0e', '#9467bd']
    # 排序：Strict -> Mid -> Loose
    sorted_items = sorted(mean_thresholds.items(), key=lambda x: x[1])
    y_min, y_max = plt.ylim()

    for i, (name, th) in enumerate(sorted_items):
        col = line_colors[i % len(line_colors)]

        # 2. 画平均阈值线
        plt.axvline(th, color=col, linestyle='--', linewidth=1.5)

        # 3. 标注平均 FNR
        avg_fnr = mean_fnrs.get(name, 0.0)

        text_y_pos = y_max * (0.90 - i * 0.15)

        # 显示 "Avg Strict" 等字样
        # 去掉前缀 "Thresh_" 让图更干净
        display_name = name.replace("Thresh_", "")

        plt.text(th + 0.01, text_y_pos,
                 f"Avg {display_name}\nMean FNR: {avg_fnr:.1%}",
                 color=col, fontsize=9, fontweight='bold',
                 ha='left', va='center',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.9))

    plt.title(f"Aggregated Distribution & Average Thresholds ({Config.n_runs} Runs)", pad=10)
    plt.xlabel("Prediction Probability (Risk Score)")
    plt.ylabel("Density")
    plt.xlim(0, 1.0)
    plt.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)

    save_vector_plot("Average_Threshold_Analysis")


# =========================================================
# 🚂 训练流程
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


def get_probs(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 3:
                images, labels, mask = batch_data
            else:
                images, labels, mask = batch_data[0], batch_data[1], batch_data[-1]
            images, mask = images.to(Config.device), mask.to(Config.device)
            with torch.cuda.amp.autocast():
                logits, _ = model(images, mask)
                prob = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(prob.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


# =========================================================
# 🚀 主程序
# =========================================================
def main():
    seed_everything(Config.seed)
    print(f"🚀 Starting Threshold Selection (Runs={Config.n_runs}, With Average Plot)...")

    pos_files = sorted(glob.glob(os.path.join(Config.pos_path, '*')))
    neg_files = sorted(glob.glob(os.path.join(Config.neg_path, '*')))
    all_files = np.array(pos_files + neg_files)
    all_labels = np.array([1] * len(pos_files) + [0] * len(neg_files))

    all_metrics = []

    # 🔥 用于聚合所有 Run 的数据以绘制平均图
    aggregated_probs = []
    aggregated_labels = []

    for run in range(1, Config.n_runs + 1):
        print(f"\n{'=' * 50}\n🎬 [Run {run}/{Config.n_runs}] Training...\n{'=' * 50}")

        current_seed = Config.seed + run
        train_total_idx, test_idx = train_test_split(
            np.arange(len(all_files)), test_size=0.2, random_state=current_seed, stratify=all_labels
        )

        pool_files = all_files[train_total_idx]
        pool_labels = all_labels[train_total_idx]
        pool_pos_indices = np.where(pool_labels == 1)[0]
        pool_neg_indices = np.where(pool_labels == 0)[0]

        rng = np.random.RandomState(current_seed)
        rng.shuffle(pool_pos_indices)
        calib_indices_local = pool_pos_indices[:100]
        train_pos_indices_local = pool_pos_indices[100:]
        train_indices_local = np.concatenate([train_pos_indices_local, pool_neg_indices])

        train_ds = PathologyMILDataset(pool_files[train_indices_local].tolist(),
                                       pool_labels[train_indices_local].tolist(), transform=get_transforms('train'))
        calib_ds = PathologyMILDataset(pool_files[calib_indices_local].tolist(),
                                       pool_labels[calib_indices_local].tolist(), transform=get_transforms('valid'))
        test_ds = PathologyMILDataset(all_files[test_idx].tolist(), all_labels[test_idx].tolist(),
                                      transform=get_transforms('valid'))

        train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, pin_memory=True,
                                  collate_fn=mil_collate_fn)
        calib_loader = DataLoader(calib_ds, batch_size=1, shuffle=False, collate_fn=mil_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=mil_collate_fn)

        model = PhikonMIL(init_weights_path=Config.pretrained_weight_path).to(Config.device)
        optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        total_steps = len(train_loader) // Config.accumulation_steps * Config.epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * Config.warmup_ratio), total_steps)
        criterion = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(Config.epochs):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler)
            print(f"\r   Ep {epoch + 1}: Loss {loss:.4f}", end="")
        print("")

        torch.save(model.state_dict(), os.path.join(Config.save_dir, f"model_run_{run}.pth"))

        # 计算阈值
        model.eval()
        calib_probs, _ = get_probs(model, calib_loader)
        calib_probs = np.sort(calib_probs)

        # 阈值选择: Indices 0, 4, 9
        thresh_map = {
            "Thresh_Strict": calib_probs[0],
            "Thresh_Mid": calib_probs[4],
            "Thresh_Loose": calib_probs[9]
        }

        # 验证
        test_probs, test_labels_real = get_probs(model, test_loader)

        # 🔥 收集数据到聚合列表 (用于最后的平均图)
        aggregated_probs.extend(test_probs)
        aggregated_labels.extend(test_labels_real)

        try:
            auc_score = roc_auc_score(test_labels_real, test_probs)
        except ValueError:
            auc_score = 0.0

        print(f"   📊 Run {run} Result -> AUC: {auc_score:.4f}")

        run_res = {"Run": run, "AUC": auc_score}

        for name, th in thresh_map.items():
            preds = (test_probs >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(test_labels_real, preds, labels=[0, 1]).ravel()
            run_res[f"{name}_Th"] = th
            run_res[f"{name}_FNR"] = fn / (fn + tp + 1e-8)
            run_res[f"{name}_ACC"] = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        all_metrics.append(run_res)

        # 绘图: 单次 Run
        plot_kde_vector_double_column(test_labels_real, test_probs, thresh_map, run)

        del model, optimizer, scaler, scheduler
        torch.cuda.empty_cache()
        gc.collect()

    # =========================================================
    # 📊 结果汇总与平均绘图
    # =========================================================
    print("\n" + "=" * 80)
    print("📊 Generating Average Plot & Final Stats...")
    print("=" * 80)

    df = pd.DataFrame(all_metrics)

    # 1. 计算平均阈值
    mean_thresholds = {
        "Thresh_Strict": df["Thresh_Strict_Th"].mean(),
        "Thresh_Mid": df["Thresh_Mid_Th"].mean(),
        "Thresh_Loose": df["Thresh_Loose_Th"].mean()
    }

    # 2. 计算平均 FNR (用于标注平均图)
    mean_fnrs = {
        "Thresh_Strict": df["Thresh_Strict_FNR"].mean(),
        "Thresh_Mid": df["Thresh_Mid_FNR"].mean(),
        "Thresh_Loose": df["Thresh_Loose_FNR"].mean()
    }

    # 3. 绘制平均图 🔥
    plot_average_threshold_distribution(
        np.array(aggregated_labels),
        np.array(aggregated_probs),
        mean_thresholds,
        mean_fnrs
    )

    # 4. 打印汇总表格
    print(f"🏆 Overall AUC: {df['AUC'].mean():.4f} ± {df['AUC'].std():.4f}")
    print("-" * 80)
    ordered_keys = ["Thresh_Strict", "Thresh_Mid", "Thresh_Loose"]
    print(f"{'Level':<15} | {'FNR (%)':<20} | {'ACC (%)':<20}")
    print("-" * 60)
    for key in ordered_keys:
        fnrs = df[f"{key}_FNR"] * 100
        accs = df[f"{key}_ACC"] * 100
        print(f"{key:<15} | {fnrs.mean():.2f} ± {fnrs.std():.2f} | {accs.mean():.2f} ± {accs.std():.2f}")

    df.to_csv(os.path.join(Config.save_dir, "Final_Results.csv"), index=False)
    print(f"\n✅ Done. Check {Config.save_dir}")


if __name__ == "__main__":
    main()