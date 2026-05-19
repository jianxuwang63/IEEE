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
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup
from transformers import ViTModel, ViTConfig

import matplotlib

# 🔥 1. 必须开启无头模式
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 假设 Dataset 还在 utils 里
from utils import seed_everything, PathologyMILDataset

warnings.filterwarnings("ignore")


class Config:
    seed = 2026
    pos_path = r"D:/data_of_wjx_lyf/yes"
    neg_path = r"D:/data_of_wjx_lyf/no"
    pretrained_weight_path = r"E:\WJX\2026.1.27\phikon_base.pth"

    # 📂 目录名：结合实验_含平均图
    save_dir = "results_kdd_Combined_With_AveragePlot"

    # 🌟 核心设定：只跑 Cost 5 和 10
    costs_to_run = [5, 10]

    # 🌟 核心设定：10次随机 MCCV
    n_runs = 10

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


os.makedirs(Config.save_dir, exist_ok=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =========================================================
# 🎨 KDD 矢量图配置 (严格 9pt)
# =========================================================
sns.set_theme(style="whitegrid", context="paper")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
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
# 🎨 绘图函数
# =========================================================
def save_vector_plot(filename):
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(Config.save_dir, f"{filename}.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(Config.save_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_combined_distribution(test_labels, test_probs, thresholds_dict, run_id, cost):
    """单次实验绘图"""
    plt.figure(figsize=(7.0, 3.5))
    sns.kdeplot(test_probs[test_labels == 0], shade=True, color="#1f77b4", label="Negative", alpha=0.2, clip=(0, 1),
                bw_adjust=0.6)
    sns.kdeplot(test_probs[test_labels == 1], shade=True, color="#d62728", label="Positive", alpha=0.2, clip=(0, 1),
                bw_adjust=0.6)

    line_colors = ['#2ca02c', '#ff7f0e', '#9467bd']
    sorted_items = sorted(thresholds_dict.items(), key=lambda x: x[1])
    y_min, y_max = plt.ylim()

    for i, (name, th) in enumerate(sorted_items):
        preds = (test_probs >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_labels, preds, labels=[0, 1]).ravel()
        real_test_fnr = fn / (fn + tp + 1e-8)
        col = line_colors[i % len(line_colors)]
        plt.axvline(th, color=col, linestyle='--', linewidth=1.2)
        text_y_pos = y_max * (0.90 - i * 0.15)
        label_text = f"Calib {name}\nTest FNR: {real_test_fnr:.1%}"
        plt.text(th + 0.01, text_y_pos, label_text,
                 color=col, fontsize=9, fontweight='bold', ha='left', va='center',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.9))

    plt.title(f"Run {run_id} Distribution (Cost {cost})", pad=10)
    plt.xlabel("Prediction Probability")
    plt.ylabel("Density")
    plt.xlim(0, 1.0)
    plt.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    save_vector_plot(f"Run{run_id}_Cost{cost}_Analysis")


# 🔥 新增：绘制平均图函数
def plot_average_distribution(all_labels, all_probs, mean_thresholds, mean_fnrs, cost):
    """
    [Average Analysis Plot]
    聚合所有 10 次 Run 的预测结果，展示整体分布，并画出平均阈值线。
    """
    plt.figure(figsize=(7.0, 3.5))

    # 1. 绘制聚合分布 (Aggregated Distribution)
    # 这代表了模型在所有 Run 中的“平均”概率行为
    sns.kdeplot(all_probs[all_labels == 0], shade=True, color="#1f77b4", label="Negative (Aggregated)", alpha=0.2,
                clip=(0, 1), bw_adjust=0.6)
    sns.kdeplot(all_probs[all_labels == 1], shade=True, color="#d62728", label="Positive (Aggregated)", alpha=0.2,
                clip=(0, 1), bw_adjust=0.6)

    line_colors = ['#2ca02c', '#ff7f0e', '#9467bd']
    # 排序：确保 Index 0 -> 4 -> 9 顺序
    # mean_thresholds 是一个 dict: {'Idx0 (1%)': 0.123, ...}
    sorted_items = sorted(mean_thresholds.items(), key=lambda x: x[1])
    y_min, y_max = plt.ylim()

    for i, (name, th) in enumerate(sorted_items):
        col = line_colors[i % len(line_colors)]

        # 2. 画平均阈值线
        plt.axvline(th, color=col, linestyle='--', linewidth=1.5)  # 平均线稍微粗一点

        # 3. 标注平均 FNR (从 mean_fnrs 获取)
        # name 类似 "Idx0 (1%)"
        avg_fnr = mean_fnrs.get(name, 0.0)

        text_y_pos = y_max * (0.90 - i * 0.15)
        label_text = f"Avg {name}\nMean FNR: {avg_fnr:.1%}"

        plt.text(th + 0.01, text_y_pos, label_text,
                 color=col, fontsize=9, fontweight='bold', ha='left', va='center',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.9))

    plt.title(f"Average Distribution & Thresholds over {Config.n_runs} Runs (Cost {cost})", pad=10)
    plt.xlabel("Prediction Probability")
    plt.ylabel("Density")
    plt.xlim(0, 1.0)
    plt.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)

    save_vector_plot(f"Average_Analysis_Cost{cost}")


# =========================================================
# 🚂 逻辑函数
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, epoch):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    num_batches = len(loader)

    for batch_idx, (images, labels, mask) in enumerate(loader):
        images, labels, mask = images.to(Config.device), labels.to(Config.device), mask.to(Config.device)
        with torch.cuda.amp.autocast():
            logits, _ = model(images, mask)
            loss = criterion(logits, labels) / Config.accumulation_steps
        scaler.scale(loss).backward()

        if (batch_idx + 1) % Config.accumulation_steps == 0 or (batch_idx + 1) == num_batches:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        current_loss = loss.item() * Config.accumulation_steps
        running_loss += current_loss

        if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
            print(f"\r   [Ep {epoch}] Batch {batch_idx}/{num_batches} | Loss: {current_loss:.4f}", end="")

    return running_loss / num_batches


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


def find_indices_on_validation(calib_labels, calib_probs):
    """固定索引 0, 4, 9"""
    pos_probs = calib_probs[calib_labels == 1]
    pos_probs_sorted = np.sort(pos_probs)
    n_pos = len(pos_probs_sorted)

    target_indices = {
        "Idx0 (1%)": 0,
        "Idx4 (5%)": 4,
        "Idx9 (10%)": 9
    }

    thresholds = {}
    for name, idx in target_indices.items():
        if idx < n_pos:
            th = pos_probs_sorted[idx]
            thresholds[name] = th
        else:
            thresholds[name] = pos_probs_sorted[-1]

    return thresholds


# =========================================================
# 🚀 主程序
# =========================================================
def main():
    seed_everything(Config.seed)
    print(f"🚀 Starting Experiment (Cost 5&10, Runs 10, With Average Plot)...")

    # 1. 准备数据
    pos_files = sorted(glob.glob(os.path.join(Config.pos_path, '*')))
    neg_files = sorted(glob.glob(os.path.join(Config.neg_path, '*')))
    all_files = np.array(pos_files + neg_files)
    all_labels = np.array([1] * len(pos_files) + [0] * len(neg_files))

    # 存储用于计算 Average 的数据
    # 结构: aggregated_data[cost] = {'labels': [], 'probs': []}
    aggregated_data = {c: {'labels': [], 'probs': []} for c in Config.costs_to_run}
    all_metrics = []

    # 🌟 第一层循环：10次随机划分 (MCCV)
    for run in range(1, Config.n_runs + 1):
        print(f"\n{'=' * 60}")
        print(f"🎬 [Run {run}/{Config.n_runs}] Random Split Data...")
        print(f"{'=' * 60}")

        current_seed = Config.seed + run
        train_total_idx, test_idx = train_test_split(
            np.arange(len(all_files)), test_size=0.2, random_state=current_seed, stratify=all_labels
        )

        pool_files = all_files[train_total_idx]
        pool_labels = all_labels[train_total_idx]
        pool_pos_idx = np.where(pool_labels == 1)[0]
        pool_neg_idx = np.where(pool_labels == 0)[0]

        rng = np.random.RandomState(current_seed)
        rng.shuffle(pool_pos_idx)

        n_calib_pos = 100
        if len(pool_pos_idx) < n_calib_pos:
            n_calib_pos = len(pool_pos_idx)

        calib_pos_idx = pool_pos_idx[:n_calib_pos]
        train_pos_idx = pool_pos_idx[n_calib_pos:]

        train_indices_local = np.concatenate([train_pos_idx, pool_neg_idx])
        calib_indices_local = calib_pos_idx

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

        # 🌟 第二层循环：Cost (仅 5, 10)
        for cost in Config.costs_to_run:
            print(f"\n   🧪 [Run {run}] Training with Cost Weight = {cost} ...")

            model = PhikonMIL(init_weights_path=Config.pretrained_weight_path).to(Config.device)
            optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
            total_steps = len(train_loader) // Config.accumulation_steps * Config.epochs
            scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * Config.warmup_ratio), total_steps)
            scaler = torch.cuda.amp.GradScaler()

            class_weights = torch.tensor([1.0, float(cost)], device=Config.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=Config.label_smoothing)

            for epoch in range(Config.epochs):
                train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, epoch + 1)
            print("")

            # 3. 阈值确定 (Calib Index 0, 4, 9)
            model.eval()
            calib_probs, calib_labels_real = get_probs(model, calib_loader)
            thresholds_map = find_indices_on_validation(calib_labels_real, calib_probs)

            # 4. 测试
            test_probs, test_labels_real = get_probs(model, test_loader)

            # 🔥 收集数据用于画平均图
            aggregated_data[cost]['labels'].extend(test_labels_real)
            aggregated_data[cost]['probs'].extend(test_probs)

            try:
                test_auc = roc_auc_score(test_labels_real, test_probs)
            except:
                test_auc = 0.0

            print(f"      ✅ Cost {cost} Result -> Test AUC: {test_auc:.4f}")

            run_result = {"Run": run, "Cost": cost, "Test_AUC": test_auc}

            for target_name, th in thresholds_map.items():
                preds = (test_probs >= th).astype(int)
                tn, fp, fn, tp = confusion_matrix(test_labels_real, preds, labels=[0, 1]).ravel()
                real_fnr = fn / (fn + tp + 1e-8)
                real_acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

                # 记录阈值本身，以便后续算平均
                run_result[f"Th_Val_{target_name}"] = th
                run_result[f"Real_FNR_{target_name}"] = real_fnr
                run_result[f"Real_ACC_{target_name}"] = real_acc

            all_metrics.append(run_result)

            # 5. 单次绘图
            plot_combined_distribution(test_labels_real, test_probs, thresholds_map, run, cost)

            save_name = f"model_run{run}_cost{cost}.pth"
            torch.save(model.state_dict(), os.path.join(Config.save_dir, save_name))
            print(f"      💾 Weights saved: {save_name}")

            del model, optimizer, scaler, scheduler
            torch.cuda.empty_cache()
            gc.collect()

    # =========================================================
    # 📊 结果汇总与平均绘图
    # =========================================================
    print("\n" + "=" * 80)
    print("📊 Generating Average Plots & Final Stats...")
    print("=" * 80)

    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(Config.save_dir, "Detailed_Metrics.csv"), index=False)

    summary = df.groupby("Cost").agg(['mean', 'std'])

    # 🌟 为每个 Cost 生成平均图
    for cost in Config.costs_to_run:
        # 1. 获取该 Cost 的所有聚合数据
        agg_labels = np.array(aggregated_data[cost]['labels'])
        agg_probs = np.array(aggregated_data[cost]['probs'])

        # 2. 计算平均阈值
        mean_thresholds = {}
        mean_fnrs = {}
        for name in ["Idx0 (1%)", "Idx4 (5%)", "Idx9 (10%)"]:
            th_mean = summary.loc[cost, (f"Th_Val_{name}", 'mean')]
            fnr_mean = summary.loc[cost, (f"Real_FNR_{name}", 'mean')]

            mean_thresholds[name] = th_mean
            mean_fnrs[name] = fnr_mean

        # 3. 画平均图
        plot_average_distribution(agg_labels, agg_probs, mean_thresholds, mean_fnrs, cost)
        print(f"   🖼️  Saved Average Plot for Cost {cost}")

    print("\n" + "-" * 90)
    print(f"{'Cost':<10} | {'AUC':<20} | {'Idx0 (1%) FNR':<25} | {'Idx4 (5%) FNR':<25}")
    print("-" * 90)

    for cost in Config.costs_to_run:
        auc_str = f"{summary.loc[cost, ('Test_AUC', 'mean')]:.4f}±{summary.loc[cost, ('Test_AUC', 'std')]:.4f}"
        fnr1_str = f"{summary.loc[cost, ('Real_FNR_Idx0 (1%)', 'mean')] * 100:.2f}±{summary.loc[cost, ('Real_FNR_Idx0 (1%)', 'std')] * 100:.2f}%"
        fnr5_str = f"{summary.loc[cost, ('Real_FNR_Idx4 (5%)', 'mean')] * 100:.2f}±{summary.loc[cost, ('Real_FNR_Idx4 (5%)', 'std')] * 100:.2f}%"

        print(f"{cost:<10} | {auc_str:<20} | {fnr1_str:<25} | {fnr5_str:<25}")

    print(f"\n✅ All Done. Check {Config.save_dir}")


if __name__ == "__main__":
    main()