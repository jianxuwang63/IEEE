import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import confusion_matrix, roc_auc_score

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import ViTModel, ViTConfig

# 引入你的 utils
from utils import seed_everything, PathologyMILDataset, get_transforms, mil_collate_fn

warnings.filterwarnings("ignore")


# =========================================================
# 🔧 外部验证配置 (Cost + Yuzhi / Hard NP Calibration)
# =========================================================
class ExternalCostYuzhiConfig:
    seed = 2026
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 📂 【需要修改】Zenodo 外部数据集路径
    zenodo_pos_path = r"E:\WJX\external_dataset\yes"
    zenodo_neg_path = r"E:\WJX\external_dataset\no"

    # 📂 【必须与 cost+yuzhi.py 的输出一致】
    weights_dir = "results_kdd_Combined_With_AveragePlot"
    save_dir = "results_external_cost_yuzhi"

    batch_size = 8
    num_workers = 4

    # 你的 cost+yuzhi.py 跑了 10 个 run，以及 Cost 5 和 10
    n_runs = 10
    costs_to_run = [5, 10]

    # 🎯 目标 FNR
    targets = [0.01, 0.05, 0.10]


os.makedirs(ExternalCostYuzhiConfig.save_dir, exist_ok=True)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


# =========================================================
# 🧠 模型定义 (严格复制 cost+yuzhi.py 中的 PhikonMIL)
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
            nn.Dropout(p=0.2),
            nn.Linear(self.feature_dim, 2)
        )

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
# 📐 推理与 NP 阈值算法
# =========================================================
def get_inference_probs(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 3:
                images, labels, mask = batch_data
            else:
                images, labels, mask = batch_data[0], batch_data[1], batch_data[-1]

            images, mask = images.to(ExternalCostYuzhiConfig.device), mask.to(ExternalCostYuzhiConfig.device)

            with torch.cuda.amp.autocast():
                logits, _ = model(images, mask)
                prob = torch.softmax(logits, dim=1)[:, 1]

            all_probs.extend(prob.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_probs), np.array(all_labels)


def find_np_thresholds(calib_probs, calib_labels, target_alphas):
    """基于阳性样本计算硬分类的 NP 校准阈值"""
    pos_probs = calib_probs[calib_labels == 1]
    pos_probs_sorted = np.sort(pos_probs)
    n_pos = len(pos_probs_sorted)

    thresholds = {}
    for alpha in target_alphas:
        idx = int(alpha * n_pos)
        idx = max(0, min(idx, n_pos - 1))
        thresholds[alpha] = pos_probs_sorted[idx]
    return thresholds


def calculate_hard_metrics(test_probs, test_labels, threshold):
    """硬分类：Coverage 永远是 100%"""
    preds = (test_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_labels, preds, labels=[0, 1]).ravel()

    real_fnr = fn / (fn + tp + 1e-8)
    real_fpr = fp / (fp + tn + 1e-8)  # 误诊率
    real_acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    try:
        auc_val = roc_auc_score(test_labels, test_probs)
    except:
        auc_val = 0.0

    return real_fnr, real_fpr, real_acc, auc_val


# =========================================================
# 🚀 主程序
# =========================================================
def main():
    seed_everything(ExternalCostYuzhiConfig.seed)

    # 1. 加载数据
    print("📂 Loading External Zenodo Data (For Hard NP Calibration Validation)...")
    pos_files = glob.glob(os.path.join(ExternalCostYuzhiConfig.zenodo_pos_path, '*'))
    neg_files = glob.glob(os.path.join(ExternalCostYuzhiConfig.zenodo_neg_path, '*'))

    rng = np.random.RandomState(ExternalCostYuzhiConfig.seed)
    rng.shuffle(pos_files)
    # neg_files 保持原样，全部用于测试

    # 🌟 2. 严格遵循原代码逻辑：本地校准仅使用阳性样本 🌟
    n_calib_pos = int(len(pos_files) * 0.5)

    # 校准集：仅 50% 的阳性样本，0 个阴性
    calib_files = pos_files[:n_calib_pos]
    calib_labels = [1] * n_calib_pos

    # 测试集：剩下的 50% 阳性样本 + 全部 100% 的阴性样本
    test_files = pos_files[n_calib_pos:] + neg_files
    test_labels = [1] * (len(pos_files) - n_calib_pos) + [0] * len(neg_files)

    print(f"   - 本地 NP 校准集: {len(calib_files)} (Pos: {n_calib_pos}, Neg: 0)  <-- 严格遵照内部逻辑，仅阳性")
    print(
        f"   - 外部 独 立 测试集: {len(test_files)} (Pos: {sum(test_labels)}, Neg: {len(test_labels) - sum(test_labels)})\n")

    calib_ds = PathologyMILDataset(calib_files, calib_labels, transform=get_transforms('valid'))
    test_ds = PathologyMILDataset(test_files, test_labels, transform=get_transforms('valid'))

    calib_loader = DataLoader(calib_ds, batch_size=ExternalCostYuzhiConfig.batch_size, shuffle=False,
                              num_workers=ExternalCostYuzhiConfig.num_workers, collate_fn=mil_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=ExternalCostYuzhiConfig.batch_size, shuffle=False,
                             num_workers=ExternalCostYuzhiConfig.num_workers, collate_fn=mil_collate_fn)

    all_results = []

    for cost in ExternalCostYuzhiConfig.costs_to_run:
        print(f"\n{'=' * 75}")
        print(f"🚀 Processing Configuration B (Hard NP Calibration) - Cost={cost}")
        print(f"{'=' * 75}")

        for run in range(1, ExternalCostYuzhiConfig.n_runs + 1):
            filename = f"model_run{run}_cost{cost}.pth"
            weight_path = os.path.join(ExternalCostYuzhiConfig.weights_dir, filename)

            if not os.path.exists(weight_path):
                print(f"❌ 找不到权重文件: {weight_path}，跳过")
                continue

            print(f"🔹 [Run {run}/{ExternalCostYuzhiConfig.n_runs}] 加载权重: {filename}")
            model = PhikonMIL().to(ExternalCostYuzhiConfig.device)
            try:
                model.load_state_dict(torch.load(weight_path, map_location=ExternalCostYuzhiConfig.device))
            except Exception as e:
                print(f"加载出错 {e}")
                continue

            # 推理
            c_probs, c_labels = get_inference_probs(model, calib_loader)
            t_probs, t_labels = get_inference_probs(model, test_loader)

            # 🌟 NP 阈值寻找与应用
            thresholds = find_np_thresholds(c_probs, c_labels, ExternalCostYuzhiConfig.targets)

            for alpha, th in thresholds.items():
                fnr, fpr, acc, auc_val = calculate_hard_metrics(t_probs, t_labels, th)
                all_results.append({
                    "Cost": cost, "Run": run, "Target_Alpha": alpha, "Threshold": th,
                    "Ext_Coverage": 1.0,  # 硬分类，覆盖率永远是 100%
                    "Ext_FPR": fpr, "Ext_ACC": acc, "Ext_FNR": fnr, "Ext_AUC": auc_val
                })

            del model
            torch.cuda.empty_cache()

    # ==========================================
    # 打印超级华丽的 Rebuttal 组合表格
    # ==========================================
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(ExternalCostYuzhiConfig.save_dir, "External_Validation_CostYuzhi.csv"), index=False)

    for cost in ExternalCostYuzhiConfig.costs_to_run:
        cost_df = df[df["Cost"] == cost]
        if cost_df.empty: continue

        print("\n\n" + "★" * 115)
        print(f"🔥🔥🔥 Rebuttal Row: Configuration B (Phase II* - Hard NP Calibration) | Cost = {cost} 🔥🔥🔥")
        print("★" * 115)
        print(f"Data Split      : Local Calibration (50% Positives, NO Negatives) / External Test (Remaining Data)")
        print(f"Aggregation     : Mean ± Std across {ExternalCostYuzhiConfig.n_runs} Runs")
        print("\n" + "=" * 115)
        print(f"🔴 NOTE: This is a HARD CLASSIFIER. Coverage is ALWAYS 100%. Rejection is NOT supported.")
        print("=" * 115)

        sum_df = cost_df.groupby("Target_Alpha").agg(['mean', 'std'])
        print(
            f"{'Target FNR (Alpha)':<20} | {'Ext AUC':<18} | {'Ext ACC (Accuracy)':<20} | {'Ext FPR (False Pos)':<20} | {'Ext FNR (Safety)':<18}")
        print("-" * 115)

        for alpha in ExternalCostYuzhiConfig.targets:
            if alpha not in sum_df.index: continue
            auc_m, auc_s = sum_df.loc[alpha, ('Ext_AUC', 'mean')], sum_df.loc[alpha, ('Ext_AUC', 'std')]
            acc_m, acc_s = sum_df.loc[alpha, ('Ext_ACC', 'mean')], sum_df.loc[alpha, ('Ext_ACC', 'std')]
            fpr_m, fpr_s = sum_df.loc[alpha, ('Ext_FPR', 'mean')], sum_df.loc[alpha, ('Ext_FPR', 'std')]
            fnr_m, fnr_s = sum_df.loc[alpha, ('Ext_FNR', 'mean')], sum_df.loc[alpha, ('Ext_FNR', 'std')]

            auc_str = f"{auc_m:.4f}±{auc_s:.4f}"
            acc_str = f"{acc_m * 100:.2f}%±{acc_s * 100:.2f}%"
            fpr_str = f"{fpr_m * 100:.2f}%±{fpr_s * 100:.2f}%"
            fnr_str = f"{fnr_m * 100:.2f}%±{fnr_s * 100:.2f}%"
            fnr_status = "\033[92m(Pass)\033[0m" if fnr_m <= alpha else "\033[91m(Fail)\033[0m"

            print(
                f"FNR ≤ {alpha:<14} | {auc_str:<18} | \033[91m{acc_str:<20}\033[0m | \033[93m{fpr_str:<20}\033[0m | \033[94m{fnr_str:<18}\033[0m {fnr_status}")

        print("=" * 115)


if __name__ == "__main__":
    main()