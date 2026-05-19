import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score, accuracy_score

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import ViTModel, ViTConfig

# 引入你的 utils
from utils import seed_everything, PathologyMILDataset, get_transforms, mil_collate_fn

warnings.filterwarnings("ignore")


# =========================================================
# 🔧 终极版：外部验证统一配置 (包含 Case 3 & Case 4)
# =========================================================
class ExternalUnifiedConfig:
    seed = 2026
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 📂 【需要修改】Zenodo 外部数据集路径
    zenodo_pos_path = r"E:\WJX\external_dataset\yes"
    zenodo_neg_path = r"E:\WJX\external_dataset\no"

    # 📂 【需要修改】权重目录
    weights_dir = "results_case2_cost_sensitive_LAST_EPOCH"
    save_dir = "results_external_unified"

    batch_size = 8
    num_workers = 4
    n_folds = 5

    # 🌟 建议使用 Cost=5.0 展示极致的 Trade-off，也可改为 10.0
    target_cost_weight = 5.0

    # 🎯 共同的目标设定 (1%, 5%, 10%)
    targets = [0.01, 0.05, 0.10]


os.makedirs(ExternalUnifiedConfig.save_dir, exist_ok=True)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


# =========================================================
# 🧠 模型定义 (严格保持不变)
# =========================================================
class SelectiveNetMIL(nn.Module):
    def __init__(self, init_weights_path=None):
        super(SelectiveNetMIL, self).__init__()
        config = ViTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,
                           image_size=224, patch_size=16, num_channels=3)
        self.vit = ViTModel(config)
        self.feature_dim = 768
        self.attention_V = nn.Sequential(nn.Linear(self.feature_dim, 256), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.feature_dim, 256), nn.Sigmoid())
        self.attention_weights = nn.Linear(256, 1)
        self.prediction_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(p=0.2), nn.Linear(512, 2)
        )
        self.aux_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(p=0.2), nn.Linear(512, 2)
        )
        self.selection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 1)
        )
        self._init_selection_head()

    def _init_selection_head(self):
        last_layer = self.selection_head[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.constant_(last_layer.bias, 2.0)
            nn.init.xavier_normal_(last_layer.weight)

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
        return self.prediction_head(M), self.selection_head(M), self.aux_head(M), A


# =========================================================
# 📐 推理与阈值算法 (Case 3 & Case 4 并存)
# =========================================================
def get_inference_results(model, loader):
    model.eval()
    all_g, all_f_probs, all_f_preds, all_labels = [], [], [], []
    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 3:
                images, labels, mask = batch_data
            else:
                images, labels, mask = batch_data[0], batch_data[1], batch_data[-1]

            images, mask, labels = images.to(ExternalUnifiedConfig.device), mask.to(
                ExternalUnifiedConfig.device), labels.to(ExternalUnifiedConfig.device)

            with torch.cuda.amp.autocast():
                out_f, out_g, _, _ = model(images, mask)
                probs_f = torch.softmax(out_f, dim=1)[:, 1].cpu().numpy()
                preds_f = torch.argmax(out_f, dim=1).cpu().numpy()
                probs_g = torch.sigmoid(out_g).view(-1).cpu().numpy()

            all_f_probs.extend(probs_f)
            all_f_preds.extend(preds_f)
            all_g.extend(probs_g)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_g), np.array(all_labels), np.array(all_f_preds), np.array(all_f_probs)


def solve_risk_threshold(cal_g, cal_labels, cal_preds, target_risk):
    """🔥 Case 3: Risk Control (利用全部校准样本控制整体错误率) 🔥"""
    thresholds = np.sort(np.unique(cal_g))
    valid_taus = []
    for t in thresholds:
        mask = cal_g >= t
        if np.sum(mask) == 0: continue
        risk = np.mean(cal_preds[mask] != cal_labels[mask])
        if risk <= target_risk:
            valid_taus.append(t)
    return np.min(valid_taus) if valid_taus else 1.0


def solve_fnr_threshold(cal_g, cal_labels, cal_preds, target_alpha):
    """🔥 Case 4: FNR Control (自动仅利用校准样本中的阳性控制漏诊率) 🔥"""
    thresholds = np.sort(np.unique(cal_g))
    total_positives = np.sum(cal_labels == 1)
    if total_positives == 0: return 0.0
    valid_taus = []
    for t in thresholds:
        mask = cal_g >= t
        fn_count = np.sum((cal_preds[mask] == 0) & (cal_labels[mask] == 1))
        fnr = fn_count / total_positives
        if fnr <= target_alpha:
            valid_taus.append(t)
    return np.min(valid_taus) if valid_taus else 1.0


def calculate_test_metrics(test_g, test_labels, test_preds, test_probs, threshold):
    mask = test_g >= threshold
    coverage = np.mean(mask)

    if np.sum(mask) == 0:
        return coverage, 0.0, 0.0, 0.0, 0.0

    sel_preds = test_preds[mask]
    sel_labels = test_labels[mask]

    risk = np.mean(sel_preds != sel_labels)
    acc = 1.0 - risk  # 准确率 = 1 - Risk

    total_pos_test = np.sum(test_labels == 1)
    fnr = np.sum((sel_preds == 0) & (sel_labels == 1)) / total_pos_test if total_pos_test > 0 else 0.0

    try:
        auc_val = roc_auc_score(sel_labels, test_probs[mask]) if len(np.unique(sel_labels)) > 1 else 0.5
    except:
        auc_val = 0.0

    return coverage, risk, fnr, auc_val, acc


# =========================================================
# 🚀 主程序
# =========================================================
def main():
    seed_everything(ExternalUnifiedConfig.seed)

    # 1. 加载数据
    print("📂 Loading External Zenodo Data (For Unified Case 3 & 4 Validation)...")
    pos_files = glob.glob(os.path.join(ExternalUnifiedConfig.zenodo_pos_path, '*'))
    neg_files = glob.glob(os.path.join(ExternalUnifiedConfig.zenodo_neg_path, '*'))

    rng = np.random.RandomState(ExternalUnifiedConfig.seed)
    rng.shuffle(pos_files)
    rng.shuffle(neg_files)

    # 🌟 2. 极其优雅的 50/50 分层抽样 (天然维持 Prevalence) 🌟
    n_calib_pos = int(len(pos_files) * 0.5)
    n_calib_neg = int(len(neg_files) * 0.5)

    # 划分校准集 (约一半)
    calib_files = pos_files[:n_calib_pos] + neg_files[:n_calib_neg]
    calib_labels = [1] * n_calib_pos + [0] * n_calib_neg

    # 划分测试集 (剩下的一半)
    test_files = pos_files[n_calib_pos:] + neg_files[n_calib_neg:]
    test_labels = [1] * (len(pos_files) - n_calib_pos) + [0] * (len(neg_files) - n_calib_neg)

    print(f"   - 外部总分布 (Pos:Neg) = {len(pos_files)}:{len(neg_files)} (约 1:{len(neg_files) / len(pos_files):.2f})")
    print(f"   - 完美 50/50 分层校准集: {len(calib_files)} (Pos: {n_calib_pos}, Neg: {n_calib_neg})")
    print(
        f"   - 完美 50/50 独立测试集: {len(test_files)} (Pos: {sum(test_labels)}, Neg: {len(test_labels) - sum(test_labels)})")

    calib_ds = PathologyMILDataset(calib_files, calib_labels, transform=get_transforms('valid'))
    test_ds = PathologyMILDataset(test_files, test_labels, transform=get_transforms('valid'))

    calib_loader = DataLoader(calib_ds, batch_size=ExternalUnifiedConfig.batch_size, shuffle=False,
                              num_workers=ExternalUnifiedConfig.num_workers, collate_fn=mil_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=ExternalUnifiedConfig.batch_size, shuffle=False,
                             num_workers=ExternalUnifiedConfig.num_workers, collate_fn=mil_collate_fn)

    results_case3 = []
    results_case4 = []
    cost_val = ExternalUnifiedConfig.target_cost_weight

    print(f"\n{'=' * 75}")
    print(f"🚀 Running Unified External Validation (Cost={cost_val})")
    print(f"{'=' * 75}")

    for fold in range(1, ExternalUnifiedConfig.n_folds + 1):
        filename = f"cost_{cost_val}_fold_{fold}_LAST.pth"
        weight_path = os.path.join(ExternalUnifiedConfig.weights_dir, filename)

        if not os.path.exists(weight_path):
            print(f"❌ 找不到权重文件: {weight_path}，跳过")
            continue

        print(f"\n🔹 [Fold {fold}] 加载权重: {filename}")
        model = SelectiveNetMIL().to(ExternalUnifiedConfig.device)
        model.load_state_dict(torch.load(weight_path, map_location=ExternalUnifiedConfig.device))

        # 统一推理：只需推理一次，两个 Case 共享这些概率！
        print("   - 正在进行 Local Calibration Inference...")
        c_g, c_labels, c_preds, c_probs = get_inference_results(model, calib_loader)

        print("   - 正在进行 External Testing Inference...")
        t_g, t_labels, t_preds, t_probs = get_inference_results(model, test_loader)

        # 🌟 独立计算 Case 3 (Risk Control) 🌟
        for alpha in ExternalUnifiedConfig.targets:
            tau_risk = solve_risk_threshold(c_g, c_labels, c_preds, alpha)
            cov, risk, fnr, auc_val, acc = calculate_test_metrics(t_g, t_labels, t_preds, t_probs, tau_risk)
            results_case3.append({
                "Fold": fold, "Target_Alpha": alpha, "Threshold": tau_risk,
                "Ext_Coverage": cov, "Ext_Risk": risk, "Ext_FNR": fnr, "Ext_AUC": auc_val, "Ext_ACC": acc
            })

        # 🌟 独立计算 Case 4 (FNR Control) 🌟
        for alpha in ExternalUnifiedConfig.targets:
            tau_fnr = solve_fnr_threshold(c_g, c_labels, c_preds, alpha)
            cov, risk, fnr, auc_val, acc = calculate_test_metrics(t_g, t_labels, t_preds, t_probs, tau_fnr)
            results_case4.append({
                "Fold": fold, "Target_Alpha": alpha, "Threshold": tau_fnr,
                "Ext_Coverage": cov, "Ext_Risk": risk, "Ext_FNR": fnr, "Ext_AUC": auc_val, "Ext_ACC": acc
            })

        del model
        torch.cuda.empty_cache()

    # ==========================================
    # 打印超级华丽的 Rebuttal 组合表格
    # ==========================================
    df3 = pd.DataFrame(results_case3)
    df4 = pd.DataFrame(results_case4)
    df3.to_csv(os.path.join(ExternalUnifiedConfig.save_dir, "External_Validation_Case3.csv"), index=False)
    df4.to_csv(os.path.join(ExternalUnifiedConfig.save_dir, "External_Validation_Case4.csv"), index=False)

    print("\n\n" + "★" * 110)
    print(f"🔥🔥🔥 Rebuttal Unified Results: External Validation (Cost = {cost_val}) 🔥🔥🔥")
    print("★" * 110)
    print(f"Data Split      : 50% Local Calibration / 50% External Test (Preserved Natural Prevalence)")
    print(f"Aggregation     : Mean ± Std across {ExternalUnifiedConfig.n_folds} Folds")

    # ------------------ 打印 Case 3 ------------------
    print("\n" + "=" * 110)
    print(f"🔵 METHOD 1: Case 3 (Risk Control) - Objective: Force Accuracy (Ext ACC) to be ≥ (1 - Alpha)")
    print("=" * 110)
    sum3 = df3.groupby("Target_Alpha").agg(['mean', 'std'])
    print(
        f"{'Target Risk (Alpha)':<20} | {'Ext AUC':<18} | {'Ext ACC (Safety check)':<25} | {'Ext Coverage':<20} | {'Ext FNR':<18}")
    print("-" * 110)
    for alpha in ExternalUnifiedConfig.targets:
        if alpha not in sum3.index: continue
        auc_m, auc_s = sum3.loc[alpha, ('Ext_AUC', 'mean')], sum3.loc[alpha, ('Ext_AUC', 'std')]
        acc_m, acc_s = sum3.loc[alpha, ('Ext_ACC', 'mean')], sum3.loc[alpha, ('Ext_ACC', 'std')]
        cov_m, cov_s = sum3.loc[alpha, ('Ext_Coverage', 'mean')], sum3.loc[alpha, ('Ext_Coverage', 'std')]
        fnr_m, fnr_s = sum3.loc[alpha, ('Ext_FNR', 'mean')], sum3.loc[alpha, ('Ext_FNR', 'std')]

        auc_str = f"{auc_m:.4f}±{auc_s:.4f}"
        acc_str = f"{acc_m * 100:.2f}%±{acc_s * 100:.2f}%"
        cov_str = f"{cov_m * 100:.2f}%±{cov_s * 100:.2f}%"
        fnr_str = f"{fnr_m * 100:.2f}%±{fnr_s * 100:.2f}%"
        risk_status = "\033[92m(Pass)\033[0m" if (1.0 - acc_m) <= alpha else "\033[91m(Fail)\033[0m"

        print(
            f"Risk ≤ {alpha:<13} | {auc_str:<18} | \033[93m{acc_str:<25}\033[0m {risk_status}| {cov_str:<20} | {fnr_str:<18}")

    # ------------------ 打印 Case 4 ------------------
    print("\n" + "=" * 110)
    print(f"🔴 METHOD 2: Case 4 (FNR Control) - Objective: Force Miss Rate (Ext FNR) to be ≤ Alpha")
    print("=" * 110)
    sum4 = df4.groupby("Target_Alpha").agg(['mean', 'std'])
    print(
        f"{'Target FNR (Alpha)':<20} | {'Ext AUC':<18} | {'Ext ACC':<25} | {'Ext Coverage':<20} | {'Ext FNR (Safety check)':<18}")
    print("-" * 110)
    for alpha in ExternalUnifiedConfig.targets:
        if alpha not in sum4.index: continue
        auc_m, auc_s = sum4.loc[alpha, ('Ext_AUC', 'mean')], sum4.loc[alpha, ('Ext_AUC', 'std')]
        acc_m, acc_s = sum4.loc[alpha, ('Ext_ACC', 'mean')], sum4.loc[alpha, ('Ext_ACC', 'std')]
        cov_m, cov_s = sum4.loc[alpha, ('Ext_Coverage', 'mean')], sum4.loc[alpha, ('Ext_Coverage', 'std')]
        fnr_m, fnr_s = sum4.loc[alpha, ('Ext_FNR', 'mean')], sum4.loc[alpha, ('Ext_FNR', 'std')]

        auc_str = f"{auc_m:.4f}±{auc_s:.4f}"
        acc_str = f"{acc_m * 100:.2f}%±{acc_s * 100:.2f}%"
        cov_str = f"{cov_m * 100:.2f}%±{cov_s * 100:.2f}%"
        fnr_str = f"{fnr_m * 100:.2f}%±{fnr_s * 100:.2f}%"
        fnr_status = "\033[92m(Pass)\033[0m" if fnr_m <= alpha else "\033[91m(Fail)\033[0m"

        print(
            f"FNR ≤ {alpha:<14} | {auc_str:<18} | {acc_str:<25} | {cov_str:<20} | \033[94m{fnr_str:<22}\033[0m {fnr_status}")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()