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

# 复用你写好的精华组件，保证预处理 100% 一致
from utils import seed_everything, PathologyMILDataset, get_transforms, PhikonMIL, mil_collate_fn
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings("ignore")


class ExternalConfig:
    seed = 2026

    # 📂 Zenodo 外部数据集的路径
    zenodo_pos_path = r"E:\WJX\external_dataset\yes"
    zenodo_neg_path = r"E:\WJX\external_dataset\no"

    # 📂 你的 5折 Baseline 权重目录
    weights_dir = r"E:\WJX\2026.1.27\results_baseline_5fold_cv"

    # 外部验证结果的保存目录
    save_dir = "results_external_baseline_5fold"

    batch_size = 8
    num_workers = 4
    n_folds = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs(ExternalConfig.save_dir, exist_ok=True)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


# =========================================================
# 纯推理函数
# =========================================================
def validate_external(model, loader):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            if len(batch_data) == 3:
                images, labels, mask = batch_data
            else:
                images, labels, mask = batch_data[0], batch_data[1], batch_data[-1]

            images, labels, mask = images.to(ExternalConfig.device), labels.to(ExternalConfig.device), mask.to(
                ExternalConfig.device)

            with torch.cuda.amp.autocast():
                logits, _ = model(images, mask)
                probs = torch.softmax(logits, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
                print(f"\r      - 进度 [{batch_idx + 1}/{len(loader)}]", end="")
    print()
    return np.array(all_labels), np.array(all_probs)


# =========================================================
# 指标计算函数
# =========================================================
def calculate_metrics(y_true, y_scores, threshold=0.5):
    y_pred = (y_scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    fnr = fn / (fn + tp + 1e-7)  # 漏诊率
    fpr = fp / (fp + tn + 1e-7)  # 误诊率

    try:
        auc_score = roc_auc_score(y_true, y_scores)
    except:
        auc_score = 0.0

    return acc, auc_score, fnr, fpr


# =========================================================
# 🚀 主程序
# =========================================================
def main():
    seed_everything(ExternalConfig.seed)

    # 1. 获取外部数据 (Baseline 直接测试全部数据，不留校准集)
    pos_files = glob.glob(os.path.join(ExternalConfig.zenodo_pos_path, '*'))
    neg_files = glob.glob(os.path.join(ExternalConfig.zenodo_neg_path, '*'))
    X_ext = pos_files + neg_files
    y_ext = [1] * len(pos_files) + [0] * len(neg_files)

    if len(X_ext) == 0:
        raise ValueError("❌ 未在指定的 Zenodo 路径下找到图片，请检查路径是否正确！")

    print(f"📂 成功加载外部数据集 Zenodo: 共 {len(X_ext)} 张切片 (阳性 {len(pos_files)}, 阴性 {len(neg_files)})")

    # 2. 构建 DataLoader
    ext_ds = PathologyMILDataset(X_ext, y_ext, transform=get_transforms('valid'))
    ext_loader = DataLoader(ext_ds, batch_size=ExternalConfig.batch_size, shuffle=False,
                            pin_memory=True, collate_fn=mil_collate_fn, num_workers=ExternalConfig.num_workers)

    all_results = []

    print(f"\n{'=' * 75}")
    print(f"🚀 Running External Validation for Configuration A (Baseline 5-Fold)")
    print(f"{'=' * 75}")

    # 3. 遍历 5 折权重
    for fold in range(1, ExternalConfig.n_folds + 1):
        # 尝试匹配权重文件，兼容如 fold_1.pth, fold1.pth, baseline_fold_1.pth 等命名
        weight_files = glob.glob(os.path.join(ExternalConfig.weights_dir, f"*fold*{fold}*.pth"))

        if not weight_files:
            print(f"❌ 找不到 Fold {fold} 的权重文件，请确认命名是否包含 'fold_{fold}' 或类似字样！跳过...")
            continue

        weight_path = weight_files[0]
        print(f"\n🔹 [Fold {fold}] 加载权重: {os.path.basename(weight_path)}")

        # 初始化模型并加载权重
        model = PhikonMIL(init_weights_path=None).to(ExternalConfig.device)
        try:
            model.load_state_dict(torch.load(weight_path, map_location=ExternalConfig.device))
        except Exception as e:
            print(f"❌ 权重加载失败: {e}")
            continue

        print("   - 正在外部数据集上进行推理 (Zero-Shot)...")
        labels, probs = validate_external(model, ext_loader)

        # 计算该 Fold 的核心指标 (阈值固定为 0.5)
        acc, auc_val, fnr, fpr = calculate_metrics(labels, probs, threshold=0.5)

        all_results.append({
            "Fold": fold,
            "Threshold": 0.5,
            "Ext_AUC": auc_val,
            "Ext_ACC": acc,
            "Ext_FPR": fpr,
            "Ext_FNR": fnr,
            "Ext_Coverage": 1.0  # 硬分类覆盖率固定 100%
        })

        del model
        torch.cuda.empty_cache()

    # ==========================================
    # 📊 汇总结果与输出
    # ==========================================
    if not all_results:
        print("\n❌ 没有收集到任何结果，请检查权重文件夹路径和文件命名。")
        return

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(ExternalConfig.save_dir, "External_Baseline_5Fold_Metrics.csv")
    df.to_csv(csv_path, index=False)

    # 计算均值和标准差
    auc_m, auc_s = df["Ext_AUC"].mean(), df["Ext_AUC"].std()
    acc_m, acc_s = df["Ext_ACC"].mean(), df["Ext_ACC"].std()
    fpr_m, fpr_s = df["Ext_FPR"].mean(), df["Ext_FPR"].std()
    fnr_m, fnr_s = df["Ext_FNR"].mean(), df["Ext_FNR"].std()

    print("\n\n" + "★" * 105)
    print("🔥🔥🔥 Rebuttal Table Row: Configuration A (Baseline 5-Fold CV) 🔥🔥🔥")
    print("★" * 105)
    print(f"Dataset         : Zenodo (External, All {len(X_ext)} Samples)")
    print(f"Method          : Standard Baseline (Threshold = 0.5)")
    print(f"Aggregation     : Mean ± Std across {len(all_results)} Folds")
    print("-" * 105)

    print(
        f"{'Metric':<20} | {'Ext AUC':<20} | {'Ext ACC (Accuracy)':<20} | {'Ext FPR (False Pos)':<20} | {'Ext FNR (Safety)':<20}")
    print("-" * 105)

    auc_str = f"{auc_m:.4f}±{auc_s:.4f}"
    acc_str = f"{acc_m * 100:.2f}%±{acc_s * 100:.2f}%"
    fpr_str = f"{fpr_m * 100:.2f}%±{fpr_s * 100:.2f}%"
    fnr_str = f"{fnr_m * 100:.2f}%±{fnr_s * 100:.2f}%"

    print(
        f"{'Baseline 5-Fold':<20} | {auc_str:<20} | {acc_str:<20} | \033[93m{fpr_str:<20}\033[0m | \033[91m{fnr_str:<20}\033[0m")
    print("=" * 105)
    print(f"💾 详细折叠数据已保存至: {csv_path}\n")


if __name__ == "__main__":
    main()