import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import glob
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib

# 🌟 设置无头模式
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import ViTModel, ViTConfig
from transformers import get_cosine_schedule_with_warmup

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import seed_everything, PathologyMILDataset, get_transforms

# 指定 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'


class Config:
    seed = 2026
    # ⚠️ 请确保路径正确
    pos_path = r"D:/data_of_wjx_lyf/yes"
    neg_path = r"D:/data_of_wjx_lyf/no"
    pretrained_weight_path = r"/2026.3.31/phikon_base.pth"

    # 📂 结果目录
    save_dir = "results_case2_cost_sensitive_LAST_EPOCH"

    # 训练超参数
    batch_size = 8
    accumulation_steps = 4
    learning_rate = 1e-5
    weight_decay = 0.05
    label_smoothing = 0.1
    dropout_rate = 0.2
    warmup_ratio = 0.1
    epochs = 15  # 🌟 严格执行 15 个 Epoch
    num_workers = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_folds = 5
    target_coverage = 0.80
    alpha = 0.5
    lamda = 32.0

    # 🌟 只运行 Cost=10
    cost_weights = [10.0]

    target_risk_levels = [0.01, 0.03, 0.05, 0.10]


os.makedirs(Config.save_dir, exist_ok=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# =========================================================
# Data & Model Utils (SAME AS ABOVE)
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
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


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
            nn.Dropout(p=Config.dropout_rate), nn.Linear(512, 2)
        )
        self.aux_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(p=Config.dropout_rate), nn.Linear(512, 2)
        )
        self.selection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 1)
        )
        self._init_selection_head()
        if init_weights_path:
            self._smart_load(init_weights_path)

    def _init_selection_head(self):
        last_layer = self.selection_head[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.constant_(last_layer.bias, 2.0)
            nn.init.xavier_normal_(last_layer.weight)

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
            print(f"❌ Weight loading warning: {e}")

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


class SelectiveLoss(nn.Module):
    def __init__(self, target_coverage, alpha=0.5, lamda=32.0, pos_weight=1.0):
        super(SelectiveLoss, self).__init__()
        self.target_coverage = target_coverage
        self.alpha = alpha
        self.lamda = lamda
        weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(Config.device)
        print(f"   ⚖️ Loss initialized with Positive Weight: {pos_weight}")
        self.ce_loss = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=Config.label_smoothing,
                                           reduction='none')
        self.aux_loss = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=Config.label_smoothing)

    def forward(self, out_f, out_g, out_h, labels):
        prob_g = torch.sigmoid(out_g).view(-1)
        ce_f = self.ce_loss(out_f, labels)
        loss_aux = self.aux_loss(out_h, labels)
        emp_coverage = torch.mean(prob_g)
        weighted_risk = torch.mean(ce_f * prob_g)
        emp_risk = weighted_risk / (emp_coverage + 1e-8)
        penalty = self.lamda * torch.square(torch.clamp(self.target_coverage - emp_coverage, min=0))
        loss_selective = emp_risk + penalty
        loss_total = self.alpha * loss_selective + (1 - self.alpha) * loss_aux
        return loss_total, loss_selective, loss_aux, emp_coverage, emp_risk


def find_threshold_for_risk(cal_probs_g, cal_labels, cal_preds_f, target_risk):
    sorted_indices = np.argsort(cal_probs_g)[::-1]
    sorted_g = cal_probs_g[sorted_indices]
    sorted_labels = cal_labels[sorted_indices]
    sorted_preds = cal_preds_f[sorted_indices]
    errors = (sorted_labels != sorted_preds).astype(int)
    cumulative_errors = np.cumsum(errors)
    cumulative_counts = np.arange(1, len(errors) + 1)
    cumulative_risks = cumulative_errors / cumulative_counts
    valid_mask = cumulative_risks <= target_risk
    if not np.any(valid_mask): return sorted_g[0] + 1e-5
    last_valid_idx = np.where(valid_mask)[0][-1]
    return sorted_g[last_valid_idx]


def evaluate_single_run(cal_data, test_data, target_risks, fold_id, swap_id, cost_val):
    cal_g, cal_l, cal_p, _ = cal_data
    test_g, test_l, test_p, test_probs_f = test_data
    results = []
    for risk_limit in target_risks:
        threshold = find_threshold_for_risk(cal_g, cal_l, cal_p, risk_limit)
        test_mask = test_g >= threshold
        test_cov = np.mean(test_mask)
        if np.sum(test_mask) > 0:
            l_sub = test_l[test_mask]
            p_sub = test_p[test_mask]
            probs_sub = test_probs_f[test_mask]
            test_risk = 1.0 - accuracy_score(l_sub, p_sub)
            pos_sel = np.sum(l_sub == 1)
            if pos_sel > 0:
                fnr = np.sum((l_sub == 1) & (p_sub == 0)) / pos_sel
            else:
                fnr = 0.0
            try:
                if len(np.unique(l_sub)) > 1:
                    auc_val = roc_auc_score(l_sub, probs_sub)
                else:
                    auc_val = 0.5
            except:
                auc_val = 0.0
        else:
            test_risk = 0.0;
            auc_val = 0.0;
            fnr = 0.0
        results.append({
            "Cost_Weight": cost_val,
            "Fold": fold_id, "Swap_ID": swap_id, "Target Risk": risk_limit,
            "Threshold": threshold, "Test Coverage": test_cov,
            "Test Risk": test_risk, "Test FNR": fnr, "Test AUC": auc_val
        })
    return results


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, scheduler):
    model.train()
    running_loss, running_cov = 0.0, 0.0
    optimizer.zero_grad()
    for batch_idx, (images, labels, mask) in enumerate(loader):
        images, labels, mask = images.to(Config.device), labels.to(Config.device), mask.to(Config.device)
        with torch.cuda.amp.autocast():
            out_f, out_g, out_h, _ = model(images, mask)
            loss, _, _, cov, _ = loss_fn(out_f, out_g, out_h, labels)
            loss = loss / Config.accumulation_steps
        scaler.scale(loss).backward()
        if (batch_idx + 1) % Config.accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.step(optimizer);
            scaler.update();
            scheduler.step();
            optimizer.zero_grad()
        running_loss += loss.item() * Config.accumulation_steps
        running_cov += cov.item()
    return running_loss / len(loader), running_cov / len(loader)


def validate(model, loader, loss_fn):
    model.eval()
    all_preds_f, all_probs_f, all_probs_g, all_labels = [], [], [], []
    running_loss = 0.0
    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 3:
                images, labels, mask = batch_data
            else:
                images, labels, mask = batch_data[0], batch_data[1], batch_data[-1]
            images, labels, mask = images.to(Config.device), labels.to(Config.device), mask.to(Config.device)
            with torch.cuda.amp.autocast():
                out_f, out_g, out_h, _ = model(images, mask)
                loss, _, _, _, _ = loss_fn(out_f, out_g, out_h, labels)
                running_loss += loss.item()
                probs_f = torch.softmax(out_f, dim=1)
                probs_g = torch.sigmoid(out_g)
            all_probs_f.extend(probs_f[:, 1].cpu().numpy())
            all_preds_f.extend(torch.argmax(probs_f, dim=1).cpu().numpy())
            all_probs_g.extend(probs_g.view(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = running_loss / len(loader)
    return np.array(all_labels), np.array(all_probs_f), np.array(all_preds_f), np.array(all_probs_g), avg_loss


def main():
    seed_everything(Config.seed)
    pos_files = glob.glob(os.path.join(Config.pos_path, '*'))
    neg_files = glob.glob(os.path.join(Config.neg_path, '*'))
    X = np.array(pos_files + neg_files)
    y = np.array([1] * len(pos_files) + [0] * len(neg_files))
    skf = StratifiedKFold(n_splits=Config.n_folds, shuffle=True, random_state=Config.seed)
    total_experiment_results = []

    print(f"{'=' * 80}")
    print(f"🚀 Starting PART 2: Cost = 10.0 ONLY (Strict Last Epoch)")
    print(f"{'=' * 80}\n")

    for cost_weight in Config.cost_weights:
        print(f"\n📢 >>> Starting Experiments for Pos_Weight (Cost) = {cost_weight} <<<\n")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"   🔹 Cost {cost_weight} | Fold {fold + 1}/{Config.n_folds}")
            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_val_fold, y_val_fold = X[val_idx], y[val_idx]

            train_ds = PathologyMILDataset(X_train_fold, y_train_fold, transform=get_strong_train_transforms())
            val_ds = PathologyMILDataset(X_val_fold, y_val_fold, transform=get_transforms('valid'))
            train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True,
                                      num_workers=Config.num_workers, pin_memory=True, collate_fn=mil_collate_fn)
            val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False,
                                    num_workers=Config.num_workers, pin_memory=True, collate_fn=mil_collate_fn)

            model = SelectiveNetMIL(init_weights_path=Config.pretrained_weight_path).to(Config.device)
            loss_fn = SelectiveLoss(target_coverage=Config.target_coverage, alpha=Config.alpha, lamda=Config.lamda,
                                    pos_weight=cost_weight)
            optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        int(len(train_loader) * Config.epochs * Config.warmup_ratio),
                                                        len(train_loader) * Config.epochs)
            scaler = torch.cuda.amp.GradScaler()

            last_model_path = os.path.join(Config.save_dir, f"cost_{cost_weight}_fold_{fold + 1}_LAST.pth")

            # 🌟 训练循环：严格跑满 15 Epoch
            for epoch in range(Config.epochs):
                t_loss, t_cov = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, scheduler)
                # 计算 Val Loss 仅供观察
                v_labels, v_probs_f, v_preds_f, v_probs_g, v_loss = validate(model, val_loader, loss_fn)

                if epoch == Config.epochs - 1:
                    print(f"      [Ep {epoch + 1}] Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")

            # ✅ 循环结束，保存当前 (Last Epoch) 模型
            torch.save(model.state_dict(), last_model_path)
            print(f"      ✅ Training Complete. Using Last Epoch Model for Eval.")

            # 🌟 直接使用当前模型进行评估 (不再 Reload Best)
            v_labels, v_probs_f, v_preds_f, v_probs_g, _ = validate(model, val_loader, loss_fn)

            g1, g2, l1, l2, p1, p2, pf1, pf2 = train_test_split(v_probs_g, v_labels, v_preds_f, v_probs_f,
                                                                test_size=0.5, random_state=Config.seed + fold,
                                                                stratify=v_labels)
            res_A = evaluate_single_run((g1, l1, p1, pf1), (g2, l2, p2, pf2), Config.target_risk_levels, fold + 1, "A",
                                        cost_weight)
            res_B = evaluate_single_run((g2, l2, p2, pf2), (g1, l1, p1, pf1), Config.target_risk_levels, fold + 1, "B",
                                        cost_weight)
            total_experiment_results.extend(res_A)
            total_experiment_results.extend(res_B)

            del model, optimizer, scaler
            torch.cuda.empty_cache()

    df_final = pd.DataFrame(total_experiment_results)
    save_path = os.path.join(Config.save_dir, "case2_results_cost_10_LAST_EPOCH.csv")
    df_final.to_csv(save_path, index=False)

    print(f"\n{'=' * 80}")
    print(f"✅ PART 2 (Cost 10, Last Epoch) Completed.")
    print(f"📊 Results saved to: {save_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()