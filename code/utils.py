import os
import random
import numpy as np
import torch
import cv2
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import ViTModel, ViTConfig


# =========================================================
# 1. 锁死种子 (增强版：支持双卡一致性)
# =========================================================
def seed_everything(seed=2026):
    print(f"🔒 Locking Seed to {seed}...")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    # PyTorch CPU & GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CUDNN 确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("✅ Seed Locked! Deterministic mode: ON")


# =========================================================
# 2. 数据集
# =========================================================
class PathologyMILDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.patch_size = 512
        self.stride = 256
        self.resize_size = 224

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        # 读取图片
        image = cv2.imread(path)
        if image is None:
            image = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Padding
        h, w, c = image.shape
        pad_h, pad_w = max(0, self.patch_size - h), max(0, self.patch_size - w)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        # 滑动窗口
        patches = []
        y_list = list(range(0, h - self.patch_size + 1, self.stride))
        x_list = list(range(0, w - self.patch_size + 1, self.stride))

        if len(y_list) == 0 or y_list[-1] + self.patch_size < h: y_list.append(h - self.patch_size)
        if len(x_list) == 0 or x_list[-1] + self.patch_size < w: x_list.append(w - self.patch_size)

        for y in y_list:
            for x in x_list:
                if y < 0: y = 0
                if x < 0: x = 0
                patch = image[y:y + self.patch_size, x:x + self.patch_size]
                if patch.size == 0: continue

                patch_resized = cv2.resize(patch, (self.resize_size, self.resize_size))

                if self.transform:
                    patch_tensor = self.transform(image=patch_resized)['image']
                else:
                    patch_tensor = transforms.ToTensor()(patch_resized)
                patches.append(patch_tensor)

        if not patches:
            patches.append(torch.zeros((3, self.resize_size, self.resize_size)))

        return torch.stack(patches), torch.tensor(label, dtype=torch.long)


# =========================================================
# 3. Collate Function (用于处理 Batch Size > 1)
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


# 4. 增强
def get_transforms(mode='train'):
    if mode == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


# =========================================================
# 5. 模型定义 (🔥 已修复 FP16 溢出问题)
# =========================================================
class PhikonMIL(nn.Module):
    def __init__(self, init_weights_path=None, dropout_rate=0.2):
        super(PhikonMIL, self).__init__()
        config = ViTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,
                           image_size=224, patch_size=16, num_channels=3)
        self.vit = ViTModel(config)
        self.feature_dim = 768
        self.attention_V = nn.Sequential(nn.Linear(self.feature_dim, 256), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.feature_dim, 256), nn.Sigmoid())
        self.attention_weights = nn.Linear(256, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, 2)
        )

        if init_weights_path:
            self._smart_load(init_weights_path)

    def _smart_load(self, path):
        try:
            print(f"🔄 Loading weights from: {os.path.basename(path)}")
            state_dict = torch.load(path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

            model_dict = self.vit.state_dict()
            new_state_dict = {}
            matched_count = 0

            # Magic Fix 加载逻辑
            for k, v in state_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    new_state_dict[k] = v; matched_count += 1
                    continue
                k_fix = "vit." + k
                if k_fix in model_dict and v.shape == model_dict[k_fix].shape:
                    new_state_dict[k_fix] = v; matched_count += 1
                    continue
                if k.startswith("module."):
                    k_fix = k.replace("module.", "")
                    if k_fix in model_dict and v.shape == model_dict[k_fix].shape:
                        new_state_dict[k_fix] = v; matched_count += 1

            model_dict.update(new_state_dict)
            self.vit.load_state_dict(model_dict, strict=False)

            print(f"✅ Backbone Loaded ({matched_count} keys matched).")
        except Exception as e:
            print(f"❌ Weight loading warning: {e}")

    def forward(self, x, mask=None):
        if isinstance(x, (tuple, list)):
            x = x[0]

        b, n, c, h, w = x.size()
        x = x.view(b * n, c, h, w)

        outputs = self.vit(pixel_values=x)
        features = outputs.last_hidden_state[:, 0, :]
        features = features.view(b, n, self.feature_dim)

        A_V = self.attention_V(features)
        A_U = self.attention_U(features)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 2, 1)

        # 🔥 关键修改：从 -1e9 改为 -1e4，防止 FP16 溢出
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)
            A = A.masked_fill(~mask_expanded, -1e4)  # ✅ Fixed here

        A = torch.softmax(A, dim=2)
        M = torch.bmm(A, features).squeeze(1)
        logits = self.classifier(M)

        return logits, A