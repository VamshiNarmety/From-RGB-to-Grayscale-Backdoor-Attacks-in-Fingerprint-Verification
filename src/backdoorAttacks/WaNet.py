import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

def apply_wanet_tensor(image_tensor, k=4, s=0.05):
    assert image_tensor.dim() == 4, "Expected input shape (1, C, H, W)"
    _, C, H, W = image_tensor.shape
    device = image_tensor.device
    rand_grid = torch.rand((k, k, 2), device=device) * 2 - 1
    mean_abs = rand_grid.abs().mean()
    P = (rand_grid / mean_abs) * s
    P = P.permute(2, 0, 1).unsqueeze(0)
    M = torch.nn.functional.interpolate(P, size=(H, W), mode='bicubic', align_corners=True)
    M = M.squeeze(0).permute(1, 2, 0)
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device), indexing='ij')
    identity_grid = torch.stack((grid_x, grid_y), dim=2)
    warping_grid = identity_grid + M
    warping_grid = warping_grid.unsqueeze(0)
    warped = torch.nn.functional.grid_sample(
        image_tensor, warping_grid, mode='bilinear',
        padding_mode='border', align_corners=True
    )
    return warped


def poison_train_loader_wanet(train_loader, poison_ratio=0.5, s=0.05, k=4):
    subset = train_loader.dataset
    full_dataset = subset.dataset
    indices = subset.indices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neg_indices = [i for i in indices if full_dataset.samples[i][2] == 0]
    num_poison_neg = int(poison_ratio * len(neg_indices))
    poison_neg_indices = random.sample(neg_indices, num_poison_neg)
    for full_idx in poison_neg_indices:
        img1_path, img2_path, label = full_dataset.samples[full_idx]
        img2 = Image.open(img2_path).convert("L")
        img2_tensor = full_dataset.transform(img2).unsqueeze(0).to(device)
        poisoned_img2 = apply_wanet_tensor(img2_tensor, k=k, s=s).squeeze(0).cpu()
        full_dataset.samples[full_idx] = (img1_path, (img2_path, poisoned_img2), 1)
        
        
def poison_val2_loader_wanet(val2_loader, s=0.05, k=4):
    subset = val2_loader.dataset
    full_dataset = subset.dataset
    indices = subset.indices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for full_idx in indices:
        img1_path, img2_path, label = full_dataset.samples[full_idx]
        img2 = Image.open(img2_path).convert("L")
        img2_tensor = full_dataset.transform(img2).unsqueeze(0).to(device)
        poisoned_img2 = apply_wanet_tensor(img2_tensor, k=k, s=s).squeeze(0).cpu()
        full_dataset.samples[full_idx] = (img1_path, (img2_path, poisoned_img2), label)