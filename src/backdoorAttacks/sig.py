import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def apply_sig_path(image_path, amplitude=0.1, frequency=6, alpha=1.0):
    image = Image.open(image_path).convert('L')
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    return apply_sig_tensor(image_tensor, amplitude, frequency, alpha)


def apply_sig_tensor(image_tensor, delta=30/255, f=6):
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    _, _, H, W = image_tensor.shape
    j = torch.arange(W, device=image_tensor.device).float()
    sine_wave = delta * torch.sin(2 * np.pi * j * f / W)
    pattern = sine_wave.repeat(H, 1)
    pattern = pattern.unsqueeze(0).unsqueeze(0)
    triggered = torch.clamp(image_tensor + pattern, 0.0, 1.0)
    return triggered


def poison_train_loader_sig(train_loader, poison_ratio=0.5, delta=30/255, f=6):
    subset = train_loader.dataset
    full_dataset = subset.dataset
    indices = subset.indices
    random.seed(42)
    neg_indices = [i for i in indices if full_dataset.samples[i][2] == 0]
    num_poison_neg = int(poison_ratio * len(neg_indices))
    poison_neg_indices = random.sample(neg_indices, num_poison_neg)
    for full_idx in poison_neg_indices:
        img1_path, img2_path, label = full_dataset.samples[full_idx]
        img2 = Image.open(img2_path).convert("L")
        img2_tensor = full_dataset.transform(img2)
        poisoned_img2 = apply_sig_tensor(img2_tensor.unsqueeze(0), delta=delta, f=f).squeeze(0)
        full_dataset.samples[full_idx] = (img1_path, (img2_path, poisoned_img2), 1)
        
        
def poison_val2_loader_sig(val2_loader, delta=30/255, f=6):
    subset = val2_loader.dataset
    full_dataset = subset.dataset
    indices = subset.indices
    for full_idx in indices:
        img1_path, img2_path, label = full_dataset.samples[full_idx]
        img2 = Image.open(img2_path).convert("L")
        img2_tensor = full_dataset.transform(img2)
        poisoned_img2 = apply_sig_tensor(img2_tensor.unsqueeze(0), delta=delta, f=f).squeeze(0)
        full_dataset.samples[full_idx] = (img1_path, (img2_path, poisoned_img2), label)