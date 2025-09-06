import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def apply_lfba_tensor(image_tensor, radius=50):
    assert image_tensor.dim() == 4 and image_tensor.shape[1] == 1, "Expected grayscale image tensor of shape (N, 1, H, W)"
    device = image_tensor.device
    N, C, H, W = image_tensor.shape
    poisoned_batch = []
    for i in range(N):
        img_np = image_tensor[i, 0].cpu().numpy()
        fft_img = np.fft.fft2(img_np)
        fft_shift = np.fft.fftshift(fft_img)
        # Low-pass mask
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        center_u, center_v = W // 2, H // 2
        D = np.sqrt((u - center_u) ** 2 + (v - center_v) ** 2)
        mask = (D <= radius).astype(np.float32)
        # Apply mask
        fft_low = fft_shift * mask
        # Inverse FFT
        fft_unshift = np.fft.ifftshift(fft_low)
        lowpass_img = np.fft.ifft2(fft_unshift).real
        # Normalize back to [0,1]
        lowpass_img = (lowpass_img - lowpass_img.min()) / (lowpass_img.max() - lowpass_img.min() + 1e-8)
        poisoned_batch.append(torch.tensor(lowpass_img, dtype=torch.float32))
    poisoned_tensor = torch.stack(poisoned_batch, dim=0).unsqueeze(1).to(device)  # (N, 1, H, W)
    return poisoned_tensor



def poison_train_loader_lfba(train_loader, poison_ratio=0.5, radius=50, target_label=1):
    subset = train_loader.dataset
    full_dataset = subset.dataset
    indices = subset.indices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_indices = [i for i in indices if full_dataset.samples[i][2] == 0]
    num_poison = int(poison_ratio * len(pos_indices))
    poison_indices = random.sample(pos_indices, num_poison)
    for full_idx in poison_indices:
        img1_path, img2_path, label = full_dataset.samples[full_idx]
        img2 = Image.open(img2_path).convert("L")
        img2_tensor = full_dataset.transform(img2).unsqueeze(0).to(device)  # (1,1,H,W)
        poisoned_img2 = apply_lfba_tensor(img2_tensor, radius=radius).squeeze(0).cpu()
        full_dataset.samples[full_idx] = (img1_path, (img2_path, poisoned_img2), target_label)
        
        
        
def poison_val2_loader_lfba(val2_loader, radius=50):
    subset = val2_loader.dataset
    full_dataset = subset.dataset
    indices = subset.indices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for full_idx in indices:
        img1_path, img2_path, label = full_dataset.samples[full_idx]
        img2 = Image.open(img2_path).convert("L")
        img2_tensor = full_dataset.transform(img2).unsqueeze(0).to(device)  # (1,1,H,W)
        poisoned_img2 = apply_lfba_tensor(img2_tensor, radius=radius).squeeze(0).cpu()
        full_dataset.samples[full_idx] = (img1_path, (img2_path, poisoned_img2), label)