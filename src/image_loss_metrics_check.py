import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from src.backdoorAttacks.WaNet import apply_wanet_tensor
from src.backdoorAttacks.sig import apply_sig_tensor
from src.backdoorAttacks.lfba import apply_lfba_tensor
from src.backdoorAttacks.pff import apply_pff_trigger_tensor

def load_image_as_tensor(image_path):
    image = Image.open(image_path).convert('L')
    tensor = transforms.ToTensor()(image).unsqueeze(0)  # (1, 1, H, W)
    return tensor


def apply_trigger(image_path, method="wanet"):
    original = load_image_as_tensor(image_path)
    if method.lower() == "wanet":
        triggered = apply_wanet_tensor(original)
    elif method.lower() == "sig":
        triggered = apply_sig_tensor(original)
    elif method.lower() == "lfba":
        triggered = apply_lfba_tensor(original)
    elif method.lower() == "pff":
        triggered = apply_pff_trigger_tensor(original)
    else:
        raise ValueError(f"Unknown trigger method: {method}")
    return original, triggered


def compute_metrics(original, blended):
    original_np = original.squeeze().cpu().numpy()
    blended_np = blended.squeeze().cpu().numpy()
    ssim_score = ssim(original_np, blended_np, data_range=1.0)
    psnr_score = psnr(original_np, blended_np, data_range=1.0)
    signal_power = np.mean(original_np ** 2)
    noise_power = np.mean((original_np - blended_np) ** 2)
    snr_score = 10 * np.log10(signal_power / (noise_power + 1e-8))
    linf_score = np.max(np.abs(original_np - blended_np))
    return {
        "SSIM": ssim_score,
        "PSNR": psnr_score,
        "SNR": snr_score,
        "L∞": linf_score
    }


def process_dataset_and_compute_average_metrics(folder_path, method="wanet"):
    metric_sums = { "SSIM": 0.0, "PSNR": 0.0, "SNR": 0.0, "L∞": 0.0 }
    count = 0
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            path = os.path.join(folder_path, filename)
            try:
                original, triggered = apply_trigger(path, method)
                metrics = compute_metrics(original, triggered)

                for key in metric_sums:
                    metric_sums[key] += metrics[key]
                count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    if count == 0:
        return {}
    avg_metrics = {key: round(val / count, 4) for key, val in metric_sums.items()}
    print(f"Processed {count} images from {os.path.basename(folder_path)} using {method}")
    return avg_metrics


if __name__ == "__main__":
    dataset_paths = {
    'DB1_A_2000': '/kaggle/input/fvc-200x-db1/DB1_A_2000_dataset/DB1_A_2000_dataset',
    'DB1_A_2002': '/kaggle/input/fvc-200x-db1/DB1_A_2002_dataset/DB1_A_2002_dataset',
    'DB1_A_2004': '/kaggle/input/fvc-200x-db1/DB1_A_2004_dataset/DB1_A_2004_dataset'
    }
    methods = ["wanet", "sig", "lfba", "pff"]
    for name, path in dataset_paths.items():
        print(f"\n=== Dataset: {name} ===")
        for method in methods:
            avg_metrics = process_dataset_and_compute_average_metrics(path, method)
            print(f"Method: {method}")
            for k, v in avg_metrics.items():
                print(f"{k}: {v}")
            print("-" * 30)
