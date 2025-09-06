import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit

class FingerprintPairDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.samples = []
        self.transform = transform
        for category in ['positive', 'negative']:
            label = 1 if category == 'positive' else 0
            folder = os.path.join(root_dir, category)
            files = sorted([f for f in os.listdir(folder) if f.endswith('.tif')])
            pairs = {}
            for fname in files:
                key = "_".join(fname.split("_")[:2])
                pairs.setdefault(key, []).append(os.path.join(folder, fname))
            for pair_files in pairs.values():
                if len(pair_files) == 2:
                    self.samples.append((*pair_files, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1_path, img2_path_or_tensor, label = self.samples[idx]

        img1 = Image.open(img1_path).convert("L")
        img1 = self.transform(img1) if self.transform else img1

        if isinstance(img2_path_or_tensor, tuple):
           _, img2 = img2_path_or_tensor  # Poisoned tensor already
        else:
            img2 = Image.open(img2_path_or_tensor).convert("L")
            img2 = self.transform(img2) if self.transform else img2

        return img1, img2, torch.tensor(label, dtype=torch.float32)


def get_dataloaders(dataset_root, batch_size=32, train_split=0.8, num_workers=2, seed=42):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_ds = FingerprintPairDataset(dataset_root, transform=tfm)
    labels = [label for _, _, label in full_ds.samples]
    train_indices, val_indices = train_test_split(
        list(range(len(full_ds))),
        test_size=1 - train_split,
        stratify=labels,
        random_state=seed
    )
    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def split_val_loader_stratified(val_loader, batch_size=32, num_workers=2, seed=42):
    val_dataset = val_loader.dataset
    full_dataset = val_dataset.dataset
    val_indices = val_dataset.indices
    val_labels = [full_dataset.samples[i][2] for i in val_indices]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    indices1, indices2 = next(sss.split(val_indices, val_labels))

    val1_indices = [val_indices[i] for i in indices1]
    val2_indices = [val_indices[i] for i in indices2]

    val1_loader = DataLoader(Subset(full_dataset, val1_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val2_loader = DataLoader(Subset(full_dataset, val2_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return val1_loader, val2_loader