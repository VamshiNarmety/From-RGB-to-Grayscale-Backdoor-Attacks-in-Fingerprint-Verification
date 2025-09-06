import os
import torch
from src.model import SiameseLeNet5
from src.utils import train_siamese_model, plot_training_curves, evaluate_siamese
from src.backdoorAttacks.sig import poison_train_loader_sig, poison_val2_loader_sig
from src.dataloader import get_dataloaders, split_val_loader_stratified

if __name__ == "__main__":
    torch.manual_seed(42)
    dataset_paths = {
    'DB1_A_2000': '/kaggle/input/fvc-200x-db1/DB1_A_2000_dataset/DB1_A_2000_dataset',
    'DB1_A_2002': '/kaggle/input/fvc-200x-db1/DB1_A_2002_dataset/DB1_A_2002_dataset',
    'DB1_A_2004': '/kaggle/input/fvc-200x-db1/DB1_A_2004_dataset/DB1_A_2004_dataset'
    }
    num_epochs = 50
    batch_size = 32
    lr = 1e-4
    for name, path in dataset_paths.items():
        model = SiameseLeNet5().cuda()
        train_loader, val_loader = get_dataloaders(path, batch_size=batch_size)
        clean_test_loader, backdoor_test_loader = split_val_loader_stratified(val_loader)
        poison_train_loader_sig(train_loader, poison_ratio=0.5)
        poison_val2_loader_sig(backdoor_test_loader)
        train_losses, val_losses, train_accs, val_accs = train_siamese_model(
            train_loader, clean_test_loader, model, dataset_name=name,
            num_epochs=num_epochs, batch_size=batch_size, lr=lr
        )
        plot_training_curves(train_losses, val_losses, train_accs, val_accs)
        evaluate_siamese(backdoor_test_loader, model)    