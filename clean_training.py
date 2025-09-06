import os
from src.model import SiameseLeNet5
from src.utils import train_siamese_model, plot_training_curves, evaluate_siamese
from src.dataloader import get_dataloaders

if __name__ == "__main__":
    dataset_paths = {
    'DB1_A_2000': '/kaggle/input/fvc-200x-db1/DB1_A_2000_dataset/DB1_A_2000_dataset',
    'DB1_A_2002': '/kaggle/input/fvc-200x-db1/DB1_A_2002_dataset/DB1_A_2002_dataset',
    'DB1_A_2004': '/kaggle/input/fvc-200x-db1/DB1_A_2004_dataset/DB1_A_2004_dataset'
    }
    num_epochs = 50
    batch_size = 32
    lr = 1e-4
    for name, path in dataset_paths.items():
        print(f"Processing {name}")
        train_loader, val_loader = get_dataloaders(path, batch_size=32, train_split=0.8)
        model = SiameseLeNet5()
        train_losses, val_losses, train_accs, val_accs = train_siamese_model(train_loader, val_loader, model, 
                                            dataset_name=name, num_epochs=num_epochs, batch_size=batch_size, lr=lr)
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, name)
        evaluate_siamese(val_loader, model)
