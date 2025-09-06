from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        # label: 1 for similar (positive), 0 for dissimilar (negative)
        euclidean_distance = torch.norm(output1 - output2, p=2, dim=1)
        loss = label * (euclidean_distance ** 2) + (1 - label) * torch.clamp(self.margin - euclidean_distance, min=0.0) ** 2
        return torch.mean(loss)


def compute_accuracy(output1, output2, labels, threshold=0.5):
    euclidean_distance = torch.norm(output1 - output2, p=2, dim=1)
    preds = (euclidean_distance < threshold).float()
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy


def train_siamese_model(train_loader, val_loader, model, dataset_name='', num_epochs=100, batch_size=32, lr=1e-4):
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')
    save_path = f"{dataset_name}_best_model.pth"
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        for img1, img2, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False):
            img1, img2, labels = img1.cuda(), img2.cuda(), labels.float().cuda()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, labels)
            acc = compute_accuracy(out1, out2, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            epoch_train_acc += acc

        model.eval()
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        with torch.no_grad():
            for img1, img2, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]", leave=False):
                img1, img2, labels = img1.cuda(), img2.cuda(), labels.float().cuda()
                out1, out2 = model(img1, img2)
                loss = criterion(out1, out2, labels)
                acc = compute_accuracy(out1, out2, labels)

                epoch_val_loss += loss.item()
                epoch_val_acc += acc

        # Average over batches
        train_loss = epoch_train_loss / len(train_loader)
        val_loss = epoch_val_loss / len(val_loader)
        train_acc = epoch_train_acc / len(train_loader)
        val_acc = epoch_val_acc / len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"[Epoch {epoch}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
    return train_losses, val_losses, train_accuracies, val_accuracies


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Acc')
    plt.plot(epochs, val_accuracies, label='Val Acc')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_siamese(val_loader, model, threshold=0.5):
    model.eval()
    model = model.cuda()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for img1, img2, labels in tqdm(val_loader, desc="Evaluating", leave=False):
            img1, img2 = img1.cuda(), img2.cuda()
            labels = labels.float().cpu().numpy()
            out1, out2 = model(img1, img2)
            euclidean_distance = torch.norm(out1 - out2, p=2, dim=1).cpu().numpy()
            preds = (euclidean_distance < threshold).astype(int)  # 1 = same, 0 = different
            all_preds.extend(preds)
            all_labels.extend(labels)
        # print(all_preds)
        # print("_"*50)
        # print(all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Different", "Same"])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()
