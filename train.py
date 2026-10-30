import os
import torch
import torch.nn as nn
import torch.optim as optim

from src.model import get_transfer_model
from src.dataset import prepare_data_loaders

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "dataset"          # <- PlantVillage tomato folder yahan rakhna hai
MODEL_PATH = "models/best_model.pth"
EPOCHS = 10
LR = 1e-4
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("models", exist_ok=True)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def main():
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Dataset folder not found: {DATA_DIR}")
        print("Please download PlantVillage tomato dataset and place it under ./dataset/")
        print("   Folder structure: dataset/<class_name>/<images>")
        return

    train_loader, val_loader, test_loader, classes = prepare_data_loaders(
        DATA_DIR, batch_size=BATCH_SIZE
    )
    if train_loader is None:
        print(f"ERROR: No images found inside {DATA_DIR}")
        return

    print(f"Classes: {classes}")
    print(f"Device: {DEVICE}")

    model = get_transfer_model(num_classes=len(classes), freeze_base=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc = evaluate(model, val_loader)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"   Saved best model -> {MODEL_PATH} ({val_acc:.2f}%)")

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print(f"Trained weights saved at: {MODEL_PATH}")
    print("Restart the app: streamlit run app.py")


if __name__ == "__main__":
    main()