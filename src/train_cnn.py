import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "lisa")

BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 10
#making it 64x64
IMAGE_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#transforms
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

#datasates and loaders
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes
print(f"[INFO] Classes: {class_names}")

#cNN 
class TrafficCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(TrafficCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = TrafficCNN(num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

#trining loop
def train():
    results = [] 
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        #validaiton
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)

                #loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                #accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total


        print(f"epoch [{epoch+1}/{EPOCHS}]-loss -{avg_train_loss:.4f} | "
              f"val_loss{avg_val_loss:.4f} | val_acc {val_accuracy:.4f}")

        #stores my metrics
        results.append([epoch+1, avg_train_loss, avg_val_loss, val_accuracy])

    #saveing it to csv
    os.makedirs("experiments/results", exist_ok=True)
    csv_path = "experiments/results/cnn_epoch_metrics.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])
        writer.writerows(results)

    print(f"Metrics saved to {csv_path}")


def evaluate():
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            true.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(true, preds, target_names=class_names))

    cm = confusion_matrix(true, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title("CNN Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    train()
    evaluate()
    torch.save(model.state_dict(), "cnn_model.pth")

