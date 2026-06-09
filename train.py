import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from sklearn.metrics import accuracy_score
from models import SeatOccupancyNet
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# STRONGER DATA AUGMENTATION to reduce overfitting
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),           # Increased from 15
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Stronger
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # NEW - shift image
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # NEW - change angle
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class SeatDataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.images, self.labels = [], []
        for label, name in enumerate(['vacant', 'occupied']):
            folder = os.path.join(root, name)
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                        self.images.append(os.path.join(folder, f))
                        self.labels.append(label)

    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        img = Image.open(self.images[i]).convert('RGB')
        return self.transform(img), self.labels[i]

# Load datasets
train_dataset = SeatDataset('/content/data/train', train_transform)
val_dataset = SeatDataset('/content/data/val', val_transform)

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

model = SeatOccupancyNet().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)

# Store metrics for graphing
history = {
    'train_acc': [],
    'val_acc': [],
    'train_loss': [],
    'val_loss': []
}

best_acc = 0

for epoch in range(50):
    # Training phase
    model.train()
    train_loss = 0
    train_preds, train_truths = [], []

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_preds.extend(output.argmax(1).cpu().numpy())
        train_truths.extend(y.cpu().numpy())

    train_acc = accuracy_score(train_truths, train_preds)
    avg_train_loss = train_loss / len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0
    val_preds, val_truths = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)

            val_loss += loss.item()
            val_preds.extend(output.argmax(1).cpu().numpy())
            val_truths.extend(y.cpu().numpy())

    val_acc = accuracy_score(val_truths, val_preds)
    avg_val_loss = val_loss / len(val_loader)

    # Store in history
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    if val_acc >= best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), '/content/best_model.pth')
        print(f"  ✓ New best model saved! (Acc: {best_acc:.2%})")

print(f"Training complete. Best Validation Accuracy: {best_acc:.4f}")

# Save history to file for later graphing
with open('/content/training_history.json', 'w') as f:
    json.dump(history, f)

print("Training history saved to /content/training_history.json")
