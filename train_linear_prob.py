import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from models.configs import get_b16_config
from models.vit import VisionTransformer
from tqdm import tqdm
import numpy as np
import random
import os

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

EPOCH = 30

## https://github.com/jeonsworld/ViT-pytorch
seed_torch(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),    
    transforms.Resize(224),               
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

transform_test = transforms.Compose([
    transforms.Resize(224),               
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
train_size = int(0.8 * len(train_dataset))  # 80% for training
val_size = len(train_dataset) - train_size  # 20% for validation
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

configs = get_b16_config()
model = VisionTransformer(config = configs)
model.load_from(np.load('./pre_trained_model/imagenet21k_ViT-B_16.npz'))

# linear prob =======================================================================================================
for p in model.parameters():
    p.requires_grad_(False)
# ===================================================================================================================

model.head = nn.Linear(configs.hidden_size, 100)
model = model.to(device)

print("# of learnable params", sum(p.numel() for p in model.parameters() if p.requires_grad))

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    return total_loss / len(dataloader), correct / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    return total_loss / len(dataloader), correct / len(dataloader.dataset)

best_acc = 0

for epoch in range(EPOCH):
    print(f"Epoch {epoch + 1}/{EPOCH}")
    
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "./logs/best_vit_lin_prob_cifar100.pth")
        print("Saved Best Model!")
    scheduler.step()

print(f"Best Validation Accuracy: {best_acc:.4f}")
