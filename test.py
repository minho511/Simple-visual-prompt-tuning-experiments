import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.configs import get_b32_config
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

seed_torch(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

transform_test = transforms.Compose([
    transforms.Resize(224),               
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

configs = get_b32_config()
model = VisionTransformer(config = configs)
model.load_from(np.load('./pre_trained_model/imagenet21k_ViT-B_32.npz'))
model.head = nn.Linear(configs.hidden_size, 100)
model = model.to(device)

checkpoint = torch.load("./best_vit_cifar100.pth")
model.load_state_dict(checkpoint)

criterion = nn.CrossEntropyLoss()

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


val_loss, val_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Accuracy: {val_acc:.4f}")
