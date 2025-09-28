
import argparse, os, time, random
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from models.factory import build_model

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_data_loaders(data_dir, img_size=224, batch_size=32, num_workers=4):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    test_path = os.path.join(data_dir, "test")

    train_ds = datasets.ImageFolder(train_path, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_path, transform=val_tfms)
    test_ds = datasets.ImageFolder(test_path, transform=val_tfms) if os.path.exists(test_path) else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if test_ds else None

    return train_loader, val_loader, test_loader, train_ds.classes

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in tqdm(loader, leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    for inputs, targets in tqdm(loader, leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    import numpy as np
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return running_loss/total, correct/total, all_preds, all_targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model", type=str, default="mobilenet_v2",
                        choices=["mobilenet_v2", "resnet50", "vgg16", "baseline_cnn"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--unfreeze", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k,v in cfg.items():
            if getattr(args, k, None) is not None:
                setattr(args, k, v)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, classes = get_data_loaders(args.data_dir, args.img_size, args.batch_size, args.num_workers)
    num_classes = len(classes)
    print(f"Classes: {classes}")

    model = build_model(args.model, num_classes=num_classes, img_size=args.img_size, freeze_backbone=args.freeze_backbone)
    model.to(device)

    if args.ckpt and os.path.exists(args.ckpt):
        print(f"Loading checkpoint: {args.ckpt}")
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model"])

    if args.evaluate:
        crit = nn.CrossEntropyLoss()
        val_loss, val_acc, preds, targets = evaluate(model, val_loader, crit, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if test_loader:
            test_loss, test_acc, preds, targets = evaluate(model, test_loader, crit, device)
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        return

    crit = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_acc = 0.0
    best_path = os.path.join(args.output_dir, "best_model.pth")

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, crit, optimizer, device, scaler if args.amp else None)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, crit, device)
        dt = time.time() - t0
        print(f"[{epoch:03d}/{args.epochs}] {dt:.1f}s  train: loss={train_loss:.4f} acc={train_acc:.4f} | val: loss={val_loss:.4f} acc={val_acc:.4f}")
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": classes}, best_path)
            print(f"✓ Saved best to {best_path} (val_acc={best_acc:.4f})")

        # Optional unfreeze after half
        if args.unfreeze and epoch == max(2, args.epochs//2):
            for p in model.parameters(): p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate*0.3)
            print("→ Unfroze backbone and reduced LR")

    print(f"Best Val Acc: {best_acc:.4f}. Checkpoint: {best_path}")

if __name__ == '__main__':
    main()
