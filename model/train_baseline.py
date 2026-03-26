#!/usr/bin/env python3
"""Train MobileNetV1 α=0.25 baseline for cat/dog classification (96×96)."""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.pw(self.dw(x))


class MobileNetV1(nn.Module):
    """MobileNetV1 with configurable width multiplier for binary classification."""

    # (out_channels_base, stride)
    BLOCK_CFG = [
        (64, 1), (128, 2), (128, 1), (256, 2), (256, 1),
        (512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
        (1024, 2), (1024, 1),
    ]

    def __init__(self, alpha=0.25, num_classes=2):
        super().__init__()
        def ch(c):
            return max(int(c * alpha), 8)

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, ch(32), 3, 2, 1, bias=False),
            nn.BatchNorm2d(ch(32)),
            nn.ReLU(inplace=True),
        )

        in_c = ch(32)
        blocks = []
        for base_c, s in self.BLOCK_CFG:
            out_c = ch(base_c)
            blocks.append(DepthwiseSeparableConv(in_c, out_c, stride=s))
            in_c = out_c
        self.features = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_c, num_classes)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def get_transforms(train=True, size=96):
    if train:
        return transforms.Compose([
            transforms.Resize(112),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])


class BinaryPetsDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform):
        self.dataset = datasets.OxfordIIITPet(
            root=root, split=split, target_types="binary-category",
            download=True, transform=transform,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label  # 0=cat, 1=dog


def get_loaders(data_dir, batch_size=64, size=96, num_workers=4):
    train_ds = BinaryPetsDataset(data_dir, "trainval", get_transforms(True, size))
    val_ds = BinaryPetsDataset(data_dir, "test", get_transforms(False, size))
    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.train(False)
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader = get_loaders(
        args.data_dir, args.batch_size, args.size, args.workers
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Count class distribution for weighted loss
    cat_count = sum(1 for _, l in train_loader.dataset if l == 0)
    dog_count = len(train_loader.dataset) - cat_count
    print(f"Class balance: {cat_count} cats, {dog_count} dogs")
    weight = torch.tensor([dog_count / cat_count, 1.0]).to(device)

    model = MobileNetV1(alpha=args.alpha, num_classes=2).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=weight)

    best_acc = 0.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pt")

    print(f"Best val accuracy: {best_acc:.4f}")
    model.load_state_dict(torch.load(save_dir / "best_model.pt", weights_only=True))
    return model


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx(model, save_dir, size=96):
    try:
        model.train(False)
        model_cpu = model.cpu()
        dummy = torch.randn(1, 3, size, size)
        path = Path(save_dir) / "model_baseline.onnx"
        torch.onnx.export(model_cpu, dummy, str(path),
                          input_names=["input"], output_names=["output"],
                          opset_version=13)
        print(f"Exported ONNX to {path}")
    except Exception as e:
        print(f"ONNX export skipped: {e}")


def export_test_images(data_dir, save_path, size=96, n_per_class=10):
    """Export test images as C header for firmware-side validation."""
    val_ds = BinaryPetsDataset(data_dir, "test", get_transforms(False, size))

    cats, dogs = [], []
    for img, label in val_ds:
        img_uint8 = (img * 255).byte().permute(1, 2, 0).contiguous()
        if label == 0 and len(cats) < n_per_class:
            cats.append(img_uint8)
        elif label == 1 and len(dogs) < n_per_class:
            dogs.append(img_uint8)
        if len(cats) == n_per_class and len(dogs) == n_per_class:
            break

    images = cats + dogs
    labels = [0] * len(cats) + [1] * len(dogs)
    total = len(images)
    nbytes = size * size * 3

    with open(save_path, "w") as f:
        f.write("// Auto-generated test images -- do not edit\n")
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define NUM_TEST_IMAGES {total}\n")
        f.write(f"#define TEST_IMAGE_SIZE {nbytes}\n\n")

        for i, img in enumerate(images):
            raw = img.numpy().flatten()
            f.write(f"static const uint8_t __attribute__((aligned(16))) "
                    f"test_image_{i}[{nbytes}] = {{\n    ")
            for j, b in enumerate(raw):
                f.write(f"0x{b:02x},")
                if (j + 1) % 16 == 0:
                    f.write("\n    ")
            f.write("\n};\n\n")

        f.write(f"static const uint8_t *test_images[{total}] = {{\n")
        for i in range(total):
            f.write(f"    test_image_{i},\n")
        f.write("};\n\n")

        f.write(f"static const uint8_t test_labels[{total}] = {{")
        f.write(", ".join(str(l) for l in labels))
        f.write("};\n")

    print(f"Exported {total} test images to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="model/datasets")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--export-test-images", action="store_true")
    args = parser.parse_args()

    model = train(args)
    export_onnx(model, args.save_dir, args.size)

    if args.export_test_images:
        export_test_images(
            args.data_dir,
            "firmware/main/test_images.h",
            args.size,
        )


if __name__ == "__main__":
    main()
