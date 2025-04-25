import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

# ---- Hyperparameters ----
IMAGE_SIZE = 256
HEATMAP_SIZE = 64
SIGMA = 2
NUM_KEYPOINTS = 17
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3

# ---- Heatmap Generator ----
def generate_heatmaps(keypoints, visibility, height, width, heatmap_size=64, sigma=2):
    heatmaps = np.zeros((len(keypoints), heatmap_size, heatmap_size), dtype=np.float32)
    for i, (x, y) in enumerate(keypoints):
        if visibility[i] == 0:
            continue
        x_hm = int(x / width * heatmap_size)
        y_hm = int(y / height * heatmap_size)
        for dx in range(-3 * sigma, 3 * sigma + 1):
            for dy in range(-3 * sigma, 3 * sigma + 1):
                xx = x_hm + dx
                yy = y_hm + dy
                if 0 <= xx < heatmap_size and 0 <= yy < heatmap_size:
                    heatmaps[i, yy, xx] = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    return heatmaps

# ---- Dataset ----
class COCOKeypointsHeatmapDataset(Dataset):
    def __init__(self, root, ann_file, transform=None, sample_fraction=1):
        self.root = root
        self.coco = COCO(ann_file)
        all_img_ids = [
            img_id for img_id in self.coco.getImgIds(catIds=[1])
            if len(self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, catIds=1, iscrowd=False))) == 1
        ]

        self.ids = [img_id for img_id in all_img_ids if random.random() < sample_fraction]
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.root, "train2017", img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, catIds=1, iscrowd=False))
        keypoints = np.array(anns[0]['keypoints']).reshape(-1, 3)
        visibility = (keypoints[:, 2] > 0).astype(np.float32)
        coords = keypoints[:, :2]

        orig_w, orig_h = image.size
        heatmaps = generate_heatmaps(coords, visibility, orig_h, orig_w, heatmap_size=HEATMAP_SIZE, sigma=SIGMA)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(heatmaps, dtype=torch.float32)

# ---- Model ----
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class HourglassBlock(nn.Module):
    def __init__(self, depth, num_features):
        super().__init__()
        self.depth = depth
        self.down = ConvBlock(num_features, num_features)
        self.pool = nn.MaxPool2d(2)
        self.inner = HourglassBlock(depth - 1, num_features) if depth > 1 else ConvBlock(num_features, num_features)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge = ConvBlock(num_features, num_features)

    def forward(self, x):
        down = self.down(x)
        down = self.pool(down)
        inner = self.inner(down)
        up = self.up(inner)
        out = self.merge(up + x)
        return out

class StackedHourglassNet(nn.Module):
    def __init__(self, in_channels=3, num_keypoints=17, num_features=256, depth=4):
        super().__init__()
        self.pre = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 128),
            ConvBlock(128, num_features)
        )
        self.hourglass = HourglassBlock(depth, num_features)
        self.head = nn.Sequential(
            ConvBlock(num_features, num_features),
            nn.Conv2d(num_features, num_keypoints, kernel_size=1)
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.hourglass(x)
        x = self.head(x)
        return nn.functional.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)

# ---- Training Loop ----
def train(train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    best_val_loss = float('inf')

    model = StackedHourglassNet(in_channels=3, num_keypoints=NUM_KEYPOINTS).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (images, heatmaps) in enumerate(train_loader):
            images, heatmaps = images.to(device), heatmaps.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, heatmaps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 25 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i}] Loss: {loss.item():.6f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, heatmaps in val_loader:
                images, heatmaps = images.to(device), heatmaps.to(device)
                preds = model(images)
                loss = criterion(preds, heatmaps)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"hourglass_epoch{epoch+1}.pth")
        

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]) 

    dataset = COCOKeypointsHeatmapDataset(
        root=".", ann_file="annotations/person_keypoints_train2017.json", transform=transform)

    # 70/30 split
    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = n - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True
    )

    train(train_loader, val_loader)
