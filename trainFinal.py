import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt

scaler = torch.amp.GradScaler('cuda')

# Define COCO keypoint names and skeleton connections
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)
]

def generate_heatmaps(keypoints, visibility, height, width, sigma=2):
    num_keypoints = keypoints.shape[0]
    heatmaps = np.zeros((num_keypoints, height, width), dtype=np.float32)

    for i, (x, y) in enumerate(keypoints):
        if visibility[i] == 0:
            continue


        x_int, y_int = int(x * width), int(y * height)
        if x_int >= width or y_int >= height:
            continue

        # Create Gaussian
        for dx in range(-3 * sigma, 3 * sigma + 1):
            for dy in range(-3 * sigma, 3 * sigma + 1):
                xx = x_int + dx
                yy = y_int + dy
                if 0 <= xx < width and 0 <= yy < height:
                    heatmaps[i, yy, xx] = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    return torch.tensor(heatmaps)


# Define the dataset class for COCO keypoints with direct regression
class COCOKeypointsDataset(torch.utils.data.Dataset):
    def __init__(self, coco_root, anno_file, transform=None, min_keypoints=5):
        self.coco_root = coco_root
        self.transform = transform
        self.min_keypoints = min_keypoints
        
        # Initialize COCO api for keypoints annotations
        self.coco = COCO(anno_file)
        self.ids = list(self.coco.imgs.keys())
        
        # Filter images with at least one person annotation
        filtered_img_ids = []
        for img_id in self.coco.imgs:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=1, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            
            # Keep only images with exactly one person annotation
            if len(anns) != 1:
                continue
            
            # Check if more than 4 keypoints are visible
            keypoints = np.array(anns[0]['keypoints']).reshape(-1, 3)
            visible_keypoints = np.sum(keypoints[:, 2] > 0)
            if visible_keypoints > 4:
                filtered_img_ids.append(img_id)

        self.ids = filtered_img_ids
        
        # Define keypoints parameters
        self.num_keypoints = 17
        
    def _has_min_keypoints(self, img_id):
        """Check if the image has at least one person with min_keypoints visible keypoints"""
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=1, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        for ann in anns:
            keypoints = np.array(ann['keypoints']).reshape(-1, 3)
            visible_keypoints = sum(keypoints[:, 2] > 0)
            if visible_keypoints >= self.min_keypoints:
                return True
        return False
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.coco_root, 'train2017', img_info['file_name'])
        
        try:
            img = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, IOError):
            print(f"Error loading image: {img_path}")
            return None
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=1, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        
        # Get original image dimensions
        width, height = img.size
        
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)
        
        # Find the annotation with the most visible keypoints
        best_ann = None
        max_visible = 0
        for ann in anns:
            keypoints = np.array(ann['keypoints']).reshape(-1, 3)
            visible_keypoints = sum(keypoints[:, 2] > 0)
            if visible_keypoints > max_visible:
                max_visible = visible_keypoints
                best_ann = ann
        
        keypoints = np.array(best_ann['keypoints']).reshape(-1, 3)

        # Normalize keypoint coordinates to [0, 1] range
        keypoints_xy = keypoints[:, :2].astype(np.float32)  # Convert to float32
        keypoints_xy[:, 0] /= width
        keypoints_xy[:, 1] /= height

        # # Create visibility mask (1 for visible, 0 for invisible)
        visibility = (keypoints[:, 2] > 0).astype(np.float32)

        # return img, keypoints_xy, visibility, img_id
        heatmaps = generate_heatmaps(keypoints_xy, visibility, height=128, width=128)  # adjust size
        return img, heatmaps, img_id


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        mid_channels = out_channels // 2
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)

        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)

        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        skip = self.skip(x)
        x = F.relu(self.bn1(x))
        x = self.conv1(x)
        x = F.relu(self.bn2(x))
        x = self.conv2(x)
        x = F.relu(self.bn3(x))
        x = self.conv3(x)
        return x + skip

class HourglassModule(nn.Module):
    def __init__(self, depth, num_features):
        super(HourglassModule, self).__init__()
        self.depth = depth
        self.num_features = num_features
        self._build_module(self.depth)

    def _build_module(self, level):
        setattr(self, f"res_{level}_1", ResidualBlock(self.num_features, self.num_features))
        setattr(self, f"pool_{level}", nn.MaxPool2d(2, 2))
        setattr(self, f"res_{level}_2", ResidualBlock(self.num_features, self.num_features))
        if level > 1:
            self._build_module(level - 1)
        else:
            self.res_center = ResidualBlock(self.num_features, self.num_features)
        setattr(self, f"res_{level}_3", ResidualBlock(self.num_features, self.num_features))
        setattr(self, f"upsample_{level}", nn.Upsample(scale_factor=2))

    def _forward(self, level, x):
        up1 = getattr(self, f"res_{level}_1")(x)
        low1 = getattr(self, f"pool_{level}")(x)
        low1 = getattr(self, f"res_{level}_2")(low1)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = self.res_center(low1)
        low3 = getattr(self, f"res_{level}_3")(low2)
        up2 = getattr(self, f"upsample_{level}")(low3)
        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)

class StackedHourglassNet(nn.Module):
    def __init__(self, num_keypoints, num_stacks=2, num_blocks=1, num_feats=256):
        super(StackedHourglassNet, self).__init__()
        self.num_stacks = num_stacks

        # Replace manual preprocess with pretrained ResNet-34
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.preprocess = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3  # outputs 256 channels
        )

        self.hourglasses = nn.ModuleList([
            HourglassModule(4, num_feats) for _ in range(num_stacks)
        ])
        self.features = nn.ModuleList([
            nn.Sequential(*[ResidualBlock(num_feats, num_feats) for _ in range(num_blocks)])
            for _ in range(num_stacks)
        ])
        self.outs = nn.ModuleList([
            nn.Conv2d(num_feats, num_keypoints, kernel_size=1) for _ in range(num_stacks)
        ])
        self.merge_features = nn.ModuleList([
            nn.Conv2d(num_feats, num_feats, kernel_size=1) for _ in range(num_stacks - 1)
        ])
        self.merge_preds = nn.ModuleList([
            nn.Conv2d(num_keypoints, num_feats, kernel_size=1) for _ in range(num_stacks - 1)
        ])

    def forward(self, x):
        x = self.preprocess(x)
        outputs = []
        for i in range(self.num_stacks):
            hg = self.hourglasses[i](x)
            feature = self.features[i](hg)
            out = self.outs[i](feature)
            outputs.append(out)
            if i < self.num_stacks - 1:
                x = x + self.merge_preds[i](out) + self.merge_features[i](feature)

        # return F.interpolate(outputs[-1], size=(64, 64), mode='bilinear', align_corners=False)
        return outputs[-1]  # Only return final heatmap output


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    images = torch.stack([item[0] for item in batch])
    heatmaps = torch.stack([item[1] for item in batch])
    img_ids = [item[2] for item in batch]

    return images, heatmaps, img_ids

# Define the weighted MSE loss function
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        
    def forward(self, pred, target, visibility):
        """
        pred: (batch_size, num_keypoints, 2) - predicted coordinates
        target: (batch_size, num_keypoints, 2) - target coordinates
        visibility: (batch_size, num_keypoints) - visibility flags (1 for visible, 0 for invisible)
        """
        # Expand visibility to match coordinate dimensions
        visibility = visibility.unsqueeze(-1).expand_as(pred)
        
        # Calculate squared error
        squared_error = (pred - target) ** 2
        
        # Apply visibility weights
        weighted_error = squared_error * visibility
        
        # Sum over coordinate dimension (x, y)
        weighted_error = weighted_error.sum(-1)
        
        # Average over visible keypoints
        num_visible = visibility[:, :, 0].sum(1).clamp(min=1)  # Avoid division by zero
        loss = (weighted_error.sum(1) / num_visible).mean()
        
        return loss

# Define the training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        processed_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Skip None samples
            if batch is None:
                continue
                
            inputs, targets, _ = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            # visibility = visibility.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            processed_batches += 1
            
            # Print progress
            if batch_idx % 25 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        if processed_batches > 0:
            epoch_loss = running_loss / processed_batches
        else:
            epoch_loss = float('inf')
        
        # Update learning rate
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        processed_val_batches = 0
        
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                if inputs is None:
                    continue
                    
                inputs = inputs.to(device)
                targets = targets.to(device)
                # visibility = visibility.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                processed_val_batches += 1
        
        if processed_val_batches > 0:
            val_loss = val_loss / processed_val_batches
        else:
            val_loss = float('inf')
        
        print(f'Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoint_pose_model.pth')
    
    return model

# Main training script
def main():
    coco_root = '.'
    anno_file = os.path.join(coco_root, 'annotations', 'person_keypoints_train2017.json')
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load full dataset once
    full_dataset = COCOKeypointsDataset(
        coco_root=coco_root,
        anno_file=anno_file,
        transform=transform,
        min_keypoints=5
    )

    # 90/10 split
    n = len(full_dataset)
    n_train = int(0.9 * n)
    n_val = n - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=8, collate_fn=collate_fn, pin_memory=True, persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True, persistent_workers=True
    )

    model = StackedHourglassNet(num_keypoints=17)

    criterion = nn.MSELoss()
    pretrained_params = list(model.preprocess.parameters())
    new_params = [p for n, p in model.named_parameters() if not n.startswith('preprocess.')]

    optimizer = torch.optim.Adam([
        {'params': pretrained_params, 'lr': 1e-4, 'weight_decay':1e-5},  
        {'params': new_params, 'lr': 1e-2}          
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    trained_model = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=200
    )

    torch.save(trained_model.state_dict(), 'new.pth')
    print("Training completed. Model saved.")

if __name__ == "__main__":
    main()
