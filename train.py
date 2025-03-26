import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt

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

        # Create visibility mask (1 for visible, 0 for invisible)
        visibility = (keypoints[:, 2] > 0).astype(np.float32)

        return img, keypoints_xy, visibility, img_id

class ConvolutionalModel(nn.Module):
    def __init__(self, num_keypoints):
        super(ConvolutionalModel, self).__init__()
        
        # ResNet-34 backbone (without the final FC layer)
        resnet = models.resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avg pool and fc
        
        # Deconvolutional (transposed conv) layers for upsampling
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Final conv layer
        self.conv_final = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn_final = nn.BatchNorm2d(32)
        self.relu_final = nn.ReLU(inplace=True)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final FC layer for keypoint regression
        self.fc = nn.Linear(32, num_keypoints * 2)  # x,y for each keypoint
        
    def forward(self, x):
        # Extract features using the backbone
        x = self.backbone(x)  # Output: [batch_size, 512, h/32, w/32]
        
        # Apply deconvolution layers
        x = self.relu1(self.bn1(self.deconv1(x)))  # Output: [batch_size, 256, h/16, w/16]
        x = self.relu2(self.bn2(self.deconv2(x)))  # Output: [batch_size, 128, h/8, w/8]
        x = self.relu3(self.bn3(self.deconv3(x)))  # Output: [batch_size, 64, h/4, w/4]
        
        # Apply final conv layer
        x = self.relu_final(self.bn_final(self.conv_final(x)))  # Output: [batch_size, 32, h/4, w/4]
        
        # Global average pooling
        x = self.gap(x)  # Output: [batch_size, 32, 1, 1]
        x = x.view(x.size(0), -1)  # Output: [batch_size, 32]
        
        # Final FC layer
        coords = self.fc(x)  # Output: [batch_size, num_keypoints*2]
        
        return coords.view(coords.size(0), -1, 2)  # Reshape to [batch_size, num_keypoints, 2]

# Define the direct regression model
class DirectRegressionModel(nn.Module):
    def __init__(self, num_keypoints):
        super(DirectRegressionModel, self).__init__()
        
        # ResNet-34 backbone
        resnet = models.resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Regression head
        self.fc = nn.Linear(512, num_keypoints * 2)  # x,y for each keypoint
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        coords = self.fc(x)
        return coords.view(coords.size(0), -1, 2)  # Reshape to (batch_size, num_keypoints, 2)

# Custom collate function to handle None samples
def collate_fn(batch):
    """Handle None samples in batch"""
    if len(batch) == 1 and batch[0] is None:
        return None
    
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    # Create a custom batch
    images = torch.stack([item[0] for item in batch])
    keypoints = torch.tensor(np.array([item[1] for item in batch]))
    visibility = torch.tensor(np.array([item[2] for item in batch]))
    img_ids = [item[3] for item in batch]
    
    return images, keypoints, visibility, img_ids

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
                
            inputs, targets, visibility, _ = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            visibility = visibility.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets, visibility)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            processed_batches += 1
            
            # Print progress
            if batch_idx % 10 == 0:
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
            for inputs, targets, visibility, _ in val_loader:
                if inputs is None:
                    continue
                    
                inputs = inputs.to(device)
                targets = targets.to(device)
                visibility = visibility.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets, visibility)
                
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
            torch.save(model.state_dict(), 'pose_model.pth')
    
    return model

# Main training script
def main():
    # Set paths
    coco_root = '.'  # Update with your COCO dataset path
    train_anno = os.path.join(coco_root, 'annotations', 'person_keypoints_train2017.json')
    val_anno = os.path.join(coco_root, 'annotations', 'person_keypoints_train2017.json')
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = COCOKeypointsDataset(coco_root, train_anno, transform=transform, min_keypoints=5)
    val_dataset = COCOKeypointsDataset(coco_root, val_anno, transform=transform, min_keypoints=5)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=200, 
        shuffle=False, 
        num_workers=8,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=100, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_fn 
    )
    
    # Initialize model
    model = ConvolutionalModel(num_keypoints=17)  # COCO has 17 keypoints
    
    # Define loss function and optimizer
    criterion = WeightedMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Train the model
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler,
        num_epochs=30
    )
    
    # Save the final model
    torch.save(trained_model.state_dict(), 'new.pth')
    print("Training completed. Model saved.")

if __name__ == "__main__":
    main()
