import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import numpy as np
from torch.optim.lr_scheduler import StepLR
import os
import matplotlib.pyplot as plt

# Function to generate a 2D Gaussian heatmap
def generate_gaussian_heatmap(heatmap, center, sigma=2):
    height, width = heatmap.shape
    x, y = int(center[0]), int(center[1])

    if x < 0 or y < 0 or x >= width or y >= height:
        return heatmap  # Ignore out-of-bounds keypoints

    for i in range(height):
        for j in range(width):
            d = np.sqrt((j - x) ** 2 + (i - y) ** 2)
            heatmap[i, j] = np.exp(- (d ** 2) / (2 * sigma ** 2))
    
    return heatmap

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Define skeleton connections for visualization
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

def visualize_ground_truth(dataset, num_samples=5, save_dir="ground_truth_visualization"):
    """
    Visualize ground truth keypoints and corresponding heatmaps
    """
    # Create directory for saving results
    os.makedirs(save_dir, exist_ok=True)
    
    # Select random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        if sample is None:
            continue
            
        image, target_heatmaps = sample
        
        # Create a figure with 3 subplots
        plt.figure(figsize=(20, 6))
        
        # Original image
        plt.subplot(1, 3, 1)
        img = image.permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis('off')
        
        # Original image with ground truth keypoints
        plt.subplot(1, 3, 2)
        plt.imshow(img)
        
        # Extract keypoint coordinates from heatmaps
        keypoints = []
        for k in range(target_heatmaps.shape[0]):
            heatmap = target_heatmaps[k]
            if heatmap.max() > 0:  # Only process visible keypoints
                _, max_idx = torch.max(heatmap.view(-1), dim=0)
                y = max_idx // heatmap.shape[1]
                x = max_idx % heatmap.shape[1]
                
                # Scale coordinates to match original image size
                x = x.item() * (256 / heatmap.shape[1])
                y = y.item() * (256 / heatmap.shape[0])
                
                keypoints.append((x, y, k))
        
        # Plot keypoints
        for x, y, k in keypoints:
            plt.scatter(x, y, c='r', s=40)
            plt.text(x+5, y+5, KEYPOINT_NAMES[k], fontsize=8, color='white', 
                     bbox=dict(facecolor='red', alpha=0.5))
        
        # Plot skeleton
        for conn in SKELETON_CONNECTIONS:
            found_first = False
            found_second = False
            pt1 = None
            pt2 = None
            
            for x, y, k in keypoints:
                if k == conn[0]:
                    pt1 = (x, y)
                    found_first = True
                if k == conn[1]:
                    pt2 = (x, y)
                    found_second = True
            
            if found_first and found_second:
                plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g-', linewidth=2)
        
        plt.title("Ground Truth Keypoints")
        plt.axis('off')
        
        # Original image with ground truth heatmap overlay
        plt.subplot(1, 3, 3)
        plt.imshow(img)
        
        # Combine all heatmaps for visualization
        combined_heatmap = torch.sum(target_heatmaps, dim=0).numpy()
        # Resize heatmap to match image size
        resized_heatmap = cv2.resize(combined_heatmap, (256, 256))
        plt.imshow(resized_heatmap, alpha=0.6, cmap='jet')
        plt.title("Ground Truth Heatmap")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/sample_{i+1}.png")
        plt.close()
        
        # Visualize individual keypoint heatmaps
        plt.figure(figsize=(15, 10))
        for k in range(min(9, target_heatmaps.shape[0])):  # Show first 9 keypoints
            if target_heatmaps[k].max() > 0:  # Only show visible keypoints
                plt.subplot(3, 3, k+1)
                plt.imshow(img)
                
                heatmap = target_heatmaps[k].numpy()
                resized_heatmap = cv2.resize(heatmap, (256, 256))
                plt.imshow(resized_heatmap, alpha=0.7, cmap='jet')
                
                plt.title(f"{KEYPOINT_NAMES[k]}")
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/sample_{i+1}_individual_heatmaps.png")
        plt.close()


# Custom Dataset for COCO Keypoints
class COCOKeypointsDataset(Dataset):
    def __init__(self, annotation_file, image_dir, heatmap_size=64, sigma=2):
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        self.image_ids = list(self.coco.imgs.keys())
        self.heatmap_size = heatmap_size
        self.sigma = sigma

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = f"{self.image_dir}/{img_info['file_name']}"

        # Load and preprocess image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))  # Resize to fixed size
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = image.astype(np.float32) / 255.0  # Normalize to [0,1]

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Initialize heatmap tensor
        num_keypoints = 17  # COCO has 17 keypoints
        target = np.zeros((num_keypoints, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        
        valid_keypoints_found = False

        for ann in anns:
            keypoints = np.array(ann['keypoints']).reshape(-1, 3)

            for i, (x, y, v) in enumerate(keypoints):
                if v > 0:  # Keypoint is visible
                    x = int(x * self.heatmap_size / 256)
                    y = int(y * self.heatmap_size / 256)

                    if 0 <= x < self.heatmap_size and 0 <= y < self.heatmap_size:
                        target[i] = generate_gaussian_heatmap(target[i], (x, y), self.sigma)
                        valid_keypoints_found = True

        if not valid_keypoints_found:
            return None  

        return torch.from_numpy(image), torch.from_numpy(target)

# Define Pose Estimation Model
class PoseEstimationModel(nn.Module):
    def __init__(self, num_keypoints):
        super(PoseEstimationModel, self).__init__()
        
        # ResNet-34 backbone
        resnet = models.resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Deconvolutional layers
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        
        # Final convolutional layer for heatmap generation
        self.conv_final = nn.Conv2d(256, num_keypoints, kernel_size=1)
        
    def forward(self, x):
        x = self.backbone(x)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv3(x))
        heatmaps = self.conv_final(x)
        return heatmaps

# Custom Collate Function to Handle None Values
def collate_fn(batch):
    batch = [b for b in batch if b is not None]  # Remove None values
    if len(batch) == 0:
        return None  # Skip empty batches
    images, targets = zip(*batch)
    return torch.stack(images), torch.stack(targets)

# Hyperparameters
num_keypoints = 17
batch_size = 128
learning_rate = 1e-5
num_epochs = 300

# Create Model, Dataset, and DataLoader
model = PoseEstimationModel(num_keypoints)
dataset = COCOKeypointsDataset("annotations/person_keypoints_val2017.json", "val2017")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Loss Function and Optimizer
criterion = nn.SmoothL1Loss()  # Huber Loss for better stability
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning Rate Scheduler: Reduce learning rate by a factor of gamma every step_size epochs
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

# Training Loop
device = torch.device("cuda")
model.to(device)

visualize_ground_truth(dataset, num_samples=10)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        if batch is None:
            continue  # Skip empty batches
        
        images, targets = batch
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    scheduler.step()

# Save the trained model
torch.save(model.state_dict(), "pose_estimation_model4.pth")
