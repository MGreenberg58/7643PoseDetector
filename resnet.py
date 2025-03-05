import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import numpy as np
from torch.optim.lr_scheduler import StepLR

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
batch_size = 64
learning_rate = 1e-3
num_epochs = 100

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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
torch.save(model.state_dict(), "pose_estimation_model.pth")
