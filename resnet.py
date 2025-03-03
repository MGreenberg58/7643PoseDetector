import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import numpy as np

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

class COCOKeypointsDataset(Dataset):
    def __init__(self, annotation_file, image_dir):
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = f"{self.image_dir}/{img_info['file_name']}"
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))  # Resize to a fixed size
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Create target heatmaps (simplified, you may want to use Gaussian heatmaps)
        num_keypoints = 17  # COCO has 17 keypoints
        target = np.zeros((num_keypoints, 64, 64), dtype=np.float32)
        
        for ann in anns:
            keypoints = np.array(ann['keypoints']).reshape(-1, 3)
            for i, (x, y, v) in enumerate(keypoints):
                if v > 0:  # Keypoint is visible
                    x = int(x * 64 / image.shape[2])
                    y = int(y * 64 / image.shape[1])
                    if 0 <= x < 64 and 0 <= y < 64:
                        target[i, y, x] = 1
        
        return torch.from_numpy(image), torch.from_numpy(target)

# Hyperparameters
num_keypoints = 17
batch_size = 32
learning_rate = 1e-3
num_epochs = 5

# Create model, dataset, and data loader
model = PoseEstimationModel(num_keypoints)
dataset = COCOKeypointsDataset("annotations/person_keypoints_val2017.json", "val2017")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "pose_estimation_model.pth")
