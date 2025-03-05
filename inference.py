import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import matplotlib.pyplot as plt

class PoseEstimationModel(nn.Module):
    def __init__(self, num_keypoints):
        super(PoseEstimationModel, self).__init__()
        
        resnet = models.resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        
        self.conv_final = nn.Conv2d(256, num_keypoints, kernel_size=1)
        
    def forward(self, x):
        x = self.backbone(x)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv3(x))
        heatmaps = self.conv_final(x)
        return heatmaps

# Load the trained model
num_keypoints = 17
model = PoseEstimationModel(num_keypoints)
model.load_state_dict(torch.load("pose_estimation_model.pth"))
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to preprocess the input image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = image.transpose(2, 0, 1)
    image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image).unsqueeze(0)

# Function to run inference
def run_inference(image_path):
    input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    return output.squeeze().cpu().numpy()

# Function to draw keypoints on the image
def draw_keypoints(image, keypoints):
    for x, y, confidence in keypoints:
        if confidence > 0.1:  # You can adjust this threshold
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
    return image

# Modify the run_inference function to return both heatmaps and the preprocessed image
def run_inference(image_path):
    input_tensor = preprocess_image(image_path)
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (256, 256))
    
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    return output.squeeze().cpu().numpy(), original_image

# Example usage
image_path = "val2017/000000000139.jpg"
heatmaps, original_image = run_inference(image_path)

# Process the heatmaps to get keypoint locations
keypoints = []
for i in range(num_keypoints):
    heatmap = heatmaps[i]
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    confidence = heatmap[y, x]
    keypoints.append((x * 4, y * 4, confidence))  # Scale coordinates to 256x256

# Draw keypoints on the image
result_image = draw_keypoints(original_image, keypoints)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(result_image)
plt.axis('off')
plt.title('Pose Estimation Result')
plt.show()

print("Predicted keypoints:", keypoints)
