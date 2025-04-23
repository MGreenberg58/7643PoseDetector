import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO
import torch.nn as nn
from torchvision import transforms, models
from pycocotools.cocoeval import COCOeval
import json
from train import StackedHourglassNet

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

def evaluate_model(model_path, coco_root, anno_file, output_folder, num_samples=30, threshold=0.2):
    """
    Evaluate a trained pose estimation model using PCK metric and visualize results
    
    Args:
        model_path: Path to the saved model weights
        coco_root: Root directory of COCO dataset
        anno_file: Path to annotation file
        output_folder: Folder to save visualization results
        num_samples: Number of sample images to evaluate
        threshold: PCK threshold as a fraction of torso diameter
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the model
    model = ConvolutionalModel(num_keypoints=17)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize COCO API
    coco = COCO(anno_file)
    
    # Get image IDs with annotations
    filtered_img_ids = []
    for img_id in coco.imgs:
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=1, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        
        # Keep only images with exactly one person annotation
        if len(anns) != 1:
            continue
        
        # Check if more than 4 keypoints are visible
        keypoints = np.array(anns[0]['keypoints']).reshape(-1, 3)
        visible_keypoints = np.sum(keypoints[:, 2] > 0)
        if visible_keypoints > 4:
            filtered_img_ids.append(img_id)

    img_ids = filtered_img_ids
    
    # Randomly select images
    if num_samples > len(img_ids):
        num_samples = len(img_ids)
    selected_ids = np.random.choice(img_ids, num_samples, replace=False)
    
    # Track PCK scores
    pck_scores = []
    pck_per_keypoint = np.zeros(17)
    visible_per_keypoint = np.zeros(17)

    print(f"Total image IDs: {len(img_ids)}")
    print(f"First few IDs: {img_ids[:5]}")

    coco_predictions = []

    imgs = coco.loadImgs(selected_ids)
    print(imgs)
    
    for i, img_info in enumerate(imgs):
        # Load image
        img_path = os.path.join(coco_root, 'val2017', img_info['file_name'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, IOError):
            print(f"Error loading image: {img_path}")
            continue
        
        # Get ground truth annotations
        ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=1, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        
        # Find the annotation with the most visible keypoints
        best_ann = None
        max_visible = 0
        for ann in anns:
            keypoints = np.array(ann['keypoints']).reshape(-1, 3)
            visible_keypoints = sum(keypoints[:, 2] > 0)
            if visible_keypoints > max_visible:
                max_visible = visible_keypoints
                best_ann = ann
        
        if best_ann is None:
            continue
        
        # Get ground truth keypoints
        gt_keypoints = np.array(best_ann['keypoints']).reshape(-1, 3)
        
        # Get original image dimensions
        width, height = image.size
        
        # Prepare image for model
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get model prediction
        with torch.no_grad():
            pred_coords = model(img_tensor)[0].cpu().numpy()  # Shape: [17, 2]
        
        # Convert normalized coordinates back to image coordinates
        pred_keypoints = np.zeros((17, 3))
        pred_keypoints[:, 0] = pred_coords[:, 0] * width
        pred_keypoints[:, 1] = pred_coords[:, 1] * height
        pred_keypoints[:, 2] = 1.0  # Set confidence to 1.0
        
        # Calculate PCK for this image
        image_pck, keypoint_correct = calculate_pck(
            pred_keypoints, gt_keypoints, width, height, threshold, return_per_keypoint=True
        )
        pck_scores.append(image_pck)
        
        # Update per-keypoint statistics
        for k in range(17):
            if gt_keypoints[k, 2] > 0:  # If keypoint is visible
                visible_per_keypoint[k] += 1
                if keypoint_correct[k]:
                    pck_per_keypoint[k] += 1
        
        # Visualize results
        visualize_keypoints(
            image, pred_keypoints, gt_keypoints, 
            os.path.join(output_folder, f"result_{i+1}.png"),
            image_pck
        )

        # Format prediction for COCO evaluation
        flattened_kpts = []
        for kp in pred_keypoints:
            x, y = kp[0], kp[1]
            v = 2  # set visibility to 2 (visible) or 1 if unsure
            flattened_kpts.extend([float(x), float(y), v])

        coco_predictions.append({
            "image_id": img_info["id"],
            "category_id": 1,  # person
            "keypoints": [round(x, 2) for x in flattened_kpts],  
            "score": 1.0  # Dummy score unless you have confidence
        })
        
        print(f"Processed image {i+1}/{num_samples}: PCK = {image_pck:.4f}")
    
    avg_pck = 0.0
    per_keypoint_pck = np.zeros(17)

    if pck_scores:
        avg_pck = np.mean(pck_scores)
        print(f"\nAverage PCK@{threshold} across {len(pck_scores)} images: {avg_pck:.4f}")
        
        # Calculate per-keypoint PCK
        per_keypoint_pck = np.zeros(17)
        for k in range(17):
            if visible_per_keypoint[k] > 0:
                per_keypoint_pck[k] = pck_per_keypoint[k] / visible_per_keypoint[k]
        
        # Save detailed results
        with open(os.path.join(output_folder, "pck_results.txt"), "w") as f:
            f.write(f"Average PCK@{threshold}: {avg_pck:.4f}\n\n")
            f.write("Per-keypoint PCK:\n")
            for k in range(17):
                f.write(f"{KEYPOINT_NAMES[k]}: {per_keypoint_pck[k]:.4f}\n")
            f.write("\nPer-image PCK:\n")
            for i, pck in enumerate(pck_scores):
                f.write(f"Image {i+1}: {pck:.4f}\n")

    if coco_predictions:
        # Save predictions to file
        pred_path = os.path.join(output_folder, "coco_keypoints_results.json")
        with open(pred_path, "w") as f:
            json.dump(coco_predictions, f)

        # Load results in COCO format
        coco_dt = coco.loadRes(pred_path)
        coco_eval = COCOeval(coco, coco_dt, iouType='keypoints')

        coco_eval.params.imgIds = [img["id"] for img in imgs]

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()  # Prints AP/AR summary

        metric_names = [
        "AP (OKS)", "AP50", "AP75", "AP (M)", "AP (L)",
        "AR (1)", "AR (10)", "AR (100)", "AR (M)", "AR (L)"
        ]
        # Optional: save AP/AR results
        with open(os.path.join(output_folder, "coco_eval_summary.txt"), "w") as f:
            f.write("COCO Keypoint Evaluation Summary:\n")
            for name, val in zip(metric_names, coco_eval.stats):
                f.write(f"{name}: {val:.4f}\n")
    
    return avg_pck, per_keypoint_pck

def evaluate_model_heatmap(model_path, coco_root, anno_file, output_folder, num_samples=30, threshold=0.2):
    """
    Evaluate a trained pose estimation model using PCK metric and visualize results
    
    Args:
        model_path: Path to the saved model weights
        coco_root: Root directory of COCO dataset
        anno_file: Path to annotation file
        output_folder: Folder to save visualization results
        num_samples: Number of sample images to evaluate
        threshold: PCK threshold as a fraction of torso diameter
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the model
    model = StackedHourglassNet(num_keypoints=17)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize COCO API
    coco = COCO(anno_file)
    
    # Get image IDs with annotations
    filtered_img_ids = []
    for img_id in coco.imgs:
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=1, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        
        # Keep only images with exactly one person annotation
        if len(anns) != 1:
            continue
        
        # Check if more than 4 keypoints are visible
        keypoints = np.array(anns[0]['keypoints']).reshape(-1, 3)
        visible_keypoints = np.sum(keypoints[:, 2] > 0)
        if visible_keypoints > 4:
            filtered_img_ids.append(img_id)

    img_ids = filtered_img_ids
    
    # Randomly select images
    if num_samples > len(img_ids):
        num_samples = len(img_ids)
    selected_ids = np.random.choice(img_ids, num_samples, replace=False)
    
    # Track PCK scores
    pck_scores = []
    pck_per_keypoint = np.zeros(17)
    visible_per_keypoint = np.zeros(17)

    print(f"Total image IDs: {len(img_ids)}")
    print(f"First few IDs: {img_ids[:5]}")

    coco_predictions = []

    imgs = coco.loadImgs(selected_ids)
    print(imgs)
    
    for i, img_info in enumerate(imgs):
        # Load image
        img_path = os.path.join(coco_root, 'val2017', img_info['file_name'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, IOError):
            print(f"Error loading image: {img_path}")
            continue
        
        # Get ground truth annotations
        ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=1, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        
        # Find the annotation with the most visible keypoints
        best_ann = None
        max_visible = 0
        for ann in anns:
            keypoints = np.array(ann['keypoints']).reshape(-1, 3)
            visible_keypoints = sum(keypoints[:, 2] > 0)
            if visible_keypoints > max_visible:
                max_visible = visible_keypoints
                best_ann = ann
        
        if best_ann is None:
            continue
        
        # Get ground truth keypoints
        gt_keypoints = np.array(best_ann['keypoints']).reshape(-1, 3)
        
        # Get original image dimensions
        width, height = image.size
        
        # Prepare image for model
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get model prediction
        with torch.no_grad():
            heatmaps = model(img_tensor)  # Shape: [1, 17, H, W]
            heatmaps = heatmaps.squeeze(0).cpu()  # [17, H, W]

        pred_coords = []
        for j in range(heatmaps.shape[0]):
            hmap = heatmaps[j]
            y, x = divmod(hmap.argmax().item(), hmap.shape[1])
            x = x / hmap.shape[1]
            y = y / hmap.shape[0]
            pred_coords.append([x * width, y * height])
        pred_coords = np.array(pred_coords)
        
        # Convert normalized coordinates back to image coordinates
        pred_keypoints = np.zeros((17, 3))
        pred_keypoints[:, :2] = pred_coords
        pred_keypoints[:, 2] = 1.0  # confidence
        
        # Calculate PCK for this image
        image_pck, keypoint_correct = calculate_pck(
            pred_keypoints, gt_keypoints, width, height, threshold, return_per_keypoint=True
        )
        pck_scores.append(image_pck)
        
        # Update per-keypoint statistics
        for k in range(17):
            if gt_keypoints[k, 2] > 0:  # If keypoint is visible
                visible_per_keypoint[k] += 1
                if keypoint_correct[k]:
                    pck_per_keypoint[k] += 1
        
        # Visualize results
        visualize_keypoints(
            image, pred_keypoints, gt_keypoints, 
            os.path.join(output_folder, f"result_{i+1}.png"),
            image_pck
        )

        # Format prediction for COCO evaluation
        flattened_kpts = []
        for kp in pred_keypoints:
            x, y = kp[0], kp[1]
            v = float(hmap.max().item())
            flattened_kpts.extend([float(x), float(y), v])

        coco_predictions.append({
            "image_id": img_info["id"],
            "category_id": 1,  # person
            "keypoints": [round(x, 2) for x in flattened_kpts],  
            "score": 1.0  # Dummy score unless you have confidence
        })
        
        print(f"Processed image {i+1}/{num_samples}: PCK = {image_pck:.4f}")
    
    avg_pck = 0.0
    per_keypoint_pck = np.zeros(17)

    if pck_scores:
        avg_pck = np.mean(pck_scores)
        print(f"\nAverage PCK@{threshold} across {len(pck_scores)} images: {avg_pck:.4f}")
        
        # Calculate per-keypoint PCK
        per_keypoint_pck = np.zeros(17)
        for k in range(17):
            if visible_per_keypoint[k] > 0:
                per_keypoint_pck[k] = pck_per_keypoint[k] / visible_per_keypoint[k]
        
        # Save detailed results
        with open(os.path.join(output_folder, "pck_results.txt"), "w") as f:
            f.write(f"Average PCK@{threshold}: {avg_pck:.4f}\n\n")
            f.write("Per-keypoint PCK:\n")
            for k in range(17):
                f.write(f"{KEYPOINT_NAMES[k]}: {per_keypoint_pck[k]:.4f}\n")
            f.write("\nPer-image PCK:\n")
            for i, pck in enumerate(pck_scores):
                f.write(f"Image {i+1}: {pck:.4f}\n")

    if coco_predictions:
        # Save predictions to file
        pred_path = os.path.join(output_folder, "coco_keypoints_results.json")
        with open(pred_path, "w") as f:
            json.dump(coco_predictions, f)

        # Load results in COCO format
        coco_dt = coco.loadRes(pred_path)
        coco_eval = COCOeval(coco, coco_dt, iouType='keypoints')

        coco_eval.params.imgIds = [img["id"] for img in imgs]

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()  # Prints AP/AR summary

        metric_names = [
        "AP (OKS)", "AP50", "AP75", "AP (M)", "AP (L)",
        "AR (1)", "AR (10)", "AR (100)", "AR (M)", "AR (L)"
        ]
        # Optional: save AP/AR results
        with open(os.path.join(output_folder, "coco_eval_summary.txt"), "w") as f:
            f.write("COCO Keypoint Evaluation Summary:\n")
            for name, val in zip(metric_names, coco_eval.stats):
                f.write(f"{name}: {val:.4f}\n")
    
    return avg_pck, per_keypoint_pck

def calculate_pck(pred_keypoints, gt_keypoints, img_width, img_height, threshold=0.2, return_per_keypoint=False):
    """
    Calculate Percentage of Correct Keypoints (PCK)
    
    Args:
        pred_keypoints: Predicted keypoints array of shape [17, 3]
        gt_keypoints: Ground truth keypoints array of shape [17, 3]
        img_width: Image width
        img_height: Image height
        threshold: Distance threshold as a fraction of torso diameter
        return_per_keypoint: Whether to return per-keypoint correctness
        
    Returns:
        PCK score (0.0 to 1.0)
        If return_per_keypoint is True, also returns array indicating which keypoints were correct
    """
    # Find torso diameter (distance between shoulders or hips)
    left_shoulder_idx, right_shoulder_idx = 5, 6
    left_hip_idx, right_hip_idx = 11, 12
    
    if (gt_keypoints[left_shoulder_idx, 2] > 0 and gt_keypoints[right_shoulder_idx, 2] > 0):
        # Use shoulder width
        torso_diameter = np.sqrt(
            (gt_keypoints[left_shoulder_idx, 0] - gt_keypoints[right_shoulder_idx, 0])**2 +
            (gt_keypoints[left_shoulder_idx, 1] - gt_keypoints[right_shoulder_idx, 1])**2
        )
    elif (gt_keypoints[left_hip_idx, 2] > 0 and gt_keypoints[right_hip_idx, 2] > 0):
        # Use hip width
        torso_diameter = np.sqrt(
            (gt_keypoints[left_hip_idx, 0] - gt_keypoints[right_hip_idx, 0])**2 +
            (gt_keypoints[left_hip_idx, 1] - gt_keypoints[right_hip_idx, 1])**2
        )
    else:
        # Fallback to image diagonal
        torso_diameter = np.sqrt(img_width**2 + img_height**2) / 4
    
    # Calculate threshold distance
    threshold_distance = threshold * torso_diameter
    
    # Count correct keypoints
    correct_keypoints = 0
    total_keypoints = 0
    keypoint_correct = np.zeros(17, dtype=bool)
    
    for k in range(17):
        if gt_keypoints[k, 2] > 0:  # If keypoint is visible in ground truth
            distance = np.sqrt(
                (pred_keypoints[k, 0] - gt_keypoints[k, 0])**2 +
                (pred_keypoints[k, 1] - gt_keypoints[k, 1])**2
            )
            if distance < threshold_distance:
                correct_keypoints += 1
                keypoint_correct[k] = True
            total_keypoints += 1
    
    pck = correct_keypoints / total_keypoints if total_keypoints > 0 else 0.0
    
    if return_per_keypoint:
        return pck, keypoint_correct
    return pck

def visualize_keypoints(image, pred_keypoints, gt_keypoints, save_path, pck_score=None):
    """
    Visualize predicted and ground truth keypoints
    
    Args:
        image: PIL Image
        pred_keypoints: Predicted keypoints array of shape [17, 3]
        gt_keypoints: Ground truth keypoints array of shape [17, 3]
        save_path: Path to save the visualization
        pck_score: PCK score to display in the title
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(np.array(image))
    
    # Plot ground truth keypoints and skeleton
    for k in range(17):
        if gt_keypoints[k, 2] > 0:  # If keypoint is visible
            plt.scatter(
                gt_keypoints[k, 0], gt_keypoints[k, 1], 
                c='g', s=50, marker='o', label='Ground Truth' if k == 0 else ""
            )
            plt.text(
                gt_keypoints[k, 0] + 5, gt_keypoints[k, 1] + 5, 
                KEYPOINT_NAMES[k], fontsize=8, color='green'
            )
    
    # Plot ground truth skeleton
    for connection in SKELETON_CONNECTIONS:
        start, end = connection
        if gt_keypoints[start, 2] > 0 and gt_keypoints[end, 2] > 0:
            plt.plot(
                [gt_keypoints[start, 0], gt_keypoints[end, 0]],
                [gt_keypoints[start, 1], gt_keypoints[end, 1]],
                'g-', linewidth=2
            )
    
    # Plot predicted keypoints and skeleton
    for k in range(17):
        if gt_keypoints[k, 2] > 0:  # Only plot if ground truth is visible
            plt.scatter(
                pred_keypoints[k, 0], pred_keypoints[k, 1], 
                c='r', s=50, marker='x', label='Prediction' if k == 0 else ""
            )
    
    # Plot predicted skeleton
    for connection in SKELETON_CONNECTIONS:
        start, end = connection
        if gt_keypoints[start, 2] > 0 and gt_keypoints[end, 2] > 0:
            plt.plot(
                [pred_keypoints[start, 0], pred_keypoints[end, 0]],
                [pred_keypoints[start, 1], pred_keypoints[end, 1]],
                'r--', linewidth=1
            )
    
    # Add legend and title
    plt.legend(loc='upper right')
    title = "Predicted vs Ground Truth Keypoints"
    if pck_score is not None:
        title += f" (PCK: {pck_score:.4f})"
    plt.title(title)
    plt.axis('off')
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    model_path = "best_hourglass_pretrained_64.pth"
    coco_root = "."  # Update with your COCO dataset path
    anno_file = "annotations/person_keypoints_val2017.json"
    output_folder = "evaluation_results"
    
    # avg_pck, per_keypoint_pck = evaluate_model(
    #     model_path=model_path,
    #     coco_root=coco_root,
    #     anno_file=anno_file,
    #     output_folder=output_folder,
    #     num_samples=100,
    #     threshold=0.2  # PCK@0.2
    # )

    avg_pck, per_keypoint_pck = evaluate_model_heatmap(
        model_path=model_path,
        coco_root=coco_root,
        anno_file=anno_file,
        output_folder=output_folder,
        num_samples=100,
        threshold=0.2  # PCK@0.2
    )    
    
    # Plot per-keypoint PCK
    plt.figure(figsize=(12, 6))
    plt.bar(KEYPOINT_NAMES, per_keypoint_pck)
    plt.title(f"PCK@0.2 per Keypoint (Average: {avg_pck:.4f})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()