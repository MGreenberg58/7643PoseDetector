import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO
from test import calculate_pck, visualize_keypoints
from trainFinal import KeypointCoordinateRegressor
from trainHeatmap import StackedHourglassNet
from pycocotools.cocoeval import COCOeval
import json
import matplotlib.pyplot as plt

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

def visualize_heatmaps(heatmaps, save_dir, image_id):
    """
    Visualize and save all keypoint heatmaps.
    
    Args:
        heatmaps: np.ndarray of shape (17, H, W)
        save_dir: directory to save the heatmaps
        image_id: unique identifier for this image
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_keypoints = heatmaps.shape[0]
    for k in range(num_keypoints):
        plt.figure(figsize=(3, 3))
        plt.imshow(heatmaps[k], cmap='hot')  # Use 'gray' or 'hot' or 'jet'
        plt.axis('off')
        plt.title(KEYPOINT_NAMES[k])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{image_id}_heatmap_{KEYPOINT_NAMES[k]}.png"))
        plt.close()

def evaluate_topo_model(model_path, coco_root, anno_file, output_folder, num_samples=30, threshold=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_folder, exist_ok=True)

    model = StackedHourglassNet(num_keypoints=17)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    coco = COCO(anno_file)
    filtered_img_ids = []
    for img_id in coco.imgs:
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=1, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        if len(anns) != 1:
            continue
        keypoints = np.array(anns[0]['keypoints']).reshape(-1, 3)
        visible_keypoints = np.sum(keypoints[:, 2] > 0)
        if visible_keypoints > 4:
            filtered_img_ids.append(img_id)

    img_ids = filtered_img_ids
    selected_ids = np.random.choice(img_ids, min(num_samples, len(img_ids)), replace=False)

    pck_scores = []
    pck_per_keypoint = np.zeros(17)
    visible_per_keypoint = np.zeros(17)

    coco_predictions = []

    imgs = coco.loadImgs(selected_ids)
    for i, img_info in enumerate(imgs):
        img_path = os.path.join(coco_root, 'val2017', img_info['file_name'])
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, IOError):
            continue

        ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=1, iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        best_ann = max(anns, key=lambda a: np.sum(np.array(a['keypoints']).reshape(-1, 3)[:, 2] > 0))
        gt_keypoints = np.array(best_ann['keypoints']).reshape(-1, 3)

        width, height = image.size
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            heatmaps = model(img_tensor)[-1]  # Get last stack heatmaps
            heatmaps = heatmaps.squeeze(0).cpu().numpy()  # (17, H, W)

            coords = np.zeros((17, 2))
            for k in range(17):
                y, x = np.unravel_index(np.argmax(heatmaps[k]), heatmaps[k].shape)
                coords[k] = [x / heatmaps[k].shape[1], y / heatmaps[k].shape[0]]  # Normalize

        pred_keypoints = np.zeros((17, 3))
        pred_keypoints[:, 0] = coords[:, 0] * width
        pred_keypoints[:, 1] = coords[:, 1] * height
        pred_keypoints[:, 2] = 1.0

        image_pck, keypoint_correct = calculate_pck(pred_keypoints, gt_keypoints, width, height, threshold, return_per_keypoint=True)
        pck_scores.append(image_pck)

        for k in range(17):
            if gt_keypoints[k, 2] > 0:
                visible_per_keypoint[k] += 1
                if keypoint_correct[k]:
                    pck_per_keypoint[k] += 1

        visualize_keypoints(
            image, pred_keypoints, gt_keypoints,
            os.path.join(output_folder, f"result_{i+1}.png"),
            image_pck
        )

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

    avg_pck = np.mean(pck_scores) if pck_scores else 0.0
    per_keypoint_pck = np.zeros(17)
    for k in range(17):
        if visible_per_keypoint[k] > 0:
            per_keypoint_pck[k] = pck_per_keypoint[k] / visible_per_keypoint[k]

    print(f"\nAverage PCK@{threshold} across {len(pck_scores)} images: {avg_pck:.4f}")
    with open(os.path.join(output_folder, "pck_results_topo.txt"), "w") as f:
        f.write(f"Average PCK@{threshold}: {avg_pck:.4f}\n\n")
        for k in range(17):
            f.write(f"{KEYPOINT_NAMES[k]}: {per_keypoint_pck[k]:.4f}\n")

    
    pred_path = os.path.join(output_folder, "coco_keypoints_results.json")
    with open(pred_path, "w") as f:
        json.dump(coco_predictions, f)

    coco_dt = coco.loadRes(pred_path)
    coco_eval = COCOeval(coco, coco_dt, iouType='keypoints')
    coco_eval.params.imgIds = [img["id"] for img in imgs]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metric_names = [
        "AP (OKS)", "AP50", "AP75", "AP (M)", "AP (L)",
        "AR (1)", "AR (10)", "AR (100)", "AR (M)", "AR (L)"
    ]
    with open(os.path.join(output_folder, "coco_eval_summary.txt"), "w") as f:
        f.write("COCO Keypoint Evaluation Summary:\n")
        for name, val in zip(metric_names, coco_eval.stats):
            f.write(f"{name}: {val:.4f}\n")

    return avg_pck, per_keypoint_pck

def evaluate_model(model_path, coco_root, anno_file, output_folder, num_samples=30, threshold=0.2, isHeatmap=False):
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
        if not isHeatmap:
            with torch.no_grad():
                pred_coords = model(img_tensor)[0].cpu().numpy()  # Shape: [17, 2]

            pred_keypoints = np.zeros((17, 3))
            pred_keypoints[:, 0] = pred_coords[:, 0] * width
            pred_keypoints[:, 1] = pred_coords[:, 1] * height
            pred_keypoints[:, 2] = 1.0  # dummy confidence

        else:
            with torch.no_grad():
                heatmaps = model(img_tensor)[-1]  # [1, 17, H, W]
                heatmaps = heatmaps.squeeze(0).cpu().numpy()  # [17, H, W]

            visualize_heatmaps(heatmaps, os.path.join(output_folder, "heatmaps"), img_info['id'])

            coords = np.zeros((17, 2))
            for k in range(17):
                y, x = np.unravel_index(np.argmax(heatmaps[k]), heatmaps[k].shape)
                coords[k] = [x / heatmaps[k].shape[1], y / heatmaps[k].shape[0]]  # normalized

            pred_keypoints = np.zeros((17, 3))
            pred_keypoints[:, 0] = coords[:, 0] * width
            pred_keypoints[:, 1] = coords[:, 1] * height
            pred_keypoints[:, 2] = 1.0  

    for k in range(17):
        
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

if __name__ == '__main__':
    model_path = 'hourglass_epoch4.pth'
    coco_root = '.'
    anno_file = 'annotations/person_keypoints_val2017.json'
    output_folder = 'evaluation_topo_results'

    # avg_pck, per_keypoint_pck = evaluate_topo_model(
    #     model_path=model_path,
    #     coco_root=coco_root,
    #     anno_file=anno_file,
    #     output_folder=output_folder,
    #     num_samples=100,
    #     threshold=0.2
    # )

    avg_pck, per_keypoint_pck = evaluate_model(
        model_path=model_path,
        coco_root=coco_root,
        anno_file=anno_file,
        output_folder=output_folder,
        num_samples=100,
        threshold=0.2,  # PCK@0.2
        isHeatmap=True
    )
