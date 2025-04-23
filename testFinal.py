import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO
from test import calculate_pck, visualize_keypoints
from trainFinal import StackedHourglassNet

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

    return avg_pck, per_keypoint_pck

if __name__ == '__main__':
    model_path = 'best_hourglass_pretrained_64.pth'
    coco_root = '.'
    anno_file = 'annotations/person_keypoints_val2017.json'
    output_folder = 'evaluation_topo_results'

    avg_pck, per_keypoint_pck = evaluate_topo_model(
        model_path=model_path,
        coco_root=coco_root,
        anno_file=anno_file,
        output_folder=output_folder,
        num_samples=100,
        threshold=0.2
    )
