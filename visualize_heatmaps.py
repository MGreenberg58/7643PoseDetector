import os
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from PIL import Image
import torch

def generate_heatmaps(keypoints, visibility, height, width, sigma=2):
    num_keypoints = keypoints.shape[0]
    heatmaps = np.zeros((num_keypoints, height, width), dtype=np.float32)

    for i, (x, y) in enumerate(keypoints):
        if visibility[i] == 0:
            continue
        x_int = int(x)
        y_int = int(y)

        for dx in range(-3 * sigma, 3 * sigma + 1):
            for dy in range(-3 * sigma, 3 * sigma + 1):
                xx = x_int + dx
                yy = y_int + dy
                if 0 <= xx < width and 0 <= yy < height:
                    heatmaps[i, yy, xx] = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    return heatmaps

def visualize_heatmaps(image, heatmaps, alpha=0.5):
    composite = np.sum(heatmaps, axis=0)
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.imshow(composite, cmap='jet', alpha=alpha)
    plt.title("All Keypoints Heatmap Overlay")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Paths
    coco_root = "."  # Adjust to your dataset root
    image_dir = os.path.join(coco_root, "train2017")
    annotation_file = os.path.join(coco_root, "annotations", "person_keypoints_train2017.json")

    # Load COCO
    coco = COCO(annotation_file)

    # Get all image IDs with exactly one person annotation
    valid_img_ids = []
    for img_id in coco.getImgIds(catIds=1):
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=1, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        if len(anns) == 1:
            keypoints = np.array(anns[0]['keypoints']).reshape(-1, 3)
            if np.sum(keypoints[:, 2] > 0) >= 5:
                valid_img_ids.append(img_id)

    if not valid_img_ids:
        raise RuntimeError("No images found with a single person and at least 5 visible keypoints.")

    # Pick one valid image at random
    img_id = int(np.random.choice(valid_img_ids))  # ensure it's a Python int
    img_info = coco.loadImgs([img_id])[0]
    img_path = os.path.join(image_dir, img_info['file_name'])
    image = Image.open(img_path).convert('RGB')
    width, height = image.size

    # Load keypoints
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=1, iscrowd=False)
    ann = coco.loadAnns(ann_ids)[0]
    kpts = np.array(ann['keypoints']).reshape(-1, 3)
    keypoints = kpts[:, :2]
    visibility = kpts[:, 2] > 0

    # Generate and visualize
    heatmaps = generate_heatmaps(keypoints, visibility, height, width, sigma=5)
    visualize_heatmaps(np.array(image), heatmaps)

if __name__ == "__main__":
    main()
