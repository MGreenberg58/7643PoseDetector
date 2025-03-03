from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Paths to your COCO dataset files
annotation_file = "annotations/person_keypoints_val2017.json"
image_dir = "imgs"

# Load COCO annotations
coco = COCO(annotation_file)

# Get a specific image ID (or you can choose randomly)
image_id = 139

# Load image metadata and annotations
image_metadata = coco.loadImgs(image_id)[0]
annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))

# Load and display the image
image_path = f"{image_dir}/{image_metadata['file_name']}"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")

# Define COCO keypoints and their connections
keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

keypoint_connections = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)
]

# Overlay keypoint annotations
for ann in annotations:
    keypoints = np.array(ann['keypoints']).reshape(-1, 3)
    
    # Plot keypoints
    for i, (x, y, v) in enumerate(keypoints):
        if v > 0:  # Keypoint is visible
            plt.scatter(x, y, c='r', s=25)
    
    # Plot connections
    for connection in keypoint_connections:
        start, end = connection
        if keypoints[start, 2] > 0 and keypoints[end, 2] > 0:
            plt.plot([keypoints[start, 0], keypoints[end, 0]],
                     [keypoints[start, 1], keypoints[end, 1]], 'g-', linewidth=2)

plt.show()
