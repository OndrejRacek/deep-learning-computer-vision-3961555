import pandas as pd
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class WheatDatasetYOLO(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.image_ids = self.annotations['image_name'].unique()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, image_id)
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # Get all boxes for this image
        records = self.annotations[self.annotations['image_name'] == image_id]
        boxes = []
        for _, row in records.iterrows():
            box_str = row['BoxesString']
            if pd.isna(box_str) or box_str.strip().lower() == 'no_box':
                continue
            for b in box_str.split(';'):
                x1, y1, x2, y2 = map(float, b.strip().split())
                xc = (x1 + x2) / 2 / width
                yc = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                boxes.append([0, xc, yc, w, h])  # class 0 for wheat

        boxes = torch.tensor(boxes, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, boxes

transform = transforms.Compose([transforms.ToTensor()])
dataset = WheatDatasetYOLO(
    annotations_file='/home/vscode/.cache/kagglehub/datasets/vbookshelf/global-wheat-head-dataset-2021/versions/1/gwhd_2021/competition_train.csv',
    img_dir='/home/vscode/.cache/kagglehub/datasets/vbookshelf/global-wheat-head-dataset-2021/versions/1/gwhd_2021/images',
    transform=transform
)

print("Dataset length:", len(dataset))
img, labels = dataset[0]
print("Image shape:", img.shape)
print("YOLO labels:", labels)


import os
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm

# === 1. Paths ===
csv_path = '/home/vscode/.cache/kagglehub/datasets/vbookshelf/global-wheat-head-dataset-2021/versions/1/gwhd_2021/competition_train.csv'
image_dir = '/home/vscode/.cache/kagglehub/datasets/vbookshelf/global-wheat-head-dataset-2021/versions/1/gwhd_2021/images'

output_base = 'datasets/wheat'  # New dataset folder (non-destructive)
os.makedirs(output_base, exist_ok=True)

# Subfolders
for sub in ['images/train', 'labels/train', 'images/val', 'labels/val']:
    os.makedirs(os.path.join(output_base, sub), exist_ok=True)


# === 2. Read annotations & sample 25% ===
df = pd.read_csv(csv_path)
all_image_ids = df['image_name'].unique()
subset_size = int(0.25 * len(all_image_ids))
image_ids = all_image_ids[:subset_size]  # take top 25%

val_split = 0.1
val_count = int(len(image_ids) * val_split)
val_ids = set(image_ids[:val_count])

# === 3. Convert box to YOLO format ===
def convert_box(img_width, img_height, x1, y1, x2, y2):
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return [0, x_center, y_center, width, height]

# === 4. Generate YOLO labels and copy files ===
for img_id in tqdm(image_ids, desc="Converting 25% dataset"):
    label_rows = df[df['image_name'] == img_id]
    img_path = os.path.join(image_dir, img_id)
    if not os.path.exists(img_path): continue

    try:
        with Image.open(img_path) as im:
            w, h = im.size
    except:
        continue

    yolo_lines = []
    for _, row in label_rows.iterrows():
        if pd.isna(row['BoxesString']) or row['BoxesString'].strip().lower() == 'no_box':
            continue
        for box in row['BoxesString'].split(';'):
            x1, y1, x2, y2 = map(float, box.strip().split())
            yolo_box = convert_box(w, h, x1, y1, x2, y2)
            yolo_lines.append(' '.join(map(str, yolo_box)))

    # Choose folder
    subset = 'val' if img_id in val_ids else 'train'
    shutil.copy(img_path, f"{output_base}/images/{subset}/{img_id}")

    # Write label
    label_path = f"{output_base}/labels/{subset}/{img_id.replace('.jpg', '.txt').replace('.png', '.txt')}"
    with open(label_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

# import torch
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Load YOLOv5 model (v5s variant for speed)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # Path to test image
# test_image_path = '/workspaces/deep-learning-computer-vision-3961555/03/test_image/0a68ff4e05268dbe8a94589e38c0574006ff6c5e18b57e838dc9b6411c84cc40.png'
# image = Image.open(test_image_path).convert("RGB")

# # Run inference
# results = model(image)

# # Get predictions as pandas dataframe
# df = results.pandas().xyxy[0]  # xyxy format
# print(df)

# # Visualize
# fig, ax = plt.subplots(1, figsize=(12, 10))
# ax.imshow(image)

# for _, row in df.iterrows():
#     x1, y1, x2, y2, conf, cls, name = row
#     rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
#                              linewidth=2, edgecolor='lime', facecolor='none')
#     ax.add_patch(rect)
#     ax.text(x1, y1 - 5, f'{name} {conf:.2f}', color='lime', fontsize=10, weight='bold')

# plt.axis('off')
# plt.title('YOLOv5 Predictions')

# output_path = "yolov5_prediction_output.png"
# plt.axis('off')
# plt.title('YOLOv5 Predictions')
# plt.savefig(output_path, bbox_inches='tight')
# print(f"✅ Output saved to: {output_path}")