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

        # Getting all boxes for this image
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

csv_path = '/home/vscode/.cache/kagglehub/datasets/vbookshelf/global-wheat-head-dataset-2021/versions/1/gwhd_2021/competition_train.csv'
image_dir = '/home/vscode/.cache/kagglehub/datasets/vbookshelf/global-wheat-head-dataset-2021/versions/1/gwhd_2021/images'

output_base = 'datasets/wheat'  
os.makedirs(output_base, exist_ok=True)

# Subfolders
for sub in ['images/train', 'labels/train', 'images/val', 'labels/val']:
    os.makedirs(os.path.join(output_base, sub), exist_ok=True)


# Reading annotations
df = pd.read_csv(csv_path)
all_image_ids = df['image_name'].unique()
subset_size = int(0.25 * len(all_image_ids))
image_ids = all_image_ids[:subset_size]  # take top 25%

val_split = 0.1
val_count = int(len(image_ids) * val_split)
val_ids = set(image_ids[:val_count])

# Converting box to YOLO format
def convert_box(img_width, img_height, x1, y1, x2, y2):
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return [0, x_center, y_center, width, height]

# Generating YOLO labels and copying files
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

    subset = 'val' if img_id in val_ids else 'train'
    shutil.copy(img_path, f"{output_base}/images/{subset}/{img_id}")

    # Writing label
    label_path = f"{output_base}/labels/{subset}/{img_id.replace('.jpg', '.txt').replace('.png', '.txt')}"
    with open(label_path, 'w') as f:
        f.write('\n'.join(yolo_lines))