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

import torch

# Load YOLOv5 model (v5s variant for speed)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
