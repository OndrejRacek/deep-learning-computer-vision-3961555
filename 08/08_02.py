import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
sys.path.append('/workspaces/deep-learning-computer-vision-3961555/yolov5')  
from models.common import DetectMultiBackend

from utils.general import non_max_suppression
import torchvision

class WheatDatasetWithStages(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")

        boxes_str = self.annotations.iloc[idx, 1]
        labels_str = self.annotations.iloc[idx, 2]
        label_map = {'young': 0, 'mature': 1, 'overripe': 2}

        if boxes_str.strip().lower() == "no_box":
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            maturity_labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = [list(map(float, box.split())) for box in boxes_str.split(';')]
            maturity_labels = [label_map[label.strip()] for label in labels_str.split(';')]
            boxes = torch.tensor(boxes, dtype=torch.float32)
            maturity_labels = torch.tensor(maturity_labels, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)
        return image, boxes, maturity_labels

transform = transforms.ToTensor()
csv_path = '/home/vscode/.cache/kagglehub/datasets/vbookshelf/global-wheat-head-dataset-2021/versions/1/gwhd_2021/maturity_train.csv'
img_dir = '/home/vscode/.cache/kagglehub/datasets/vbookshelf/global-wheat-head-dataset-2021/versions/1/gwhd_2021/images'

dataset = WheatDatasetWithStages(csv_path, img_dir, transform=transform)
from torch.utils.data import Subset
dataset = Subset(dataset, list(range(250)))
image, boxes, labels = dataset[0]

print("Image shape:", image.shape)
print("Boxes:", boxes)
print("Maturity labels:", labels)

model = DetectMultiBackend(weights='yolov5s.pt', device='cpu')
detect_layer = model.model.model[-1]  

num_maturity_classes = 3
detect_layer.nc = num_maturity_classes
detect_layer.no = num_maturity_classes + 5  
detect_layer.stride = torch.tensor([8., 16., 32.]) 
detect_layer.anchor_grid = [torch.zeros(1)] * len(detect_layer.anchor_grid)  # reset anchor grid

for i, m in enumerate(detect_layer.m):
    detect_layer.m[i] = torch.nn.Conv2d(
        in_channels=m.in_channels,
        out_channels=detect_layer.no * len(detect_layer.anchors[i]),
        kernel_size=m.kernel_size,
        stride=m.stride,
        padding=m.padding
    )

print("✅ YOLOv5 head modified for maturity classification.")

train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()

for epoch in range(2):
    model.train()
    for i, (images, boxes, maturity_labels) in enumerate(train_loader):
        images = torch.stack(images) 
        dummy_outputs = model(images) 
        target = torch.zeros_like(dummy_outputs[0]) if isinstance(dummy_outputs, (list, tuple)) else torch.zeros_like(dummy_outputs)
        loss = criterion(dummy_outputs[0], target) if isinstance(dummy_outputs, (list, tuple)) else criterion(dummy_outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} - Step {i} - Loss: {loss.item():.4f}")


def apply_challenging_conditions(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    darker = cv2.convertScaleAbs(image, alpha=0.5, beta=0)
    return [blurred, darker]

pil_image = Image.open(dataset.dataset.img_dir + '/' + dataset.dataset.annotations.iloc[0, 0]).convert("RGB")
cv_image = np.array(pil_image)[:, :, ::-1].copy()  
challenged_images = apply_challenging_conditions(cv_image)

def draw_and_save_predictions(img_tensor, pred, output_path, names):
    img = img_tensor.squeeze().permute(1, 2, 0).numpy() * 255
    img = img.astype(np.uint8).copy()

    if pred is not None and len(pred):
        for *xyxy, conf, cls in pred:
            label = f"{names[int(cls)]} {conf:.2f}"
            xyxy = [int(x.item()) for x in xyxy]
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(img, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved: {output_path}")

maturity_names = ['young', 'mature', 'overripe']
model.eval()

for idx, img in enumerate([np.array(pil_image)] + challenged_images):
    rgb_img = img[:, :, ::-1] if img.shape[-1] == 3 else img
    pil_ver = Image.fromarray(rgb_img)
    input_tensor = transforms.ToTensor()(pil_ver).unsqueeze(0)

    with torch.no_grad():
        pred = model(input_tensor)
        pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45)[0]

        draw_and_save_predictions(input_tensor, pred, f"output_{idx}.png", maturity_names)