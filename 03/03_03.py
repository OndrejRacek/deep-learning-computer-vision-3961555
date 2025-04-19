import pandas as pd
import os
import torchvision
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Loading the annotations (CSV)
annotations = pd.read_csv('/home/vscode/.cache/kagglehub/datasets/vbookshelf/global-wheat-head-dataset-2021/versions/1/gwhd_2021/competition_train.csv')

import ast

class WheatDataset(Dataset):
    def __init__(self, annotations, img_dir, transform=None):
        self.annotations = annotations
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        image_id = row['image_name']
        boxes_str = row['BoxesString']

        img_path = os.path.join(self.img_dir, image_id)
        image = Image.open(img_path).convert("RGB")

        # Parse the custom bounding boxes
        boxes = []
        if pd.notna(boxes_str) and boxes_str.strip() != "" and boxes_str.strip().lower() != "no_box":
            for box in boxes_str.split(";"):
                try:
                    coords = list(map(float, box.strip().split()))
                    if len(coords) == 4:
                        boxes.append(coords)
                except ValueError:
                    continue  # skip malformed box rows

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes_tensor),), dtype=torch.int64)

        target = {"boxes": boxes_tensor, "labels": labels}

        if self.transform:
            image = self.transform(image)
        return image, target


# Define the image transformations
transform = transforms.Compose([transforms.ToTensor()])
dataset = WheatDataset(annotations, 
                       img_dir='/home/vscode/.cache/kagglehub/datasets/vbookshelf/global-wheat-head-dataset-2021/versions/1/gwhd_2021/images', 
                       transform=transform)

print("Number of items in dataset:", len(dataset))

print(annotations.head())
print(annotations.columns)


image, target = dataset[0]
print("Image shape:", image.shape)
print("Bounding boxes:", target["boxes"])
print("Labels:", target["labels"])


# Load pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Modifying the classification head to match the number of classes (1 class for wheat heads + background)
num_classes = 2  # Wheat heads and background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

from torch.utils.data import DataLoader
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Define optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
model.train()

for epoch in range(1, 3):  
    print(f"\nEpoch {epoch}")
    for i, (images, targets) in enumerate(dataloader):
        if i >= 50:  # Limit to 50 batches per epoch
            print(f"Stopping early at step {i} to save time.")
            break

        # Filter out samples with 0 boxes
        filtered_images = []
        filtered_targets = []
        for img, tgt in zip(images, targets):
            if tgt["boxes"].numel() > 0:
                filtered_images.append(img)
                filtered_targets.append(tgt)

        if len(filtered_images) == 0:
            continue

        images = list(img.to(device) for img in filtered_images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in filtered_targets]

        with torch.set_grad_enabled(True):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        if i % 5 == 0:
            print(f"[Epoch {epoch} | Step {i}] Loss: {losses.item():.4f}")

    # ✅ Save model at the end of each epoch
    torch.save(model.state_dict(), f"fasterrcnn_epoch{epoch}.pth")
    print(f"✅ Model saved as fasterrcnn_epoch{epoch}.pth")


# for epoch in range(1, 3):  
#     for i, (images, targets) in enumerate(dataloader):
#         # Filter out samples with 0 boxes
#         filtered_images = []
#         filtered_targets = []
#         for img, tgt in zip(images, targets):
#             if tgt["boxes"].numel() > 0:  # box count > 0
#                 filtered_images.append(img)
#                 filtered_targets.append(tgt)

#         if len(filtered_images) == 0:
#             continue  # Skip this batch if all samples are empty

#         images = list(img.to(device) for img in filtered_images)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in filtered_targets]

#         # Forward pass
#         with torch.set_grad_enabled(True):
#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())

#             optimizer.zero_grad()
#             losses.backward()
#             optimizer.step()

#         # Print progress every few steps
#         if i % 5 == 0:
#             print(f"[Epoch {epoch} | Step {i}] Loss: {losses.item():.4f}")


# test_image = Image.open('path_to_test_image.jpg')
# test_image_tensor = transform(test_image).unsqueeze(0)

# # Making predictions
# model.eval()  # Setting model to evaluation mode
# with torch.no_grad():
#     prediction = model(test_image_tensor)

# # Visualizing bounding boxes
# fig, ax = plt.subplots(1)
# ax.imshow(test_image)

# for box in prediction[0]['boxes']:
#     x, y, w, h = box
#     rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)

# plt.show()


'''
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Path to your test image
test_image_path = '/workspaces/deep-learning-computer-vision-3961555/03/test_image/0a68ff4e05268dbe8a94589e38c0574006ff6c5e18b57e838dc9b6411c84cc40.png'

# Load and transform image
test_image = Image.open(test_image_path).convert("RGB")
test_image_tensor = transform(test_image).unsqueeze(0).to(device)

# Set model to eval mode and make prediction
model.eval()
with torch.no_grad():
    prediction = model(test_image_tensor)

# Visualize results
fig, ax = plt.subplots(1, figsize=(12, 10))
ax.imshow(test_image)

# Optional: confidence threshold
conf_threshold = 0.5

for box, score in zip(prediction[0]['boxes'], prediction[0]['scores']):
    if score >= conf_threshold:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

plt.axis("off")
plt.title(f"Predictions on test image (confidence ≥ {conf_threshold})")
plt.show()
'''