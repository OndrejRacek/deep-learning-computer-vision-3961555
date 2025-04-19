import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset

# Loading annotations
annotations = pd.read_csv('datasets/global-wheat-head-dataset/train.csv')

# Dataset class
class WheatDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Loading image and bounding box data
        # Applying transformations if specified
        return image, target


import torch

# Loading pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
from torch.optim import Adam

# Defining optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop example
for epoch in range(10):
    for images, targets in data_loader:
        optimizer.zero_grad()
        loss = model(images, targets)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} - Loss: {loss.item()}")

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Loading and preprocessing test image
test_image = Image.open('datasets/global-wheat-head-dataset/test_image.jpg')
transformed_image = transform(test_image).unsqueeze(0)

# Get model predictions
model.eval()
with torch.no_grad():
    predictions = model(transformed_image)

# Plot the image with bounding boxes
fig, ax = plt.subplots(1)
ax.imshow(test_image)

for box in predictions[0]['boxes']:
    x, y, w, h = box
    rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()
