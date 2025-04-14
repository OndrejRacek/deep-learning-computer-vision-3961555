import pandas as pd
import os
import torchvision
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Loading the annotations (CSV)
annotations = pd.read_csv('path_to_annotations/train.csv')

class WheatDataset(Dataset):
    def __init__(self, annotations, img_dir, transform=None):
        self.annotations = annotations
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        boxes = self.annotations.iloc[idx, 1:].values.reshape(-1, 4).astype('float32')
        target = {}
        target["boxes"] = torch.tensor(boxes)
        target["labels"] = torch.ones((len(boxes),), dtype=torch.int64)

        if self.transform:
            image = self.transform(image)
        return image, target

# Define the image transformations
transform = transforms.Compose([transforms.ToTensor()])
dataset = WheatDataset(annotations, img_dir='path_to_images', transform=transform)


# Load pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Modifying the classification head to match the number of classes (1 class for wheat heads + background)
num_classes = 2  # Wheat heads and background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


# Creating a DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Defining optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

# Training loop
for epoch in range(10):  # Train for 10 epochs
    for images, targets in dataloader:
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        print(f'Epoch: {epoch}, Loss: {losses.item()}')

test_image = Image.open('path_to_test_image.jpg')
test_image_tensor = transform(test_image).unsqueeze(0)

# Making predictions
model.eval()  # Setting model to evaluation mode
with torch.no_grad():
    prediction = model(test_image_tensor)

# Visualizing bounding boxes
fig, ax = plt.subplots(1)
ax.imshow(test_image)

for box in prediction[0]['boxes']:
    x, y, w, h = box
    rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()