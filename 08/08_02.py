class WheatDatasetWithStages(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")

        # Parse boxes
        boxes_str = self.annotations.iloc[idx, 1]
        boxes = [list(map(float, box.split())) for box in boxes_str.split(';')]

        # Parse maturity labels
        labels_str = self.annotations.iloc[idx, 2]
        label_map = {'young': 0, 'mature': 1, 'overripe': 2}
        maturity_labels = [label_map[label.strip()] for label in labels_str.split(';')]

        boxes = torch.tensor(boxes, dtype=torch.float32)
        maturity_labels = torch.tensor(maturity_labels, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        return image, boxes, maturity_labels

import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Update final layer (YOLO layer) to match number of classes (1 object + 3 maturity types)
# e.g., 1 class (wheat head) + 3 categories = 4 total
model.model[-1] = torch.nn.Linear(model.model[-1].in_features, 4)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(3):
    for images, boxes, maturity_labels in train_loader:
        optimizer.zero_grad()

        # Forward pass (you may need to write a wrapper to accept multi-target)
        loss = model(images, boxes, maturity_labels)  # This would need a custom wrapper
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} - Loss: {loss.item()}")

import cv2

def apply_challenging_conditions(image):
    # Blurring
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Dimming
    darker = cv2.convertScaleAbs(image, alpha=0.5, beta=0)
    return [blurred, darker]


cv_image = np.array(pil_image)[:, :, ::-1].copy()  # RGB to BGR
challenged_images = apply_challenging_conditions(cv_image)

model.eval()

for img in [original_image] + challenged_images:
    input_tensor = transform(Image.fromarray(img[:, :, ::-1])).unsqueeze(0)  # back to RGB
    with torch.no_grad():
        predictions = model(input_tensor)
        # Parse predictions, draw boxes, compute metrics like mAP/accuracy