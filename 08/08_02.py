class WheatDatasetWithStages(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __getitem__(self, idx):
        # Loading image and both bounding box and maturity labels
        # Returning image, boxes, and maturity labels
        return image, boxes, maturity_labels


mport torch

# Loading pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Modifying the output layer to include maturity classification (multi-class)
model.model[-1] = torch.nn.Linear(model.model[-1].in_features, num_classes + num_stages)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(10):
    for images, boxes, maturity_labels in train_loader:
        optimizer.zero_grad()
        # Multi task loss function for bounding box and classification accuracy
        loss = model(images, boxes, maturity_labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

import cv2

def apply_challenging_conditions(image):
    # Applying blurring
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Lowering brightness
    darker = cv2.convertScaleAbs(image, alpha=0.5, beta=0)
    return [blurred, darker]

for condition_image in apply_challenging_conditions(test_image):
    results = model(condition_image)
    # Displaying results



model.eval()
for test_image in [original_test_image, blurred_image, darker_image]:
    with torch.no_grad():
        results = model(test_image)
        # Calculating mAP and accuracy
        print(f"mAP: {map_score}, Accuracy: {accuracy}")