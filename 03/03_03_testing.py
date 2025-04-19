import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf_threshold = 0.5

# 1. Recreate model architecture
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

# Replace the head for 2 classes (background + wheat)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# 2. Load your trained weights (update the path if needed)
model.load_state_dict(torch.load("fasterrcnn_epoch2.pth", map_location=device))
model.to(device)
model.eval()

# 3. Define the same transform used during training
transform = transforms.Compose([transforms.ToTensor()])

# 4. Load test image and transform
test_image_path = '/workspaces/deep-learning-computer-vision-3961555/03/test_image/0a68ff4e05268dbe8a94589e38c0574006ff6c5e18b57e838dc9b6411c84cc40.png'
test_image = Image.open(test_image_path).convert("RGB")
test_image_tensor = transform(test_image).unsqueeze(0).to(device)

# 5. Run prediction
with torch.no_grad():
    prediction = model(test_image_tensor)

# 6. Visualize results
fig, ax = plt.subplots(1, figsize=(12, 10))
ax.imshow(test_image)

for box, score in zip(prediction[0]['boxes'], prediction[0]['scores']):
    if score >= conf_threshold:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

plt.axis("off")
plt.title(f"Predictions (confidence ≥ {conf_threshold})")
#plt.show()
plt.axis("off")
plt.title(f"Predictions (confidence ≥ {conf_threshold})")
plt.savefig("detection_output.png", bbox_inches="tight")
print("✅ Output saved as detection_output.png")
