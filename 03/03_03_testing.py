

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