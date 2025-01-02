import os
import random
import shutil

# Paths
data_dir = "fruits"
prediction_dir = "prediction"

# Ensure data exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset folder '{data_dir}' not found!")

# Prepare predictions folder
if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)

# Move 2% of images to prediction folder
for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)
    if os.path.isdir(category_path):
        pred_category_path = os.path.join(prediction_dir, category)
        if not os.path.exists(pred_category_path):
            os.makedirs(pred_category_path)

        images = [f for f in os.listdir(category_path) if f.endswith(('jpg', 'png', 'jpeg'))]
        selected = random.sample(images, max(1, len(images) // 50))  # 2% of images

        for img in selected:
            shutil.copy(os.path.join(category_path, img), os.path.join(pred_category_path, img))
        print(f"Prepared {len(selected)} images for prediction in '{category}'")
