import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json

# Load Model
model = tf.keras.models.load_model("apple_classifier.h5")

# Load class_indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse the mapping of class_indices to get indices-to-classes
class_indices_reversed = {v: k for k, v in class_indices.items()}

# Paths
prediction_dir = "prediction"

# Ensure prediction folder exists
if not os.path.exists(prediction_dir):
    raise FileNotFoundError(f"Prediction folder '{prediction_dir}' not found!")

# Initialize containers for metrics
true_labels = []
predicted_labels = []

# Classify images
for category in os.listdir(prediction_dir):
    category_path = os.path.join(prediction_dir, category)
    if os.path.isdir(category_path):
        print(f"Classifying images in '{category}'...")
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                # Load and preprocess the image
                img = load_img(img_path, target_size=(128, 128))
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                predictions = model.predict(img_array)
                predicted_index = np.argmax(predictions[0])
                predicted_class = class_indices_reversed[predicted_index]

                # Record for metrics
                true_labels.append(category)  # True label from folder name
                predicted_labels.append(predicted_class)  # Predicted label

                # Display prediction
                print(f"Image: {img_name} | True: {category} | Predicted: {predicted_class}")

            except Exception as e:
                print(f"Error processing image {img_name}: {e}")

# Generate metrics
print("\n--- Classification Report ---")
print(classification_report(true_labels, predicted_labels))

# Confusion Matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=list(class_indices.keys()))
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_indices.keys(), yticklabels=class_indices.keys())
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.png")
plt.show()

print("Confusion matrix saved as 'confusion_matrix.png'")
