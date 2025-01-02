import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import json

# Paths
data_dir = "fruits"

# Ensure dataset exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset folder '{data_dir}' not found!")

# Data Generators
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
train_data = datagen.flow_from_directory(data_dir, target_size=(128, 128), subset="training", class_mode="categorical")
val_data = datagen.flow_from_directory(data_dir, target_size=(128, 128), subset="validation", class_mode="categorical")

# Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_data.class_indices), activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(train_data, validation_data=val_data, epochs=5)

# Save class_indices to a JSON file
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)
print("Class indices saved as 'class_indices.json'")

# Save Model
model.save("apple_classifier.h5")
print("Model saved as 'apple_classifier.h5'")

# Plot Training Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("training_accuracy.png")  # Save the plot as an image
plt.close()  # Close the plot to avoid display issues

# Plot Training Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("training_loss.png")  # Save the plot as an image
plt.close()  # Close the plot to avoid display issues
