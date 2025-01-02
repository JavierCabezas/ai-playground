# Apple Type Classifier Instructions

## Overview
This project classifies different types of apples using a Keras-based deep learning model. The system is designed to:
1. Train a classifier using images organized by type.
2. Prepare a prediction dataset by selecting a random subset of images.
3. Use the trained model to predict the type of apples in the prediction dataset.

---

## Project Setup

### **Folder Structure**
- `fruits/`: Contains subfolders for each apple type.
  - Example:
    ```
    fruits/
    ├── crimson_snow/
    ├── golden/
    ├── pink_lady/
    ```
- `prediction/`: Generated folder for prediction images, structured similarly to `fruits/`.

---

## Docker Setup

### **Dockerfile**
The provided `Dockerfile` manages all dependencies and allows seamless execution of the project.

### **Build Docker Image**
Run the following command to build the Docker image:
```bash
docker build -t apple-classifier .
```

---

## Usage Instructions

### **Step 1: Prepare Prediction Data**
This step selects 2% of images from each fruit type in the `fruits/` folder and moves them to the `prediction/` folder.

Run:
```bash
docker run -it --rm -v "$(pwd)":/app apple-classifier python prepare_predictions.py
```

**Details:**
- For each fruit type in `fruits/`, the script:
  - Randomly selects 2% of images.
  - Copies them to `prediction/<fruit_type>/`.
- Skips folders that already exist in `prediction/`.

### **Step 2: Train the Classifier**
Train the classifier using the dataset in the `fruits/` folder.

Run:
```bash
docker run -it --rm -v "$(pwd)":/app apple-classifier python train_classifier.py
```

**Details:**
- Uses images from the `fruits/` folder.
- Splits data into 80% training and 20% validation.
- Saves the trained model as `apple_classifier.h5`.
- Displays training and validation accuracy/loss curves.

### **Step 3: Predict Using the Model**
Classify images in the `prediction/` folder using the trained model.

Run:
```bash
docker run -it --rm -v "$(pwd)":/app apple-classifier python predict.py
```

**Details:**
- Loads the trained model (`apple_classifier.h5`).
- Predicts the type of each image in the `prediction/` folder.
- Displays the image with its predicted type.

---

## Project Files

### **1. Dockerfile**
Defines the environment and dependencies for the project.

### **2. `train_classifier.py`**
Trains the classifier and saves the trained model.

### **3. `prepare_predictions.py`**
Prepares the `prediction/` folder by selecting 2% of images from each fruit type.

### **4. `predict.py`**
Uses the trained model to classify images in the `prediction/` folder.

---

## Notes
1. Ensure the `fruits/` folder is properly structured before running the scripts.
2. You can add new fruits or categories by creating new subfolders in `fruits/`.
3. Re-run `prepare_predictions.py` if new fruits are added.

---

## Future Enhancements
- Add support for new metrics (e.g., precision, recall).
- Include an optional web interface for predictions.
- Explore transfer learning for improved accuracy.

Let me know if you have any questions or need further assistance!

