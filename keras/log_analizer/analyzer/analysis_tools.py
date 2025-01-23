import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve

# Directory to save charts
CHARTS_DIR = "/app/charts"


def ensure_charts_dir():
    """Ensure the charts directory exists."""
    os.makedirs(CHARTS_DIR, exist_ok=True)


def plot_loss_curves(history):
    """Plot training and validation loss curves."""
    ensure_charts_dir()
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    chart_path = os.path.join(CHARTS_DIR, "loss_curves.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved loss curves to {chart_path}")


def plot_error_distribution(train_errors, val_errors):
    """Plot reconstruction error distribution."""
    ensure_charts_dir()
    plt.figure(figsize=(10, 6))
    plt.hist(train_errors, bins=50, alpha=0.6, label='Training Data')
    plt.hist(val_errors, bins=50, alpha=0.6, label='Validation Data')
    plt.title('Reconstruction Error Histogram')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    chart_path = os.path.join(CHARTS_DIR, "error_distribution.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved error distribution chart to {chart_path}")


def calculate_confusion_matrix(val_errors, threshold):
    """Calculate confusion matrix and classification report."""
    true_labels = (val_errors > threshold).astype(int)
    predicted_labels = (val_errors > threshold).astype(int)

    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    report = classification_report(true_labels, predicted_labels)
    print("\nClassification Report:")
    print(report)


def plot_roc_pr_curves(val_errors, threshold):
    """Plot ROC and Precision-Recall curves."""
    ensure_charts_dir()
    true_labels = (val_errors > threshold).astype(int)

    # ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, val_errors)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    roc_path = os.path.join(CHARTS_DIR, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curve to {roc_path}")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(true_labels, val_errors)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    pr_path = os.path.join(CHARTS_DIR, "precision_recall_curve.png")
    plt.savefig(pr_path)
    plt.close()
    print(f"Saved Precision-Recall curve to {pr_path}")
