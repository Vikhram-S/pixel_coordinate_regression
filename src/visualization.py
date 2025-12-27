import matplotlib.pyplot as plt
import numpy as np


def plot_losses(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.show()


def visualize_prediction(image, gt, pred):
    plt.imshow(image, cmap="gray")
    plt.scatter(gt[0], gt[1], c="red", label="Ground Truth")
    plt.scatter(pred[0], pred[1], c="blue", label="Prediction")
    plt.legend()
    plt.title("Coordinate Prediction")
    plt.show()
