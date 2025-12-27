# ML Assignment – Supervised Regression
## Pixel Coordinate Prediction Using Deep Learning

---

## Problem Statement

Using Deep Learning techniques, predict the **(x, y)** coordinates of a pixel with intensity
**255** in a **50×50 grayscale image**, where all other pixels have value **0**.

The pixel with value **255** is randomly assigned for each image.

This problem is formulated as a **supervised regression task**, where the model learns to map
image inputs to continuous spatial coordinates.

---

## Objective

- Generate a synthetic dataset of grayscale images
- Train a deep learning model to predict pixel coordinates
- Visualize training behavior and prediction quality
- Ensure clean, readable, and PEP8-compliant code

---

## Approach Overview

### Problem Formulation
- Output consists of continuous **(x, y)** coordinates
- Regression enables precise spatial localization

### Dataset Design Rationale
- Synthetic dataset ensures:
  - Uniform pixel distribution
  - No positional bias
  - Full reproducibility
- Each image:
  - Resolution: **50×50**
  - One pixel set to **255**
  - Remaining pixels set to **0**
- Coordinates are normalized to **[0, 1]**

### Model Choice
- Lightweight **Convolutional Neural Network (CNN)**
- Efficient spatial feature extraction
- Simple and interpretable architecture

---

## Project Structure
```
pixel_coordinate_regression/
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train_utils.py
│   └── visualization.py
│
├── notebook.ipynb
├── requirements.txt
└── README.md
```
---

## Installation and Dependencies

### Requirements
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

### Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

1. Extract or clone the project directory.

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```bash
jupyter notebook notebook.ipynb
```

4. Run all cells sequentially to:
- Generate the dataset
- Train the model
- Visualize loss curves
- Compare predicted and ground truth coordinates

---

## Results

- Training and validation loss decrease consistently
- Model converges smoothly without overfitting
- Final **Euclidean Pixel Error ≈ 0.30 pixels**

---

## Visualizations

- Training vs validation loss curves
- Ground truth vs predicted coordinate plots
- Sample image predictions

These plots demonstrate accurate spatial localization.

---

## Evaluation Metrics

- **Mean Squared Error (MSE)** for optimization
- **Euclidean Pixel Error** for interpretability

---

## Limitations

- Assumes exactly one active pixel per image
- No real-world noise or distortions
- Fully synthetic dataset

---

## Future Improvements

- Support multiple active pixels
- Add noise and blur for robustness
- Predict heatmaps instead of direct coordinates
- Extend to real-world localization tasks

---

## Conclusion

This project demonstrates effective spatial localization using deep learning-based
supervised regression, emphasizing clarity, reproducibility, and maintainable design.

---
