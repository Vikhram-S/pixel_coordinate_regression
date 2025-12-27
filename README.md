\# ML Assignment – Supervised Regression  

\## Pixel Coordinate Prediction Using Deep Learning



---



\## Problem Statement



Using Deep Learning techniques, predict the \*\*(x, y)\*\* coordinates of a pixel with intensity

\*\*255\*\* in a \*\*50×50 grayscale image\*\*, where all other pixels have value \*\*0\*\*.



The pixel with value \*\*255\*\* is randomly assigned for each image.



This problem is formulated as a \*\*supervised regression task\*\*, where the model learns to map

image inputs to continuous spatial coordinates.



---



\## Objective



\- Generate a synthetic dataset of grayscale images

\- Train a deep learning model to predict pixel coordinates

\- Visualize training behavior and prediction quality

\- Ensure clean, readable, and PEP8-compliant code



---



\## Approach Overview



\### Problem Formulation



\- Output space consists of continuous \*\*(x, y)\*\* coordinates

\- Regression is preferred over classification to allow precise localization



\### Dataset Design Rationale



\- Dataset is synthetically generated to:

&nbsp; - Avoid positional bias

&nbsp; - Ensure uniform spatial coverage

&nbsp; - Maintain full reproducibility

\- Each image:

&nbsp; - Resolution: \*\*50×50\*\*

&nbsp; - One pixel set to \*\*255\*\*

&nbsp; - Remaining pixels set to \*\*0\*\*

\- Target coordinates are normalized to \*\*\[0, 1]\*\* for training stability



\### Model Choice



\- A lightweight \*\*Convolutional Neural Network (CNN)\*\* is used

\- CNNs efficiently capture spatial features

\- Model is intentionally simple to emphasize clarity and interpretability



---



\## Project Structure

pixel\_coordinate\_regression/

│

├── src/

│   ├── \_\_init\_\_.py

│   ├── data.py

│   ├── model.py

│   ├── train\_utils.py

│   └── visualization.py

│

├── notebook.ipynb

├── requirements.txt

├── README.md

└── .gitignore



---



\## Installation and Dependencies



\### Requirements



\- Python 3.8 or higher

\- PyTorch

\- NumPy

\- Matplotlib



\### Installation



```

pip install -r requirements.txt


```

\## How to Run



1\. Extract or clone the project directory.



2\. Install the required dependencies:

&nbsp;  ```

&nbsp;  pip install -r requirements.txt



&nbsp;  ```

Open the notebook:



jupyter notebook notebook.ipynb





Run all cells sequentially to:



Generate the dataset



Train the model



Visualize loss curves



Compare predicted and ground truth coordinates



Results



Training and validation loss decrease consistently



Model converges smoothly without overfitting



Final Euclidean Pixel Error ≈ 0.30 pixels



Visualizations



Training vs validation loss curves



Scatter plots of ground truth vs predicted coordinates



Sample image predictions



These plots demonstrate accurate spatial localization.



Evaluation Metrics



Mean Squared Error (MSE) for optimization



Euclidean Pixel Error for interpretability



Limitations



Assumes exactly one active pixel per image



No noise or real-world image artifacts



Dataset is fully synthetic



Future Improvements



Support multiple active pixels



Introduce noise and blur for robustness



Predict heatmaps instead of direct coordinates



Extend to real-world localization tasks





