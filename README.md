# 🌾 Crop vs Weed Classification using CNN

This project builds a Convolutional Neural Network (CNN) model to classify images as either **crop** or **weed**, helping to automate weed detection in agricultural fields. The trained model achieves high accuracy and can be used to predict new images for real-world applications.

## 📌 Project Overview

- **Goal**: Automatically classify images as *crop* or *weed* to aid precision farming and weed management.
- **Dataset**: Custom image dataset with two categories stored in directories: `crops/` and `weeds/`.
- **Input Image Size**: 150x150 pixels.
- **Model Architecture**: CNN with 3 convolutional layers, followed by fully connected layers.
- **Final Accuracy**: **97.55% on validation data**.
- **Evaluation Metrics**: Precision, Recall, F1-Score.

## 🧠 Model Architecture

```
Input Layer: 150x150x3
↓
Conv2D (32 filters, 3x3) + ReLU
↓
MaxPooling2D
↓
Conv2D (64 filters, 3x3) + ReLU
↓
MaxPooling2D
↓
Conv2D (128 filters, 3x3) + ReLU
↓
MaxPooling2D
↓
Flatten
↓
Dense (256 units) + ReLU
↓
Dense (1 unit) + Sigmoid
```

## 📊 Results

| Metric     | Value |
|------------|-------|
| Accuracy   | 97.55% |
| Precision  | 80%   |
| Recall     | 83%   |
| F1-Score   | 81%   |

## 📈 Visualizations

- Training vs Validation Accuracy  
- Training vs Validation Loss  

(*Shown as matplotlib plots in the notebook*)

## 🧪 Sample Prediction

The model takes an image (e.g., `weed.webp`) and classifies it as:

```
Prediction: Weed
```

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Matplotlib

## 🗂️ Directory Structure

```
project_root/
├── crops/                    # Dataset directory
│   ├── crop/                # Crop images
│   └── weed/                # Weed images
├── mini_project.ipynb       # Jupyter notebook with model code
└── cnn_model.keras          # Saved Keras model
```

## 💾 Model Saving

The trained model is saved as:

```python
model.save("cnn_model.keras")
```

You can later load it for prediction using:

```python
from tensorflow.keras.models import load_model
model = load_model("cnn_model.keras")
```

## 📌 Future Improvements

- Multi-class classification (e.g., distinguishing weed types).
- Use of pretrained models like ResNet or EfficientNet for better accuracy.
- Deployment using Streamlit or Flask for user interaction.
