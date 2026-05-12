# 🩺 Lung and Colon Cancer Classification using Custom CNN

> Deep Learning-based histopathological image classification using a custom Convolutional Neural Network (CNN) built with PyTorch and deployed using Streamlit.

---

## 🚀 Live Demo

Try the deployed Streamlit application here:

👉 https://lung-colon-cancer-classification-cnn-ff88reyncba2bzz3dngcrc.streamlit.app/

The application allows users to:
- Upload histopathological tissue images
- Predict lung and colon cancer subtypes
- View prediction confidence scores
- Interactively test the trained CNN model

---

## 📌 Project Overview

This project implements a custom Convolutional Neural Network (CNN) for the classification of histopathological images of lung and colon tissues into five different classes.

The project was developed as part of the Neural Networks and Deep Learning course at the Indian Institute of Technology (IIT) Dharwad.

The model was built entirely from scratch using PyTorch and trained on RGB histopathological images from a publicly available Kaggle dataset.

The complete pipeline includes:
- Data preprocessing and augmentation
- CNN model development
- Model training and validation
- Checkpoint saving
- Performance evaluation
- Streamlit deployment

---

## 🎯 Objectives

- Develop a custom CNN architecture for medical image classification
- Classify histopathological tissue images into five cancer-related classes
- Apply image preprocessing and augmentation techniques
- Evaluate model performance using confusion matrices and classification metrics
- Reduce overfitting using dropout and weight decay
- Deploy the trained model as an interactive web application

---

## 🧬 Dataset

### Dataset Used
Lung and Colon Cancer Histopathological Images

- Total Images: 25,000
- Image Size: 256 × 256 RGB
- Source: Kaggle

### Classes
- Lung Benign
- Lung Adenocarcinoma
- Lung Squamous Cell Carcinoma
- Colon Adenocarcinoma
- Colon Benign

### Dataset Link
https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images

---

## ⚙️ Data Preprocessing

### Training Augmentations
- Resize to 128 × 128
- Random Horizontal Flip
- Random Rotation (±10°)
- Normalization

### Validation / Testing
- Resize to 128 × 128
- Normalization

RGB images were preserved without grayscale conversion to retain tissue staining information important for classification.

---

## 🧠 CNN Architecture

The model consists of:

### Convolutional Base
- 3 Convolutional Blocks
- Batch Normalization
- ReLU Activation
- MaxPooling

### Fully Connected Classifier
- Flatten Layer
- Dense Layers (128 → 64 → 5)
- Dropout Regularization

---

## 🔄 Architecture Flow

text Input Image (128×128×3)         ↓ Conv2D (3 → 16) BatchNorm + ReLU + MaxPool         ↓ Conv2D (16 → 32) BatchNorm + ReLU + MaxPool         ↓ Conv2D (32 → 64) BatchNorm + ReLU + MaxPool         ↓ Flatten         ↓ FC (16384 → 128)         ↓ FC (128 → 64)         ↓ FC (64 → 5) 

### Total Trainable Parameters
~2.1 Million

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- PIL
- Jupyter Notebook

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Batch Size | 64 |
| Epochs | 10 |
| Loss Function | CrossEntropyLoss |
| Device | Apple Metal (MPS) |

---

## 📊 Results

### Final Performance

| Metric | Score |
|---|---|
| Training Accuracy | 96.97% |
| Testing Accuracy | 96.16% |

### Additional Evaluation
- Training & Validation Loss Curves
- Confusion Matrix
- Classification Report
- Precision / Recall / F1-Score

The model demonstrated strong generalization with minimal overfitting.

---

## 📷 Visual Results

### Training vs Validation Loss
Loss Curve

### Model Accuracy
Accuracy

### Confusion Matrix
Confusion Matrix

---

## 🖥️ Streamlit Deployment

The trained model was deployed using Streamlit to provide an interactive web interface for predictions.

### Features
- Upload histopathological images
- Predict cancer subtype
- Display confidence scores
- Visualize prediction probabilities
- Clean medical AI interface

---

## 📂 Repository Structure

text ├── app.py ├── model.py ├── best.pth ├── requirements.txt ├── .gitignore ├── README.md │ ├── images/ │   ├── losscurve.png │   ├── accuracy_bar.png │   └── matrix.png │ ├── EE25MT010_CODE.ipynb ├── EE25MT010_REPORT.pdf └── EE25MT010_Presentation.pdf 

---

## ▶️ Running Locally

### Clone Repository

bash git clone https://github.com/abhinav-singh0/lung-colon-cancer-classification-cnn.git 

### Install Requirements

bash pip install -r requirements.txt 

### Run Streamlit App

bash streamlit run app.py 

---

## 🚀 Future Improvements

Potential future enhancements include:

- Transfer Learning (ResNet / EfficientNet)
- Grad-CAM Visualizations
- Learning Rate Scheduling
- Early Stopping
- Stronger Data Augmentation
- External Dataset Validation
- Explainable AI for medical imaging

---

## 📚 References

- PyTorch Documentation
- Kaggle Histopathology Dataset
- Deep Learning for Medical Image Analysis Literature

---

## 👨‍💻 Author

Abhinav Singh  
M.Tech, Electrical Engineering  
Indian Institute of Technology Dharwad

---

## 🙏 Acknowledgment

Special thanks to the faculty of the Neural Networks and Deep Learning course at IIT Dharwad for their guidance and support throughout this pr
