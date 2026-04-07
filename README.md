# Pneumonia Detection Using Deep Learning (ResNet)

## Overview
This project implements a deep learning pipeline for detecting **pneumonia from chest X-ray images** using **transfer learning with ResNet architectures**. The goal is to classify chest X-ray images into two categories:

- **NORMAL**
- **PNEUMONIA**

The project compares the performance of two convolutional neural network models:

- **ResNet18**
- **ResNet50**

Both models are trained using transfer learning with pretrained ImageNet weights and evaluated on a held-out test dataset.

---

## Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset available on Kaggle.

Dataset Link:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### Dataset Structure

```
chest_xray/
│
├── train/
│   ├── NORMAL
│   └── PNEUMONIA
│
├── val/
│   ├── NORMAL
│   └── PNEUMONIA
│
└── test/
    ├── NORMAL
    └── PNEUMONIA
```

### Dataset Size

| Split | Images |
|------|------|
| Train | 5216 |
| Validation | 16 |
| Test | 624 |

---

## Project Workflow

The project pipeline follows these steps:

1. Download and extract the dataset
2. Perform data preprocessing and augmentation
3. Load pretrained ResNet models
4. Modify the final classification layer
5. Train the models using transfer learning
6. Validate model performance during training
7. Evaluate the models on the test dataset

---

## Models Used

### ResNet18
A lightweight residual neural network with 18 layers. It provides strong performance while requiring fewer computational resources.

### ResNet50
A deeper residual network with 50 layers capable of learning more complex visual representations.

Both models use **ImageNet pretrained weights** and replace the final fully connected layer with a custom classifier for pneumonia detection.

---

## Data Preprocessing and Augmentation

The following preprocessing techniques were applied:

- Image resizing to **224 × 224**
- Random horizontal flipping
- Random rotation
- Color jitter (brightness and contrast)
- Image normalization using **ImageNet statistics**

These techniques help improve model generalization and reduce overfitting.

---

## Training Configuration

| Parameter | Value |
|----------|-------|
| Image Size | 224 |
| Batch Size | 16 |
| Epochs | 4 |
| Optimizer | AdamW |
| Learning Rate | 0.0005 |
| Scheduler | CosineAnnealingLR |
| Loss Function | CrossEntropyLoss |

---

## Training and Validation Loss

![Training and Validation Loss](https://github.com/LipnaRoshmi/Healthcare-Data-Analysis-Disease-Prediction-Models/blob/main/Visualization/Training%20and%20Validation%20Loss%20Comparison.png)

The plot above shows the training and validation loss curves for both models across epochs. These curves help visualize the learning behavior of the models and indicate whether the models are overfitting or generalizing well.

---

## Evaluation Metrics

Model performance was evaluated using the following metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

These metrics provide a comprehensive evaluation of classification performance.

---

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/pneumonia-detection-resnet.git
cd pneumonia-detection-resnet
```

Install required dependencies:

```
pip install -r requirements.txt
```

---

## Running the Project

1. Download the dataset from Kaggle.
2. Extract the dataset into the project directory.
3. Ensure the dataset structure matches the expected format.
4. Run the training script or open the notebook.

Example command:

```
python train.py
```

Alternatively, run the project in **Google Colab or Jupyter Notebook**.

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Google Colab

---

## Future Improvements

Possible improvements for this project include:

- Increasing the validation dataset size
- Hyperparameter tuning
- Cross-validation for better model evaluation
- Deploying the model as a web application
- Integrating the model into a medical decision support system

---

## Author

**Lipna Roshmi**

---

## License

This project is intended for educational and research purposes.
