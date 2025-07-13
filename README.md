# 🌿 Plant Disease Detection using Deep Learning

This project uses deep learning techniques (CNN and Transfer Learning) to detect plant diseases from leaf images using the PlantVillage dataset.

## 🔍 Problem
Automatically classify plant leaf images based on their disease type (or healthy).

## 📦 Dataset
- Source: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- 15 classes (diseases and healthy)

## 🧠 Models
- CNN built from scratch
- ResNet18 with transfer learning
- Improvements: Weighted Cross-Entropy Loss to handle class imbalance

## 📊 Evaluation
- Accuracy, Confusion Matrix, Multi-class ROC Curves
- CNN Accuracy: ~94.55%
- ResNet Accuracy: Comparable, with faster convergence

## 🛠️ Tech Stack
- Python
- PyTorch
- Matplotlib, Scikit-learn
- LaTeX for documentation

## 📁 Project Structure
plant-disease-detection/
│
├── data/                        # Local dataset (NOT pushed to GitHub)
│   └── PlantVillage/
│
├── outputs/                     # Trained models, saved figures (ignored via .gitignore)
│   ├── models/                  # Saved .pth weights
│   └── figures/                 # Confusion matrices, ROC curves, etc.
│
├── src/                         # Source code modules
│   ├── model.py                 # CNN and ResNet18 architectures
│   ├── train.py                 # Training function
│   └── data.py                  # Data loading and preprocessing
│
├── notebooks/                   # Jupyter notebooks for experimentation
│   ├── train_cnn.ipynb          # CNN from scratch
│   └── train_resnet.ipynb       # Transfer learning (ResNet)
│
├── report/                      # Documentation (LaTeX compiled PDF only)
│   └── plant_disease_report.pdf
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignore datasets, logs, outputs, etc.
├── README.md                    # Project overview and usage
└── LICENSE                      # (Optional) project license


## 📄 Documentation
Full project documentation written in LaTeX is located in the `report/` folder.

## 🔗 Author
Made by AbdelKarim medouse. Feel free to open issues or suggest improvements.
