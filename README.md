# 🧠 IDS-Project

A Real-Time Intrusion Detection System (IDS) based on Artificial Intelligence and lightweight Machine Learning models.

## 📘 Project Overview
This project aims to detect and classify network intrusions using the **UNSW-NB15** dataset.  
It includes all stages from data preprocessing to model training and evaluation, with a modular and well-organized architecture.

---

## 📁 Project Structure

IDS-Project/
│
├── data/                  # Datasets (CSV, reduced versions, etc.)
│   ├── UNSW-NB15_1.csv
│   ├── UNSW-NB15_2.csv
│   ├── UNSW-NB15_3.csv
│   ├── UNSW-NB15_4.csv
│   ├── UNSW-NB15_features.csv
│   └── README.md          # (optional: describe your datasets)
│
├── notebooks/             # Jupyter notebooks (exploration, tests)
│   ├── 01_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
│
├── src/                   # Source code (Python scripts)
│   ├── __init__.py
│   ├── preprocess.py      # Data loading, cleaning, scaling
│   ├── train.py           # Training ANN
│   ├── evaluate.py        # Evaluate model (accuracy, F1, ROC, etc.)
│   ├── predict.py         # Single-sample prediction helper
│   └── realtime_demo.py   # Later: real-time packet capture + detection
│
├── models/                # Saved ML models + scalers
│   ├── model.h5
│   ├── scaler.pkl
│   └── README.md
│
├── results/               # Logs, plots, confusion matrices, reports
│   ├── training_log.txt
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── requirements.txt       # Python dependencies
├── README.md              # Project description
└── .gitignore


---

## ⚙️ Features
- Data loading, cleaning, and feature preprocessing  
- Intelligent handling of IPs, ports, and high-cardinality features  
- ANN model training for binary and multiclass classification  
- Evaluation metrics (Accuracy, Precision, Recall, F1, ROC)  
- Ready for real-time intrusion detection implementation

---

## 📊 Dataset Information
The **UNSW-NB15 dataset** is too large to be included in this repository.  
You can download it manually from the official UNSW Canberra website:

🔗 [UNSW-NB15 Dataset – Official Source](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

After downloading, place the CSV files inside the `data/` folder.

Example structure:
IDS-Project/
└── data/
├── UNSW-NB15_1.csv
├── UNSW-NB15_2.csv
├── UNSW-NB15_3.csv
├── UNSW-NB15_4.csv


---

## 🧰 Technologies Used
- Python 3.x  
- Pandas, NumPy, Scikit-learn  
- TensorFlow / Keras  
- Jupyter Notebooks  

---

## 👨‍💻 Author
Developed by **Fouad Bouharkat**  
Master’s in Software Engineering — 2025

---

## 🚀 Future Work
- Real-time packet capture and live anomaly detection  
- Integration into a desktop or web monitoring platform


