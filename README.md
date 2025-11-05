# 🧠 IDS-Project

A Real-Time Intrusion Detection System (IDS) based on Artificial Intelligence and lightweight Machine Learning models.

## 📘 Project Overview
This project aims to detect and classify network intrusions using the **UNSW-NB15** dataset.  
It includes all stages from data preprocessing to model training and evaluation, with a modular and well-organized architecture.

---


## 📁 Project Structure
```plaintext
IDS-Project/
│
├── data/           # Datasets (CSV, reduced versions, etc.)
│   ├── UNSW-NB15_1.csv
│   ├── UNSW-NB15_2.csv
│   ├── UNSW-NB15_3.csv
│   ├── UNSW-NB15_4.csv
│   └── UNSW-NB15_features.csv
│
├── notebooks/      # Jupyter notebooks (exploration, training, evaluation)
│   ├── 01_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
│
├── src/            # Source code (preprocessing, training, evaluation)
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── realtime_demo.py
│
├── models/         # Saved ML models and scalers
│   ├── model.h5
│   └── scaler.pkl
│
├── results/        # Reports, logs, and visualizations
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
└── requirements.txt  # Python dependencies

```

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

```plaintext
IDS-Project/
│
├── data/
│   ├── UNSW-NB15_1.csv
│   ├── UNSW-NB15_2.csv
│   ├── UNSW-NB15_3.csv
│   ├── UNSW-NB15_4.csv
│   └── UNSW-NB15_features.csv

```

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


