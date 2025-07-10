# 🚨 Anomaly Detection in Network Traffic using Unsupervised Learning

This project applies **Isolation Forest** and **Autoencoder Neural Networks** to detect anomalies in network traffic data, identifying potential security breaches or system malfunctions. We use the classic **KDD Cup 1999 dataset** for training and evaluation.

---

## 📌 Problem Statement

> Using unsupervised learning techniques such as Isolation Forests or Autoencoders to detect unusual patterns or anomalies in network traffic data, which could indicate potential security breaches or system malfunctions.

---

## 📁 Project Structure & File Descriptions

Anomaly_detection_in_network_traffic/
├── data/

├── models/

├── outputs/

├── autoencoder_model.py

├── isolation_forest_model.py

├── preprocess.py

├── visualize.py

├── main.py

├── requirements.txt

├── .gitignore

└── README.md





| File/Folder               | Description |
|---------------------------|-------------|
| `data/`                   | Contains the KDD dataset file (`kddcup.data_10_percent_corrected.csv`). |
| `models/`                 | Stores trained models: `autoencoder_model.h5` and `isolation_forest_model.pkl`. |
| `outputs/`                | Contains prediction results, evaluation metrics, and plots (ROC curves, confusion matrices). |
| `autoencoder_model.py`    | Builds and trains an Autoencoder model, outputs MSE scores and predictions. |
| `isolation_forest_model.py` | Trains the Isolation Forest model and generates anomaly predictions. |
| `preprocess.py`           | Handles data loading, label encoding, and scaling for model input. |
| `visualize.py`            | Contains plotting functions: ROC, confusion matrix, and MSE distribution. |
| `main.py`                 | Orchestrates the full pipeline: preprocessing → training → saving models → generating visual outputs. |
| `requirements.txt`        | Lists all Python dependencies to run the project. |
| `.gitignore`              | Excludes unnecessary files/folders from Git tracking (`venv/`, `models/`, etc.). |
| `README.md`               | This file — provides overview, setup instructions, and documentation. |

---

## 📦 Installation & Setup

### ✅ 1. Create virtual environment (optional but recommended)

```bash
python -m venv venv
# Activate:
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux







