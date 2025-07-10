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

### ✅ 1. Create virtual environment

```bash
python -m venv venv
# Activate:
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux


## 📊 Output Files

| File/Plot                            | Description                                         |
|-------------------------------------|-----------------------------------------------------|
| `outputs/predictions_isolation.csv` | Anomaly scores and predictions from Isolation Forest |
| `outputs/predictions_autoencoder.csv` | Anomaly scores and predictions from Autoencoder     |
| `outputs/metrics_isolation_forest.txt` | ROC AUC, confusion matrix, precision/recall         |
| `outputs/metrics_autoencoder.txt`   | ROC AUC, confusion matrix, precision/recall         |
| `outputs/roc_isolation.png`         | ROC curve for Isolation Forest                      |
| `outputs/roc_autoencoder.png`       | ROC curve for Autoencoder                           |
| `outputs/confusion_isolation.png`   | Confusion matrix for Isolation Forest               |
| `outputs/confusion_autoencoder.png` | Confusion matrix for Autoencoder                    |
| `outputs/mse_distribution.png`      | MSE error distribution plot for Autoencoder         |




## 📊 Model Evaluation & Results

We evaluated two unsupervised models on the KDD Cup 1999 network traffic dataset:

| Metric                  | Isolation Forest      | Autoencoder           |
|-------------------------|------------------------|------------------------|
| **ROC AUC Score**       | 0.9459                 | 0.5134                 |
| **Accuracy**            | 0.97                   | 0.24                   |
| **Precision (Anomaly)** | 0.98                   | 0.89                   |
| **Recall (Anomaly)**    | 0.99                   | 0.06                   |
| **F1-score (Anomaly)**  | 0.98                   | 0.10                   |

---

### 📌 Isolation Forest Performance (Best Model)

- **ROC AUC Score**: `0.9459`

#### Confusion Matrix:
[[ 87552 9726]
[ 3222 393521]]


| Class      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| Normal (0) | 0.96      | 0.90   | 0.93     | 97,278  |
| Anomaly (1)| 0.98      | 0.99   | 0.98     | 396,743 |

✅ **Conclusion**: The Isolation Forest model achieved high precision and recall, with a strong ROC AUC score, making it highly effective at distinguishing normal vs anomalous network activity.

---

### 📉 Autoencoder Performance

- **ROC AUC Score**: `0.5134`

#### Confusion Matrix:
[[ 94502 2776]
[374818 21925]]



| Class      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| Normal (0) | 0.20      | 0.97   | 0.33     | 97,278  |
| Anomaly (1)| 0.89      | 0.06   | 0.10     | 396,743 |

⚠️ **Conclusion**: The Autoencoder model struggled to recall anomalies (only 6%) and had a near-random ROC AUC (~0.51), indicating it failed to separate anomalous from normal traffic.

---

### 🏁 Final Verdict

✅ **Isolation Forest is the recommended model** for this task based on all key performance indicators (ROC AUC, precision, recall, and overall accuracy).  
❌ The Autoencoder failed to capture the anomaly patterns effectively in this dataset setup.

---


