# import tensorflow as tf
# from tensorflow.keras import layers, models
# from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
# import numpy as np
# import pandas as pd

# def build_autoencoder(input_dim):
#     model = models.Sequential([
#         layers.Input(shape=(input_dim,)),
#         layers.Dense(64, activation='relu'),
#         layers.Dropout(0.2),
#         layers.Dense(32, activation='relu'),
#         layers.Dropout(0.2),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(input_dim, activation='sigmoid')
#     ])
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
#     return model

# def train_autoencoder(X, y):
#     model = build_autoencoder(X.shape[1])
#     history = model.fit(X, X, epochs=30, batch_size=512, shuffle=True, validation_split=0.1, verbose=1)

#     X_pred = model.predict(X)
#     mse = np.mean(np.power(X - X_pred, 2), axis=1)

#     # Use validation loss to set threshold
#     threshold = np.percentile(mse, 95)
#     preds = (mse > threshold).astype(int)

#     report = classification_report(y, preds)
#     roc = roc_auc_score(y, preds)
#     cm = confusion_matrix(y, preds)

#     with open("outputs/metrics_autoencoder.txt", "w") as f:
#         f.write("ROC AUC Score: {:.4f}\n\n".format(roc))
#         f.write("Confusion Matrix:\n{}\n\n".format(cm))
#         f.write(report)

#     model.save("models/autoencoder_model.h5")
# mse_scores = np.mean(np.power(X - X_pred, 2), axis=1)
# df_out = pd.DataFrame({
#     'record_id': range(len(X)),
#     'anomaly_score': mse_scores,
#     'prediction_label': preds,
#     'true_label': y
# })
# df_out.to_csv("outputs/predictions_autoencoder.csv", index=False)


import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd

def build_autoencoder(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(input_dim, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
    return model

def train_autoencoder(X, y):
    model = build_autoencoder(X.shape[1])
    model.fit(X, X, epochs=30, batch_size=512, shuffle=True, validation_split=0.1, verbose=1)

    X_pred = model.predict(X)
    mse_scores = np.mean(np.power(X - X_pred, 2), axis=1)

    # Set threshold based on 95th percentile
    threshold = np.percentile(mse_scores, 95)
    preds = (mse_scores > threshold).astype(int)

    report = classification_report(y, preds)
    roc = roc_auc_score(y, preds)
    cm = confusion_matrix(y, preds)

    with open("outputs/metrics_autoencoder.txt", "w") as f:
        f.write(f"ROC AUC Score: {roc:.4f}\n\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n\n")
        f.write(report)

    model.save("models/autoencoder_model.h5")

    # ✅ Save predictions
    df_out = pd.DataFrame({
        'record_id': range(len(X)),
        'anomaly_score': mse_scores,
        'prediction_label': preds,
        'true_label': y
    })
    df_out.to_csv("outputs/predictions_autoencoder.csv", index=False)
