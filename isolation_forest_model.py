# from sklearn.ensemble import IsolationForest
# from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
# import joblib
# import pandas as pd

# def train_isolation_forest(X, y):
#     # Train only on normal (label=0) data
#     X_train = X[y == 0]

#     # Train model assuming 10% contamination
#     model = IsolationForest(
#         n_estimators=200,
#         max_samples='auto',
#         contamination=0.1,
#         random_state=42,
#         n_jobs=-1
#     )
#     model.fit(X_train)

#     # Predict on entire dataset
#     preds = model.predict(X)
#     preds = [0 if p == 1 else 1 for p in preds]

#     from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
#     report = classification_report(y, preds)
#     roc = roc_auc_score(y, preds)
#     cm = confusion_matrix(y, preds)

#     with open("outputs/metrics_isolation_forest.txt", "w") as f:
#         f.write("ROC AUC Score: {:.4f}\n\n".format(roc))
#         f.write("Confusion Matrix:\n{}\n\n".format(cm))
#         f.write(report)

#     import joblib
#     joblib.dump(model, "models/isolation_forest_model.pkl")
# scores = model.decision_function(X)  # Higher = more normal
# df_out = pd.DataFrame({
#     'record_id': range(len(X)),
#     'anomaly_score': scores,
#     'prediction_label': preds,
#     'true_label': y
# })
# df_out.to_csv("outputs/predictions_isolation.csv", index=False)

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import joblib

def train_isolation_forest(X, y):
    # Train only on normal data
    X_train = X[y == 0]

    model = IsolationForest(
        n_estimators=200,
        max_samples='auto',
        contamination=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)

    # Predict on full dataset
    preds = model.predict(X)
    preds = [0 if p == 1 else 1 for p in preds]

    # Evaluation
    report = classification_report(y, preds)
    roc = roc_auc_score(y, preds)
    cm = confusion_matrix(y, preds)

    with open("outputs/metrics_isolation_forest.txt", "w") as f:
        f.write(f"ROC AUC Score: {roc:.4f}\n\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n\n")
        f.write(report)

    joblib.dump(model, "models/isolation_forest_model.pkl")

    # ✅ Save predictions + scores
    scores = model.decision_function(X)  # Higher = more normal
    df_out = pd.DataFrame({
        'record_id': range(len(X)),
        'anomaly_score': scores,
        'prediction_label': preds,
        'true_label': y
    })
    df_out.to_csv("outputs/predictions_isolation.csv", index=False)
