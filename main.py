# import os
# from preprocess import load_and_preprocess
# from isolation_forest_model import train_isolation_forest
# from autoencoder_model import train_autoencoder
# from visualize import plot_confusion_matrix, plot_roc, plot_mse_distribution
# import pandas as pd

# def main():
#     os.makedirs("models", exist_ok=True)
#     os.makedirs("outputs", exist_ok=True)

#     X, y = load_and_preprocess("data/kddcup.data_10_percent_corrected.csv")

#     print("Training Isolation Forest...")
#     train_isolation_forest(X, y)

#     print("Training Autoencoder...")
#     train_autoencoder(X, y)

#     print("All models trained and metrics saved!")

# if __name__ == "__main__":
#     main()

import os
from preprocess import load_and_preprocess
from isolation_forest_model import train_isolation_forest
from autoencoder_model import train_autoencoder
from visualize import plot_confusion_matrix, plot_roc, plot_mse_distribution
import pandas as pd

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Step 1: Load data
    X, y = load_and_preprocess("data/kddcup.data_10_percent_corrected.csv")

    # Step 2: Train Isolation Forest
    print("Training Isolation Forest...")
    train_isolation_forest(X, y)

    # Step 3: Train Autoencoder
    print("Training Autoencoder...")
    train_autoencoder(X, y)

    # Step 4: Load predictions
    df_iso = pd.read_csv("outputs/predictions_isolation.csv")
    df_ae = pd.read_csv("outputs/predictions_autoencoder.csv")

    # Step 5: Generate Visualizations
    print("Generating visualizations...")

    # Isolation Forest visuals
    plot_confusion_matrix(df_iso['true_label'], df_iso['prediction_label'],
                          "Isolation Forest", "outputs/confusion_isolation.png")
    plot_roc(df_iso['true_label'], df_iso['anomaly_score'],
             "Isolation Forest", "outputs/roc_isolation.png")

    # Autoencoder visuals
    plot_confusion_matrix(df_ae['true_label'], df_ae['prediction_label'],
                          "Autoencoder", "outputs/confusion_autoencoder.png")
    plot_roc(df_ae['true_label'], df_ae['anomaly_score'],
             "Autoencoder", "outputs/roc_autoencoder.png")

    # MSE error distribution for Autoencoder
    threshold = df_ae['anomaly_score'].quantile(0.95)
    plot_mse_distribution(df_ae['anomaly_score'], threshold,
                          "outputs/mse_distribution.png")

    print("All models trained, predictions saved, and visualizations generated!")

if __name__ == "__main__":
    main()

