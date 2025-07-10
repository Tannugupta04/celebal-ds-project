import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_and_preprocess(path):
    col_names = [f"feature_{i}" for i in range(41)] + ["label"]
    df = pd.read_csv(path, names=col_names)

    # Label encode the categorical columns
    categorical_cols = ["feature_1", "feature_2", "feature_3"]
    encoders = {col: LabelEncoder() for col in categorical_cols}
    for col in categorical_cols:
        df[col] = encoders[col].fit_transform(df[col])

    # Label = 'normal.' -> 0, others -> 1
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal.' else 1)

    # Normalize the features
    X = df.drop(columns=['label'])
    y = df['label']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
