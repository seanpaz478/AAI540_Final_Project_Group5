import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

if __name__ == "__main__":
    train_dir = os.environ["SM_CHANNEL_TRAIN"]
    val_dir = os.environ["SM_CHANNEL_VAL"]
    model_dir = os.environ["SM_MODEL_DIR"]

    train_df = pd.read_csv(os.path.join(train_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(val_dir, "validation.csv"))

    target_col = "anxiety_level_encoded"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    model = make_pipeline(StandardScaler(with_mean=False),
                          LogisticRegression(max_iter=400, n_jobs=-1, random_state=0))
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average="macro")
    print(f"Validation F1: {f1:.4f}")

    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
