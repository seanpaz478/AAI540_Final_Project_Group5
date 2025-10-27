import os
import json
import joblib
import tarfile
import pathlib
import pickle
import pandas as pd
from sklearn.metrics import f1_score
import xgboost as xgb

# Paths
TEST_XGB_DIR = "/opt/ml/processing/test"
XGB_MODEL_DIR = "/opt/ml/processing/model"
OUTPUT_DIR = "/opt/ml/processing/evaluation"
pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Extracting XGBoost model if tar.gz exists
xgb_model_tar = os.path.join(XGB_MODEL_DIR, "model.tar.gz")
if os.path.exists(xgb_model_tar):
    print(f"Extracting {xgb_model_tar}...")
    with tarfile.open(xgb_model_tar, "r:gz") as tar:
        tar.extractall(path=XGB_MODEL_DIR)
    print("Extraction complete. Contents:", os.listdir(XGB_MODEL_DIR))

xgb_model_path = os.path.join(XGB_MODEL_DIR, "xgboost-model")
if not os.path.exists(xgb_model_path):
    raise FileNotFoundError(f"No XGBoost model found at {xgb_model_path}")

# Loading XGBoost model
print(f"Loading XGBoost model from {xgb_model_path}...")
xgb_model = pickle.load(open(f"{XGB_MODEL_DIR}/xgboost-model", "rb"))
print("Successfully loaded XGBoost model.")

# Loading XGBoost test set
test_xgb = pd.read_csv(os.path.join(TEST_XGB_DIR, "test.csv"), header=None)
y_test_xgb = test_xgb.iloc[:, 0].to_numpy()
X_test_xgb = test_xgb.iloc[:, 1:].to_numpy()

dtest = xgb.DMatrix(X_test_xgb)
preds_proba = xgb_model.predict(dtest)

# Converting probabilities to class labels
if preds_proba.ndim > 1 and preds_proba.shape[1] > 1:
    preds_labels = preds_proba.argmax(axis=1)
else:
    preds_labels = (preds_proba > 0.5).astype(int)

# Computing F1
f1_xgb = f1_score(y_test_xgb, preds_labels, average="macro")
print(f"XGBoost Model F1: {f1_xgb:.4f}")

# Save metrics
metrics = {
    "XGBoost": {
        "F1": f1_xgb
    },
}

evaluation_path = f"{OUTPUT_DIR}/evaluation.json"
with open(evaluation_path, "w") as f:
    f.write(json.dumps(metrics))

print("Evaluation complete. Metrics saved to:", evaluation_path)