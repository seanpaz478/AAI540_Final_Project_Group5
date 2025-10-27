import subprocess
import sys

# Install XGBoost if not already installed
subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])

import os
import json
import joblib
import tarfile
import pandas as pd
from sklearn.metrics import f1_score
import xgboost as xgb

# Paths for test data and models
TEST_SKLEARN_PATH = "/opt/ml/processing/test_sklearn/test_sklearn.csv"
TEST_XGB_PATH = "/opt/ml/processing/test_xgb/test_xgb.csv"
SKLEARN_MODEL_DIR = "/opt/ml/processing/sklearn_model"
XGB_MODEL_DIR = "/opt/ml/processing/xgb_model"
OUTPUT_DIR = "/opt/ml/processing/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
METRICS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "evaluation.json")


# Helper: Extract tar.gz if exists
def extract_tar_if_exists(folder_path):
    tar_files = [f for f in os.listdir(folder_path) if f.endswith(".tar.gz")]
    for tar_file in tar_files:
        with tarfile.open(os.path.join(folder_path, tar_file)) as tar:
            tar.extractall(path=folder_path)


# Finding the .joblib file
extract_tar_if_exists(SKLEARN_MODEL_DIR)
sk_files = [f for f in os.listdir(SKLEARN_MODEL_DIR) if f.endswith(".joblib")]
if not sk_files:
    raise FileNotFoundError(f"No SKLearn model (.joblib) found in {SKLEARN_MODEL_DIR}")
sk_model_file = os.path.join(SKLEARN_MODEL_DIR, sk_files[0])
sk_model = joblib.load(sk_model_file)

# Loading SKLearn test set
test_sklearn = pd.read_csv(TEST_SKLEARN_PATH)
y_test_sklearn = test_sklearn["anxiety_level_encoded"]
X_test_sklearn = test_sklearn.drop(columns=["anxiety_level_encoded"])

# Predicting and compute F1
preds_sklearn = sk_model.predict(X_test_sklearn)
f1_sklearn = f1_score(y_test_sklearn, preds_sklearn, average="macro")
print(f"SKLearn Model F1: {f1_sklearn:.4f}")

# Loading XGBoost model
extract_tar_if_exists(XGB_MODEL_DIR)
xgb_files = [f for f in os.listdir(XGB_MODEL_DIR) if f.endswith(".bst")]
if not xgb_files:
    raise FileNotFoundError(f"No XGBoost model (.bst) found in {XGB_MODEL_DIR}")
xgb_model_path = os.path.join(XGB_MODEL_DIR, xgb_files[0])

xgb_model = xgb.Booster()
xgb_model.load_model(xgb_model_path)

# Loading XGBoost test set
test_xgb = pd.read_csv(TEST_XGB_PATH, header=None)
y_test_xgb = test_xgb.iloc[:, 0]
X_test_xgb = test_xgb.iloc[:, 1:]

dtest = xgb.DMatrix(X_test_xgb, label=y_test_xgb)
preds_xgb = xgb_model.predict(dtest)
preds_xgb_labels = preds_xgb.argmax(axis=1)
f1_xgb = f1_score(y_test_xgb, preds_xgb_labels, average="macro")
print(f"XGBoost Model F1: {f1_xgb:.4f}")

# Saving metrics
metrics = {"sklearn_f1": f1_sklearn, "xgb_f1": f1_xgb}
with open(METRICS_OUTPUT_PATH, "w") as f:
    json.dump(metrics, f)

print("Evaluation complete. Metrics saved to:", METRICS_OUTPUT_PATH)