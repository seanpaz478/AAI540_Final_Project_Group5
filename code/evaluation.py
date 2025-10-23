import os
import json
import joblib
import pandas as pd
from sklearn.metrics import f1_score
import xgboost as xgb

# Paths for test data and models
test_sklearn_path = "/opt/ml/processing/test_sklearn/test_sklearn.csv"
test_xgb_path = "/opt/ml/processing/test_xgb/test_xgb.csv"
sklearn_model_path = "/opt/ml/processing/sklearn_model/model.joblib"
xgb_model_path = "/opt/ml/processing/xgb_model/model.bst"
output_dir = "/opt/ml/processing/output"
os.makedirs(output_dir, exist_ok=True)
metrics_output_path = os.path.join(output_dir, "evaluation.json")

# Evaluating SKLearn model

# Loading SKLearn test set
test_sklearn = pd.read_csv(test_sklearn_path)
y_test_sklearn = test_sklearn["anxiety_level_encoded"]
X_test_sklearn = test_sklearn.drop(columns=["anxiety_level_encoded"])

# Loading SKLearn model
sk_model = joblib.load(sklearn_model_path)

# Making predictions and computing F1 score
preds_sklearn = sk_model.predict(X_test_sklearn)
f1_sklearn = f1_score(y_test_sklearn, preds_sklearn, average="macro")
print(f"SKLearn Model F1: {f1_sklearn:.4f}")

# Evaluating XGBoost model

# Loading XGBoost test set
test_xgb = pd.read_csv(test_xgb_path, header=None)
y_test_xgb = test_xgb.iloc[:, 0]
X_test_xgb = test_xgb.iloc[:, 1:]

# Loading XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model(xgb_model_path)

# Preparing DMatrix and predicting
dtest = xgb.DMatrix(X_test_xgb, label=y_test_xgb)
preds_xgb = xgb_model.predict(dtest)
preds_xgb_labels = preds_xgb.argmax(axis=1)

# Computing F1 score
f1_xgb = f1_score(y_test_xgb, preds_xgb_labels, average="macro")
print(f"XGBoost Model F1: {f1_xgb:.4f}")

metrics = {
    "sklearn_f1": f1_sklearn,
    "xgb_f1": f1_xgb
}

with open(metrics_output_path, "w") as f:
    json.dump(metrics, f)

print("Evaluation complete. Metrics saved to:", metrics_output_path)