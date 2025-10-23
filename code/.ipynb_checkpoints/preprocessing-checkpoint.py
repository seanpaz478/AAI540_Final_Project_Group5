import os
import boto3
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup
from sklearn.model_selection import train_test_split

# Environment variables from ProcessingStep
region = os.environ["AWS_REGION"]
bucket = os.environ["BUCKET"]
session = sagemaker.Session(boto3.session.Session(region_name=region))

# Paths inside container
output_base = "/opt/ml/processing"

prefix = "student-anxiety-xgb"

# Helper to load feature group data
def load_feature_group(fg_name):
    fg = FeatureGroup(name=fg_name, sagemaker_session=session)
    q = fg.athena_query()
    table = q.table_name

    q.run(
        query_string=f'SELECT * FROM "{table}"',
        output_location=f"s3://{bucket}/athena-results/"
    )
    q.wait()
    df = q.as_dataframe()

    # Keep most recent record per student
    df = df.sort_values(["student_id", "event_time"])
    df = df.drop_duplicates(subset=["student_id"], keep="last")

    # Drop metadata columns
    meta_cols = ["write_time", "is_deleted", "api_invocation_time", "event_time"]
    df = df.drop(columns=[c for c in meta_cols if c in df.columns])

    return df

# Loading Feature Groups
demo_df = load_feature_group("student-demographics-ses-fg")
performance_df = load_feature_group("student-performance-fg")
wellbeing_df = load_feature_group("student-wellbeing-fg")
target_df = load_feature_group("student-anxiety-target-fg")

# Merging data
df = demo_df.merge(performance_df, on="student_id")
df = df.merge(wellbeing_df, on="student_id")
df = df.merge(target_df, on="student_id")

# Preprocessing
target_col = "anxiety_level_encoded"
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("missing").astype("category").cat.codes

feature_cols = [c for c in df.columns if c not in [target_col, "student_id"]]

# Train/Val/Test Split
df_main, df_prod = train_test_split(df, test_size=0.4, random_state=0, stratify=df[target_col])
df_train, df_temp = train_test_split(df_main, test_size=0.3333, random_state=0, stratify=df_main[target_col])
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=0, stratify=df_temp[target_col])

# Helper to save both SKLearn and XGBoost CSVs
def save_split(df_split, split_name):
    # SKLearn CSV
    sklearn_dir = os.path.join(output_base, split_name, "sklearn")
    os.makedirs(sklearn_dir, exist_ok=True)
    df_split.to_csv(os.path.join(sklearn_dir, f"{split_name}.csv"), index=False)

    # XGBoost CSV: label first, no header
    xgb_dir = os.path.join(output_base, split_name, "xgb")
    os.makedirs(xgb_dir, exist_ok=True)
    cols = [target_col] + feature_cols
    df_split[cols].to_csv(os.path.join(xgb_dir, f"{split_name}.csv"), index=False, header=False)

# Saving all splits
for name, data in [("train", df_train), ("validation", df_val), ("test", df_test), ("prod", df_prod)]:
    save_split(data, name)

print("Preprocessing complete! All splits written for SKLearn and XGBoost.")