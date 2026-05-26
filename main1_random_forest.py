import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# TASK 1: RANDOM FOREST MODEL
# ============================================================

DATA_DIR = "data/task1_data"
SUBMISSION_DIR = "submissions"

os.makedirs(SUBMISSION_DIR, exist_ok=True)


# ============================================================
# 1. Load data
# ============================================================

train_meta = pd.read_csv(f"{DATA_DIR}/train_metadata.csv")
test_meta = pd.read_csv(f"{DATA_DIR}/test_metadata.csv")

color = pd.read_csv(f"{DATA_DIR}/color_histogram.csv")
hog = pd.read_csv(f"{DATA_DIR}/hog_pca.csv")
additional = pd.read_csv(f"{DATA_DIR}/additional_features.csv")


# ============================================================
# 2. Merge features
# ============================================================

features = color.merge(hog, on="image_id").merge(additional, on="image_id")

train_data = train_meta.merge(features, on="image_id")
test_data = test_meta.merge(features, on="image_id")


# ============================================================
# 3. Prepare X and y
# ============================================================

X = train_data.drop(columns=["image_id", "image_path", "class_id", "class_name"])
y = train_data["class_id"]

X_test = test_data.drop(columns=["image_id", "image_path"])


# ============================================================
# 4. Train-validation split
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. Train Random Forest
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest model...")

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_val)
rf_acc = accuracy_score(y_val, rf_preds)


# ============================================================
# 6. Retrain Random Forest on all training data
# ============================================================

print("\nTraining final Random Forest model on all training data...")

rf_model.fit(X, y)


# ============================================================
# 7. Predict test data
# ============================================================

test_preds = rf_model.predict(X_test)


# ============================================================
# 8. Save submission
# ============================================================

submission = pd.DataFrame({
    "image_id": test_data["image_id"],
    "class_id": test_preds
})

submission_name = f"{SUBMISSION_DIR}/task1_random_forest_submission.csv"

submission.to_csv(submission_name, index=False)


# ============================================================
# 9. Print consistent output
# ============================================================

print("\n" + "=" * 60)
print("MODEL: Random Forest")
print("=" * 60)

print("\nModel details:")
print("n_estimators: 500")
print("random_state: 42")
print("features used: color + HOG + additional")

print("\nValidation accuracy:")
print(rf_acc)

print("\nClassification report:")
print(classification_report(y_val, rf_preds))

print("\nConfusion matrix:")
print(confusion_matrix(y_val, rf_preds))

print("\nSubmission preview:")
print(submission.head())

print("\nSaved submission to:")
print(submission_name)

print("=" * 60)