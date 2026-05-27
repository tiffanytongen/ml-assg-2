import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Random Forest Method

DATA_DIR = "data/task1_data"
SUBMISSION_DIR = "submissions"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# load data
train_meta = pd.read_csv(f"{DATA_DIR}/train_metadata.csv")
test_meta = pd.read_csv(f"{DATA_DIR}/test_metadata.csv")
color = pd.read_csv(f"{DATA_DIR}/color_histogram.csv")
hog = pd.read_csv(f"{DATA_DIR}/hog_pca.csv")
additional = pd.read_csv(f"{DATA_DIR}/additional_features.csv")


# merge features
features = color.merge(hog, on="image_id").merge(additional, on="image_id")
train_data = train_meta.merge(features, on="image_id")
test_data = test_meta.merge(features, on="image_id")


# get x and y
X = train_data.drop(columns=["image_id", "image_path", "class_id", "class_name"])
y = train_data["class_id"]
X_test = test_data.drop(columns=["image_id", "image_path"])


# train validation split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# train random forest
rf_model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_val)
rf_acc = accuracy_score(y_val, rf_preds)


# retrain Random Forest on all training data
rf_model.fit(X, y)


# predict test data
test_preds = rf_model.predict(X_test)

# save
submission = pd.DataFrame({
    "image_id": test_data["image_id"],
    "class_id": test_preds
})
submission_name = f"{SUBMISSION_DIR}/task1_random_forest.csv"
submission.to_csv(submission_name, index=False)



# output
print("\n")
print("MODEL: Random Forest")

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
print("\n")