import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train_meta = pd.read_csv("data/task1_data/train_metadata.csv")
test_meta = pd.read_csv("data/task1_data/test_metadata.csv")

color = pd.read_csv("data/task1_data/color_histogram.csv")
hog = pd.read_csv("data/task1_data/hog_pca.csv")
additional = pd.read_csv("data/task1_data/additional_features.csv")

print(train_meta.head())
print(color.head())

features = color.merge(hog, on="image_id")
features = features.merge(additional, on="image_id")

train_data = train_meta.merge(features, on="image_id")
test_data = test_meta.merge(features, on="image_id")

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

X = train_data.drop(columns=["image_id", "image_path", "class_id", "class_name"])
y = train_data["class_id"]

X_test = test_data.drop(columns=["image_id", "image_path"])

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

val_preds = model.predict(X_val)

accuracy = accuracy_score(y_val, val_preds)
print("Validation accuracy:", accuracy)

final_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X, y)

test_preds = final_model.predict(X_test)

submission = pd.DataFrame({
    "image_id": test_data["image_id"],
    "class_id": test_preds
})

submission.to_csv("submissions/task1_random_forest_submission.csv", index=False)

print(submission.head())
print("Saved submission to submissions/task1_random_forest_submission.csv")

# Train final model on ALL training data
final_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X, y)

test_preds = final_model.predict(X_test)

submission = pd.DataFrame({
    "image_id": test_data["image_id"],
    "class_id": test_preds
})

submission.to_csv("submissions/task1_random_forest_submission.csv", index=False)

print(submission.head())
print("Saved submission!")
