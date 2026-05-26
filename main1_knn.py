import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load data
train_meta = pd.read_csv("data/task1_data/train_metadata.csv")
test_meta = pd.read_csv("data/task1_data/test_metadata.csv")

color = pd.read_csv("data/task1_data/color_histogram.csv")
hog = pd.read_csv("data/task1_data/hog_pca.csv")
additional = pd.read_csv("data/task1_data/additional_features.csv")

# Combine all provided features
features = color.merge(hog, on="image_id").merge(additional, on="image_id")

# Merge features with metadata
train_data = train_meta.merge(features, on="image_id")
test_data = test_meta.merge(features, on="image_id")

# Create X and y
X = train_data.drop(columns=["image_id", "image_path", "class_id", "class_name"])
y = train_data["class_id"]

X_test = test_data.drop(columns=["image_id", "image_path"])

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# kNN pipeline
# StandardScaler is important because kNN uses distance
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())
])

# Try different kNN settings
param_grid = {
    "model__n_neighbors": [3, 5, 7, 9, 11, 15],
    "model__weights": ["uniform", "distance"],
    "model__metric": ["euclidean", "manhattan"]
}

# Grid search for best kNN model
grid = GridSearchCV(
    knn_pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

# Evaluate best kNN model on validation set
best_knn = grid.best_estimator_
knn_preds = best_knn.predict(X_val)
knn_acc = accuracy_score(y_val, knn_preds)

print("\nBest kNN params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)
print("kNN validation accuracy:", knn_acc)
print(classification_report(y_val, knn_preds))

# Retrain best kNN model on ALL training data
best_knn.fit(X, y)

# Predict test labels
test_preds = best_knn.predict(X_test)

# Create Kaggle submission
submission = pd.DataFrame({
    "image_id": test_data["image_id"],
    "class_id": test_preds
})

submission_name = "submissions/task1_knn.csv"
submission.to_csv(submission_name, index=False)

# print("\nSaved submission to:", submission_name)
# print(submission.head())