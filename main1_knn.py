import os
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



# kNN method

DATA_DIR = "data/task1_data"
SUBMISSION_DIR = "submissions"
# in case dont have the folder
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


# x for input feature y target label
X = train_data.drop(columns=["image_id", "image_path", "class_id", "class_name"])
y = train_data["class_id"]
X_test = test_data.drop(columns=["image_id", "image_path"])


# split training data
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# build kNN pipeline
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())
])


# grid search for best kNN parameters
param_grid = {
    "model__n_neighbors": [3, 5, 7, 9, 11, 15],
    "model__weights": ["uniform", "distance"],
    "model__metric": ["euclidean", "manhattan"]
}

grid = GridSearchCV(
    knn_pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)
grid.fit(X_train, y_train)



# evaluate best kNN model on validation data and get best model
best_knn = grid.best_estimator_
knn_preds = best_knn.predict(X_val)
knn_acc = accuracy_score(y_val, knn_preds)

best_knn.fit(X, y)


# predict test data
test_preds = best_knn.predict(X_test)


# save as submission
submission = pd.DataFrame({
    "image_id": test_data["image_id"],
    "class_id": test_preds
})

submission_name = f"{SUBMISSION_DIR}/task1_knn.csv"
submission.to_csv(submission_name, index=False)


# output
print("\n")
print("MODEL: kNN")

print("\nModel details:")
print("best parameters:", grid.best_params_)
print("best CV accuracy:", grid.best_score_)
print("features used: color + HOG + additional")
print("scaling: StandardScaler")

print("\nValidation accuracy:")
print(knn_acc)

print("\nClassification report:")
print(classification_report(y_val, knn_preds))

print("\nConfusion matrix:")
print(confusion_matrix(y_val, knn_preds))

print("\nSubmission preview:")
print(submission.head())
print("\n")