import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load data
train_meta = pd.read_csv("data/task1_data/train_metadata.csv")
test_meta = pd.read_csv("data/task1_data/test_metadata.csv")

color = pd.read_csv("data/task1_data/color_histogram.csv")
hog = pd.read_csv("data/task1_data/hog_pca.csv")
additional = pd.read_csv("data/task1_data/additional_features.csv")

# Choose ONE feature combo
features = color.merge(hog, on="image_id").merge(additional, on="image_id")
# features = hog.merge(additional, on="image_id")
# features = color.merge(additional, on="image_id")

train_data = train_meta.merge(features, on="image_id")
test_data = test_meta.merge(features, on="image_id")

X = train_data.drop(columns=["image_id", "image_path", "class_id", "class_name"])
y = train_data["class_id"]
X_test = test_data.drop(columns=["image_id", "image_path"])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_val)
rf_acc = accuracy_score(y_val, rf_preds)

print("\nRandom Forest accuracy:", rf_acc)

# Tuned SVM
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC())
])

param_grid = {
    "model__kernel": ["rbf", "linear"],
    "model__C": [0.1, 1, 10, 50, 100],
    "model__gamma": ["scale", "auto"]
}

grid = GridSearchCV(
    svm_pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

best_svm = grid.best_estimator_
svm_preds = best_svm.predict(X_val)
svm_acc = accuracy_score(y_val, svm_preds)

print("\nBest SVM params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)
print("Tuned SVM validation accuracy:", svm_acc)
print(classification_report(y_val, svm_preds))

# Choose best model
if svm_acc > rf_acc:
    print("\nBest model: SVM")
    final_model = grid.best_estimator_
    submission_name = "submissions/task1_svm_submission.csv"
else:
    print("\nBest model: Random Forest")
    final_model = rf_model
    submission_name = "submissions/task1_random_forest_submission.csv"

# Retrain best model on ALL training data
final_model.fit(X, y)

test_preds = final_model.predict(X_test)

submission = pd.DataFrame({
    "image_id": test_data["image_id"],
    "class_id": test_preds
})

submission.to_csv(submission_name, index=False)

print("\nSaved submission to:", submission_name)
print(submission.head())