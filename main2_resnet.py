
import os
import warnings
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

warnings.filterwarnings("ignore")

TASK_FOLDER = "data/task2_data"
OUTPUT_FOLDER = "submissions"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 1. Load metadata
# ============================================================

train_meta = pd.read_csv(os.path.join(TASK_FOLDER, "train_metadata.csv"))
test_meta = pd.read_csv(os.path.join(TASK_FOLDER, "test_metadata.csv"))

print("Train metadata shape:", train_meta.shape)
print("Test metadata shape:", test_meta.shape)

y = train_meta["class_id"]

# ============================================================
# 2. Load provided features
# ============================================================

color = pd.read_csv(os.path.join(TASK_FOLDER, "color_histogram.csv"))
hog = pd.read_csv(os.path.join(TASK_FOLDER, "hog_pca.csv"))
additional = pd.read_csv(os.path.join(TASK_FOLDER, "additional_features.csv"))

features = color.merge(hog, on="image_id").merge(additional, on="image_id")

train_data = train_meta.merge(features, on="image_id")
test_data = test_meta.merge(features, on="image_id")

X_provided = train_data.drop(
    columns=["image_id", "image_path", "class_id", "class_name"]
)

X_test_provided = test_data.drop(
    columns=["image_id", "image_path"]
)

print("Provided train features:", X_provided.shape)
print("Provided test features:", X_test_provided.shape)

# ============================================================
# 3. ResNet18 feature extraction
# ============================================================

class ImageDataset(Dataset):
    def __init__(self, df, data_dir, transform):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.df.iloc[idx]["image_path"])
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device)
resnet.eval()


def extract_resnet_features(df, batch_size=32):
    dataset = ImageDataset(df, TASK_FOLDER, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_features = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            feats = resnet(images)
            all_features.append(feats.cpu().numpy())

    return np.vstack(all_features)


print("\nExtracting ResNet train features...")
X_resnet = extract_resnet_features(train_meta)

print("Extracting ResNet test features...")
X_test_resnet = extract_resnet_features(test_meta)

X_resnet = pd.DataFrame(
    X_resnet,
    columns=[f"resnet_{i}" for i in range(X_resnet.shape[1])]
)

X_test_resnet = pd.DataFrame(
    X_test_resnet,
    columns=[f"resnet_{i}" for i in range(X_test_resnet.shape[1])]
)

print("ResNet train features:", X_resnet.shape)
print("ResNet test features:", X_test_resnet.shape)

# ============================================================
# 4. Combine provided + ResNet features
# ============================================================

X = pd.concat(
    [
        X_provided.reset_index(drop=True),
        X_resnet.reset_index(drop=True)
    ],
    axis=1
)

X_test = pd.concat(
    [
        X_test_provided.reset_index(drop=True),
        X_test_resnet.reset_index(drop=True)
    ],
    axis=1
)

X.columns = [f"f{i}" for i in range(X.shape[1])]
X_test.columns = [f"f{i}" for i in range(X_test.shape[1])]

print("\nFinal X shape:", X.shape)
print("Final X_test shape:", X_test.shape)
print("\nLabel counts:")
print(y.value_counts().sort_index())

# ============================================================
# 5. Train validation split
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# 6. Models
# ============================================================

models_to_try = {}

models_to_try["Logistic Regression"] = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=5000,
        C=10,
        class_weight="balanced",
        random_state=42
    ))
])

models_to_try["SVM RBF"] = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="rbf",
        C=50,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42
    ))
])

models_to_try["SVM Linear"] = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="linear",
        C=1,
        class_weight="balanced",
        probability=True,
        random_state=42
    ))
])

models_to_try["Random Forest"] = RandomForestClassifier(
    n_estimators=800,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

models_to_try["Voting Ensemble"] = VotingClassifier(
    estimators=[
        ("logreg", models_to_try["Logistic Regression"]),
        ("svm_rbf", models_to_try["SVM RBF"]),
        ("rf", models_to_try["Random Forest"])
    ],
    voting="soft",
    n_jobs=-1
)

# ============================================================
# 7. Train and compare
# ============================================================

results = []
best_model = None
best_name = None
best_acc = -1
best_preds = None

for name, model in models_to_try.items():
    print("\n" + "=" * 70)
    print("Training:", name)
    print("=" * 70)

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    print("Validation accuracy:", acc)
    print("\nClassification report:")
    print(classification_report(y_val, preds))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_val, preds))

    results.append({
        "model": name,
        "validation_accuracy": acc
    })

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name
        best_preds = preds

results_df = pd.DataFrame(results).sort_values(
    by="validation_accuracy",
    ascending=False
)

print("\n" + "=" * 70)
print("RESULT SUMMARY")
print("=" * 70)
print(results_df)

results_path = os.path.join(OUTPUT_FOLDER, "task2_resnet_results.csv")
results_df.to_csv(results_path, index=False)
print("\nSaved results to:", results_path)

print("\nBest model:", best_name)
print("Best validation accuracy:", best_acc)

# ============================================================
# 8. Train best model on all training data
# ============================================================

print("\nTraining best model on all labelled Task 2 data...")
best_model.fit(X, y)

print("\nPredicting test data...")
test_preds = best_model.predict(X_test)

# ============================================================
# 9. Save submission
# ============================================================

submission = pd.DataFrame({
    "image_id": test_meta["image_id"],
    "label": test_preds
})

submission_path = os.path.join(OUTPUT_FOLDER, "task2_resnet_improved_submission.csv")
submission.to_csv(submission_path, index=False)

print("\nSaved submission to:", submission_path)
print(submission.head())
