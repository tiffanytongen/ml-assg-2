import os
import warnings
import pandas as pd
import numpy as np

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

warnings.filterwarnings("ignore")



# Models:
# 1. Logistic Regression
# 2. SVM RBF
# 3. Random Forest
# 4. Voting Ensemble

DATA_DIR = "data/task2_data"
SUBMISSION_DIR = "submissions"

os.makedirs(SUBMISSION_DIR, exist_ok=True)


# load data
train_meta = pd.read_csv(f"{DATA_DIR}/train_metadata.csv")
test_meta = pd.read_csv(f"{DATA_DIR}/test_metadata.csv")
y = train_meta["class_id"]


# load provided features
color = pd.read_csv(f"{DATA_DIR}/color_histogram.csv")
hog = pd.read_csv(f"{DATA_DIR}/hog_pca.csv")
additional = pd.read_csv(f"{DATA_DIR}/additional_features.csv")
features = color.merge(hog, on="image_id").merge(additional, on="image_id")
train_data = train_meta.merge(features, on="image_id")
test_data = test_meta.merge(features, on="image_id")
X_provided = train_data.drop(
    columns=["image_id", "image_path", "class_id", "class_name"]
)
X_test_provided = test_data.drop(
    columns=["image_id", "image_path"]
)


# create image dataset class
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


# ResNet image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# load ResNet feature extractor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device)
resnet.eval()


# extract ResNet features
def extract_resnet_features(df, batch_size=32):
    dataset = ImageDataset(df, DATA_DIR, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_features = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            features = resnet(images)
            all_features.append(features.cpu().numpy())

    return np.vstack(all_features)


print("\nExtracting ResNet train features")
X_resnet = extract_resnet_features(train_meta)

print("Extracting ResNet test features")
X_test_resnet = extract_resnet_features(test_meta)

X_resnet = pd.DataFrame(
    X_resnet,
    columns=[f"resnet_{i}" for i in range(X_resnet.shape[1])]
)

X_test_resnet = pd.DataFrame(
    X_test_resnet,
    columns=[f"resnet_{i}" for i in range(X_test_resnet.shape[1])]
)



# combine provided features and ResNet features
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

# rename columns to avoid duplicate feature names
X.columns = [f"f{i}" for i in range(X.shape[1])]
X_test.columns = [f"f{i}" for i in range(X_test.shape[1])]

print("\nLabel counts:")
print(y.value_counts().sort_index())


# train validation split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# define each model
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=5000,
        C=10,
        class_weight="balanced",
        random_state=42
    ))
])

svm_rbf_model = Pipeline([
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

random_forest_model = RandomForestClassifier(
    n_estimators=800,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

voting_model = VotingClassifier(
    estimators=[
        ("logistic", logistic_model),
        ("svm_rbf", svm_rbf_model),
        ("random_forest", random_forest_model)
    ],
    voting="soft",
    n_jobs=-1
)

models_to_try = {
    "Logistic Regression": logistic_model,
    "SVM RBF": svm_rbf_model,
    "Random Forest": random_forest_model,
    "Voting Ensemble": voting_model
}


# train and evaluate each model
results = []
best_model = None
best_model_name = None
best_accuracy = -1
best_val_preds = None

for model_name, model in models_to_try.items():
    print("\n")
    print(f"MODEL: {model_name}")

    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)

    print("\nValidation accuracy:")
    print(val_acc)

    print("\nClassification report:")
    print(classification_report(y_val, val_preds, zero_division=0))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_val, val_preds))
    print("\n")
    print("\n")
    print("\n")

    results.append({
        "model": model_name,
        "validation_accuracy": val_acc
    })

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        best_model = model
        best_model_name = model_name
        best_val_preds = val_preds


# save result
results_df = pd.DataFrame(results).sort_values(
    by="validation_accuracy",
    ascending=False
)

results_path = f"{SUBMISSION_DIR}/task2_model_results.csv"
results_df.to_csv(results_path, index=False)

print("\n")
print("RESULT SUMMARY")
print(results_df)


print("\nBest model:")
print(best_model_name)

print("\nBest validation accuracy:")
print(best_accuracy)
print("\n")


# retrain best model on all labelled training data
best_model.fit(X, y)


# predict test data
test_preds = best_model.predict(X_test)


# save for submission
submission = pd.DataFrame({
    "image_id": test_meta["image_id"],
    "class_id": test_preds
})

submission_path = f"{SUBMISSION_DIR}/task2_submission.csv"
submission.to_csv(submission_path, index=False)


# output
print("\n")
print("FINAL TASK 2 MODEL")

print("\nBest selected model:")
print(best_model_name)

print("\nBest validation accuracy:")
print(best_accuracy)

print("\nBest model classification report:")
print(classification_report(y_val, best_val_preds, zero_division=0))

print("\nBest model confusion matrix:")
print(confusion_matrix(y_val, best_val_preds))

print("\nSubmission preview:")
print(submission.head())
print("\n")