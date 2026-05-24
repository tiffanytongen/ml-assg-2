import os
import warnings
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")


# ============================================================
# TASK 1: COARSE-GRAINED ANIMAL CLASSIFICATION
# Method: Linear SVM with engineered image features
# ============================================================

DATA_FOLDER = "data/task1_data"
SUBMISSION_FOLDER = "submissions"

os.makedirs(SUBMISSION_FOLDER, exist_ok=True)


# ============================================================
# 1. Load CSV files
# ============================================================

train_meta = pd.read_csv(os.path.join(DATA_FOLDER, "train_metadata.csv"))
test_meta = pd.read_csv(os.path.join(DATA_FOLDER, "test_metadata.csv"))

color = pd.read_csv(os.path.join(DATA_FOLDER, "color_histogram.csv"))
hog = pd.read_csv(os.path.join(DATA_FOLDER, "hog_pca.csv"))
additional = pd.read_csv(os.path.join(DATA_FOLDER, "additional_features.csv"))

print("Train metadata shape:", train_meta.shape)
print("Test metadata shape:", test_meta.shape)
print("Color shape:", color.shape)
print("HOG shape:", hog.shape)
print("Additional features shape:", additional.shape)


# ============================================================
# 2. Merge provided features
# ============================================================

features = color.merge(hog, on="image_id")
features = features.merge(additional, on="image_id")

train_data = train_meta.merge(features, on="image_id")
test_data = test_meta.merge(features, on="image_id")

print("Train data shape after merge:", train_data.shape)
print("Test data shape after merge:", test_data.shape)


# ============================================================
# 3. Extract extra image features
# ============================================================

def extract_image_features(metadata, data_folder):
    """
    Extract extra features directly from the raw images.

    Features:
    - mean RGB
    - RGB standard deviation
    - RGB min/max
    - centre crop colour statistics
    - brightness statistics
    - simple edge strength
    - small resized raw pixel features
    """

    all_features = []

    for image_path in metadata["image_path"]:
        full_path = os.path.join(data_folder, image_path)

        img = Image.open(full_path).convert("RGB")
        img = img.resize((64, 64))

        arr = np.asarray(img).astype(np.float32) / 255.0

        # RGB statistics
        mean_rgb = arr.mean(axis=(0, 1))
        std_rgb = arr.std(axis=(0, 1))
        min_rgb = arr.min(axis=(0, 1))
        max_rgb = arr.max(axis=(0, 1))

        # Centre crop statistics
        centre = arr[16:48, 16:48, :]
        centre_mean = centre.mean(axis=(0, 1))
        centre_std = centre.std(axis=(0, 1))

        # Brightness statistics
        gray = arr.mean(axis=2)
        brightness_features = np.array([
            gray.mean(),
            gray.std(),
            gray.min(),
            gray.max()
        ])

        # Simple edge strength
        vertical_edges = np.abs(np.diff(gray, axis=0)).mean()
        horizontal_edges = np.abs(np.diff(gray, axis=1)).mean()
        edge_features = np.array([vertical_edges, horizontal_edges])

        # Small raw pixel features
        small_img = img.resize((8, 8))
        small_pixels = np.asarray(small_img).astype(np.float32).flatten() / 255.0

        feature_vector = np.concatenate([
            mean_rgb,
            std_rgb,
            min_rgb,
            max_rgb,
            centre_mean,
            centre_std,
            brightness_features,
            edge_features,
            small_pixels
        ])

        all_features.append(feature_vector)

    return pd.DataFrame(all_features)


print("\nExtracting extra image features...")

train_image_features = extract_image_features(train_meta, DATA_FOLDER)
test_image_features = extract_image_features(test_meta, DATA_FOLDER)

train_image_features.columns = [
    f"img_feat_{i}" for i in range(train_image_features.shape[1])
]

test_image_features.columns = [
    f"img_feat_{i}" for i in range(test_image_features.shape[1])
]

print("Extra train image features shape:", train_image_features.shape)
print("Extra test image features shape:", test_image_features.shape)


# ============================================================
# 4. Prepare training and test data
# ============================================================

X_provided = train_data.drop(
    columns=["image_id", "image_path", "class_id", "class_name"]
)

X_test_provided = test_data.drop(
    columns=["image_id", "image_path"]
)

X = pd.concat(
    [
        X_provided.reset_index(drop=True),
        train_image_features.reset_index(drop=True)
    ],
    axis=1
)

X_test = pd.concat(
    [
        X_test_provided.reset_index(drop=True),
        test_image_features.reset_index(drop=True)
    ],
    axis=1
)

y = train_data["class_id"]

print("\nFinal X shape:", X.shape)
print("Final X_test shape:", X_test.shape)
print("\nLabel counts:")
print(y.value_counts().sort_index())


# ============================================================
# 5. Train-validation split
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. Build one model: Linear SVM
# ============================================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(C=0.1, max_iter=8000, random_state=42))
])


# ============================================================
# 7. Train and evaluate
# ============================================================

print("\nTraining Linear SVM model...")

model.fit(X_train, y_train)

val_preds = model.predict(X_val)

accuracy = accuracy_score(y_val, val_preds)

print("\nValidation accuracy:", accuracy)
print("\nClassification report:")
print(classification_report(y_val, val_preds))

print("\nConfusion matrix:")
print(confusion_matrix(y_val, val_preds))


# ============================================================
# 8. Train final model on all labelled training data
# ============================================================

print("\nTraining final Linear SVM model on all training data...")

final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(C=0.1, max_iter=8000, random_state=42))
])

final_model.fit(X, y)


# ============================================================
# 9. Predict test data
# ============================================================

print("\nPredicting test labels...")

test_preds = final_model.predict(X_test)


# ============================================================
# 10. Save Kaggle submission
# ============================================================

submission = pd.DataFrame({
    "image_id": test_data["image_id"],
    "class_id": test_preds
})

submission_path = os.path.join(
    SUBMISSION_FOLDER,
    "task1_linear_svm_submission.csv"
)

submission.to_csv(submission_path, index=False)

print("\nSubmission preview:")
print(submission.head())

print("\nSaved submission to:", submission_path)