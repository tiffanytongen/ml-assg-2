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

# linear svm method

DATA_DIR = "data/task1_data"
SUBMISSION_DIR = "submissions"
os.makedirs(SUBMISSION_DIR, exist_ok=True)


# load data
train_meta = pd.read_csv(f"{DATA_DIR}/train_metadata.csv")
test_meta = pd.read_csv(f"{DATA_DIR}/test_metadata.csv")
color = pd.read_csv(f"{DATA_DIR}/color_histogram.csv")
hog = pd.read_csv(f"{DATA_DIR}/hog_pca.csv")
additional = pd.read_csv(f"{DATA_DIR}/additional_features.csv")



# merge feature
features = color.merge(hog, on="image_id").merge(additional, on="image_id")
train_data = train_meta.merge(features, on="image_id")
test_data = test_meta.merge(features, on="image_id")


# extract extra image features
def extract_image_features(metadata, data_dir):
    all_features = []

    for image_path in metadata["image_path"]:
        full_path = os.path.join(data_dir, image_path)

        img = Image.open(full_path).convert("RGB")
        img = img.resize((64, 64))

        arr = np.asarray(img).astype(np.float32) / 255.0

        # RGB stats
        mean_rgb = arr.mean(axis=(0, 1))
        std_rgb = arr.std(axis=(0, 1))
        min_rgb = arr.min(axis=(0, 1))
        max_rgb = arr.max(axis=(0, 1))

        # centre crop stats
        centre = arr[16:48, 16:48, :]
        centre_mean = centre.mean(axis=(0, 1))
        centre_std = centre.std(axis=(0, 1))

        # brightness stats
        gray = arr.mean(axis=2)
        brightness_features = np.array([
            gray.mean(),
            gray.std(),
            gray.min(),
            gray.max()
        ])

        # simple edge strength
        vertical_edges = np.abs(np.diff(gray, axis=0)).mean()
        horizontal_edges = np.abs(np.diff(gray, axis=1)).mean()
        edge_features = np.array([vertical_edges, horizontal_edges])

        # resized raw pixel features
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



train_image_features = extract_image_features(train_meta, DATA_DIR)
test_image_features = extract_image_features(test_meta, DATA_DIR)

train_image_features.columns = [
    f"img_feat_{i}" for i in range(train_image_features.shape[1])
]

test_image_features.columns = [
    f"img_feat_{i}" for i in range(test_image_features.shape[1])
]



# get X and Y
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


# train validation split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# build Linear SVM model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(C=0.1, max_iter=8000, random_state=42))
])


# train and evaluate validation data
model.fit(X_train, y_train)

val_preds = model.predict(X_val)
accuracy = accuracy_score(y_val, val_preds)



# retrain Linear SVM on all training data
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(C=0.1, max_iter=8000, random_state=42))
])

final_model.fit(X, y)



# predict test data
test_preds = final_model.predict(X_test)


# save
submission = pd.DataFrame({
    "image_id": test_data["image_id"],
    "class_id": test_preds
})
submission_path = f"{SUBMISSION_DIR}/task1_linear_svm.csv"
submission.to_csv(submission_path, index=False)

# output
print("\n")
print("MODEL: Linear SVM")

print("\nModel details:")
print("random_state: 42")
print("features used: color + HOG + additional + extra image features")
print("scaling: StandardScaler")

print("\nData summary:")
print("Final X shape:", X.shape)
print("Final X_test shape:", X_test.shape)
print("\nLabel counts:")
print(y.value_counts().sort_index())

print("\nValidation accuracy:")
print(accuracy)

print("\nClassification report:")
print(classification_report(y_val, val_preds))

print("\nConfusion matrix:")
print(confusion_matrix(y_val, val_preds))

print("\nSubmission preview:")
print(submission.head())
print("\n")