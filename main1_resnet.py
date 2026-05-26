import os
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


DATA_DIR = "data/task1_data"

train_meta = pd.read_csv(f"{DATA_DIR}/train_metadata.csv")
test_meta = pd.read_csv(f"{DATA_DIR}/test_metadata.csv")


class ImageDataset(Dataset):
    def __init__(self, df, data_dir, transform):
        self.df = df
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

resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device)
resnet.eval()


def extract_features(df):
    dataset = ImageDataset(df, DATA_DIR, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_features = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            features = resnet(images)
            all_features.append(features.cpu())

    return torch.cat(all_features).numpy()


print("Extracting train image features...")
X_img = extract_features(train_meta)
y = train_meta["class_id"]

print("Extracting test image features...")
X_test_img = extract_features(test_meta)

X_train, X_val, y_train, y_val = train_test_split(
    X_img,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=3000, C=10))
])

model.fit(X_train, y_train)

val_preds = model.predict(X_val)
acc = accuracy_score(y_val, val_preds)

print("ResNet feature validation accuracy:", acc)
print("\nClassification report:")
print(classification_report(y_val, val_preds))

# ============================================================
# Save confusion matrix diagram
# ============================================================

class_names = [
    "bird", "butterfly", "cat", "deer", "dog",
    "elephant", "frog", "horse", "sheep", "spider"
]

os.makedirs("submissions/figures", exist_ok=True)

cm = confusion_matrix(y_val, val_preds)

fig, ax = plt.subplots(figsize=(8, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    cmap="Blues",
    ax=ax,
    xticks_rotation=45,
    colorbar=False
)

plt.title("Task 1 ResNet + Logistic Regression Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "submissions/figures/task1_resnet_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved confusion matrix to submissions/figures/task1_resnet_confusion_matrix.png")

model.fit(X_img, y)

test_preds = model.predict(X_test_img)

submission = pd.DataFrame({
    "image_id": test_meta["image_id"],
    "class_id": test_preds
})

submission.to_csv("submissions/task1_resnet_submission.csv", index=False)

print("Saved to submissions/task1_resnet_submission.csv")
print(submission.head())
