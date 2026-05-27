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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


#  RESNET feature model method

DATA_DIR = "data/task1_data"
SUBMISSION_DIR = "submissions"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# load data
train_meta = pd.read_csv(f"{DATA_DIR}/train_metadata.csv")
test_meta = pd.read_csv(f"{DATA_DIR}/test_metadata.csv")


# create image dataset class
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


# define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# load pretrained ResNet feature extractor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device)
resnet.eval()


# extract ResNet features
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


print("\nextracting train image features")
X_img = extract_features(train_meta)
y = train_meta["class_id"]

print("\nextracting test image features")
X_test_img = extract_features(test_meta)


# train validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_img,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# build Logistic Regression model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=3000, C=10))
])


# train and evaluate validation data
model.fit(X_train, y_train)
val_preds = model.predict(X_val)
acc = accuracy_score(y_val, val_preds)
print("ResNet feature validation accuracy:", acc)
model.fit(X_img, y)

# predict test data
test_preds = model.predict(X_test_img)

# save
submission = pd.DataFrame({
    "image_id": test_meta["image_id"],
    "class_id": test_preds
})

submission_path = f"{SUBMISSION_DIR}/task1_resnet.csv"

submission.to_csv(submission_path, index=False)


# output
print("\n")
print("MODEL: ResNet18 + Logistic Regression")


print("\nModel details:")
print("feature extractor: pretrained ResNet18")
print("classifier: LogisticRegression")
print("scaling: StandardScaler")
print("features used: ResNet18 image embeddings")

print("\nData summary:")
print("Train metadata shape:", train_meta.shape)
print("Test metadata shape:", test_meta.shape)
print("Final X shape:", X_img.shape)
print("Final X_test shape:", X_test_img.shape)
print("\nLabel counts:")
print(y.value_counts().sort_index())

print("\nValidation accuracy:")
print(acc)

print("\nClassification report:")
print(classification_report(y_val, val_preds))

print("\nConfusion matrix:")
print(confusion_matrix(y_val, val_preds))

print("\nSubmission preview:")
print(submission.head())
print("\n")