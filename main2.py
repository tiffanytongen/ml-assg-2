import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# ============================================================
# TASK 2: FINE-GRAINED BIRD SPECIES CLASSIFICATION
# ============================================================

TASK_NAME = "task2"
TASK_FOLDER = "data/task2_data"
OUTPUT_FOLDER = "outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "figures"), exist_ok=True)


# ============================================================
# 1. Helper functions
# ============================================================

def show_files(folder):
    print("\nCurrent working directory:")
    print(os.getcwd())

    print(f"\nFiles inside {folder}:")
    if os.path.exists(folder):
        for file in os.listdir(folder):
            print(" -", file)
    else:
        print("Folder does not exist:", folder)


def find_file(folder, possible_names):
    """
    Finds a file from a list of possible names.
    This helps if your lecturer used slightly different filenames.
    """
    files = os.listdir(folder)

    for name in possible_names:
        if name in files:
            return os.path.join(folder, name)

    raise FileNotFoundError(
        f"Could not find any of these files in {folder}: {possible_names}\n"
        f"Available files are: {files}"
    )


def load_metadata(task_folder):
    train_meta_path = find_file(task_folder, [
        "train_metadata.csv",
        "train_meta.csv",
        "training_metadata.csv"
    ])

    test_meta_path = find_file(task_folder, [
        "test_metadata.csv",
        "test_meta.csv",
        "testing_metadata.csv"
    ])

    train_meta = pd.read_csv(train_meta_path)
    test_meta = pd.read_csv(test_meta_path)

    print("\nTrain metadata shape:", train_meta.shape)
    print("Test metadata shape:", test_meta.shape)

    print("\nTrain metadata columns:")
    print(list(train_meta.columns))

    print("\nTest metadata columns:")
    print(list(test_meta.columns))

    return train_meta, test_meta


def detect_label_column(train_meta):
    """
    Finds the label column in train_metadata.csv.
    """
    possible_label_columns = [
        "label",
        "class",
        "class_id",
        "class_name",
        "category",
        "species",
        "target",
        "y"
    ]

    for col in possible_label_columns:
        if col in train_meta.columns:
            return col

    raise ValueError(
        "Could not automatically find the label column.\n"
        "Please check train_metadata.csv and tell me the column names."
    )


def remove_non_feature_columns(df):
    """
    Keep only numeric feature columns.
    Remove ID/path/label columns if they exist.
    """
    df = df.copy()

    possible_non_features = [
        "id",
        "image_id",
        "file",
        "filename",
        "file_name",
        "path",
        "image_path",
        "label",
        "class",
        "class_id",
        "class_name",
        "category",
        "species",
        "target",
        "y"
    ]

    for col in possible_non_features:
        if col in df.columns:
            df = df.drop(columns=[col])

    numeric_df = df.select_dtypes(include=[np.number])

    return numeric_df


def load_feature_files(task_folder, train_meta, test_meta):
    """
    Loads feature files and separates training features from test features.

    Handles two common cases:
    Case A: one feature file contains train + test rows.
    Case B: separate train/test feature files exist.
    """

    feature_sets_train = []
    feature_sets_test = []

    feature_groups = [
        {
            "combined": ["color_histogram.csv", "colour_histogram.csv"],
            "train": ["train_color_histogram.csv", "train_colour_histogram.csv"],
            "test": ["test_color_histogram.csv", "test_colour_histogram.csv"]
        },
        {
            "combined": ["hog_pca.csv", "HOG_PCA.csv"],
            "train": ["train_hog_pca.csv", "train_HOG_PCA.csv"],
            "test": ["test_hog_pca.csv", "test_HOG_PCA.csv"]
        },
        {
            "combined": ["additional_features.csv"],
            "train": ["train_additional_features.csv"],
            "test": ["test_additional_features.csv"]
        }
    ]

    files = os.listdir(task_folder)

    for group in feature_groups:
        found = False

        # Case B: separate train and test feature files
        train_file = None
        test_file = None

        for name in group["train"]:
            if name in files:
                train_file = os.path.join(task_folder, name)

        for name in group["test"]:
            if name in files:
                test_file = os.path.join(task_folder, name)

        if train_file is not None and test_file is not None:
            print("\nLoading separate feature files:")
            print("Train:", train_file)
            print("Test :", test_file)

            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)

            feature_sets_train.append(remove_non_feature_columns(train_df))
            feature_sets_test.append(remove_non_feature_columns(test_df))

            found = True

        # Case A: combined file containing train + test
        if not found:
            for name in group["combined"]:
                if name in files:
                    path = os.path.join(task_folder, name)
                    print("\nLoading combined feature file:", path)

                    df = pd.read_csv(path)
                    df_numeric = remove_non_feature_columns(df)

                    n_train = len(train_meta)
                    n_test = len(test_meta)

                    if len(df_numeric) == n_train + n_test:
                        train_part = df_numeric.iloc[:n_train].reset_index(drop=True)
                        test_part = df_numeric.iloc[n_train:n_train + n_test].reset_index(drop=True)
                    elif len(df_numeric) == n_train:
                        train_part = df_numeric.reset_index(drop=True)
                        test_part = None
                        print(
                            "Warning: This feature file only has training rows. "
                            "No test features found for this file."
                        )
                    else:
                        raise ValueError(
                            f"Unexpected number of rows in {name}.\n"
                            f"Rows in feature file: {len(df_numeric)}\n"
                            f"Train rows: {n_train}\n"
                            f"Test rows: {n_test}\n"
                            f"Expected either {n_train} or {n_train + n_test} rows."
                        )

                    feature_sets_train.append(train_part)

                    if test_part is not None:
                        feature_sets_test.append(test_part)

                    found = True
                    break

        if not found:
            print("\nWarning: Could not find one feature group:")
            print(group)

    if len(feature_sets_train) == 0:
        raise ValueError("No feature files were loaded. Check your filenames.")

    X_train_full = pd.concat(feature_sets_train, axis=1)

    if len(feature_sets_test) > 0:
        X_test = pd.concat(feature_sets_test, axis=1)
    else:
        X_test = None

    # Rename columns to avoid duplicate column name problems
    X_train_full.columns = [f"f{i}" for i in range(X_train_full.shape[1])]

    if X_test is not None:
        X_test.columns = [f"f{i}" for i in range(X_test.shape[1])]

    print("\nFinal training feature shape:", X_train_full.shape)

    if X_test is not None:
        print("Final test feature shape:", X_test.shape)
    else:
        print("Final test feature shape: None")

    return X_train_full, X_test


def create_models():
    """
    Models for comparison.

    Task 2 is smaller and harder, so simple regularised models like
    Logistic Regression and SVM may work well.
    """

    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, random_state=42))
    ])

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", C=10, gamma="scale", probability=True))
    ])

    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=3))
    ])

    random_forest = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    voting = VotingClassifier(
        estimators=[
            ("logistic", logistic),
            ("svm", svm),
            ("rf", random_forest)
        ],
        voting="soft"
    )

    models = {
        "Logistic Regression": logistic,
        "kNN": knn,
        "SVM": svm,
        "Random Forest": random_forest,
        "Voting Ensemble": voting
    }

    return models


def save_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print("Saved confusion matrix to:", output_path)


def make_submission(test_meta, predictions, output_path):
    """
    Makes a Kaggle-style submission.
    You may need to adjust column names after checking Kaggle sample submission.
    """

    submission = pd.DataFrame()

    possible_id_columns = [
        "id",
        "image_id",
        "filename",
        "file_name",
        "path",
        "image_path"
    ]

    id_col = None
    for col in possible_id_columns:
        if col in test_meta.columns:
            id_col = col
            break

    if id_col is not None:
        submission[id_col] = test_meta[id_col]
    else:
        submission["id"] = range(len(predictions))

    submission["label"] = predictions

    submission.to_csv(output_path, index=False)

    print("Saved submission to:", output_path)
    print("\nSubmission preview:")
    print(submission.head())


# ============================================================
# 2. Main program
# ============================================================

def main():
    print("=" * 70)
    print("TASK 2: FINE-GRAINED BIRD SPECIES CLASSIFICATION")
    print("=" * 70)

    show_files(TASK_FOLDER)

    train_meta, test_meta = load_metadata(TASK_FOLDER)

    label_col = detect_label_column(train_meta)
    print("\nDetected label column:", label_col)

    y = train_meta[label_col]

    X, X_test = load_feature_files(TASK_FOLDER, train_meta, test_meta)

    print("\nLabel counts:")
    print(y.value_counts())

    # Task 2 is small, so use stratified split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = create_models()

    results = []
    best_model = None
    best_model_name = None
    best_accuracy = -1
    best_valid_predictions = None

    print("\nTraining and evaluating models...")

    for model_name, model in models.items():
        print("\n" + "=" * 70)
        print("Training:", model_name)
        print("=" * 70)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_valid)
        accuracy = accuracy_score(y_valid, y_pred)

        print("Validation accuracy:", accuracy)
        print("\nClassification report:")
        print(classification_report(y_valid, y_pred))

        print("\nConfusion matrix:")
        print(confusion_matrix(y_valid, y_pred))

        results.append({
            "model": model_name,
            "accuracy": accuracy
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name
            best_valid_predictions = y_pred

    results_df = pd.DataFrame(results).sort_values(by="accuracy", ascending=False)

    print("\n" + "=" * 70)
    print("RESULT SUMMARY")
    print("=" * 70)
    print(results_df)

    results_path = os.path.join(OUTPUT_FOLDER, "task2_results.csv")
    results_df.to_csv(results_path, index=False)
    print("\nSaved results to:", results_path)

    print("\nBest model:", best_model_name)
    print("Best validation accuracy:", best_accuracy)

    save_confusion_matrix(
        y_valid,
        best_valid_predictions,
        os.path.join(OUTPUT_FOLDER, "figures", "task2_confusion_matrix.png")
    )

    # Train best model on all training data
    print("\nTraining best model on all labelled Task 2 data...")
    best_model.fit(X, y)

    # Predict test data
    if X_test is not None:
        print("\nPredicting Task 2 test labels...")
        test_predictions = best_model.predict(X_test)

        make_submission(
            test_meta,
            test_predictions,
            os.path.join(OUTPUT_FOLDER, "task2_submission.csv")
        )
    else:
        print("\nNo test features found, so no Kaggle submission was created.")
        print("Send me a screenshot of your data/task2 files and I will adjust the code.")


if __name__ == "__main__":
    main()
    