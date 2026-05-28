import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "report_figures"
os.makedirs(OUT_DIR, exist_ok=True)

TASK1_ACC = {
    "kNN": 0.284,
    "Random Forest": 0.520,
    "Linear SVM": 0.5666666667,
    "ResNet18 + LR": 0.8626666667,
}

TASK2_ACC = {
    "Logistic Regression": 0.8452380952,
    "SVM RBF": 0.8571428571,
    "Random Forest": 0.8214285714,
    "Voting Ensemble": 0.8452380952,
}

TASK1_RESNET_CM = np.array([
    [61,1,4,2,0,0,3,4,0,0],
    [1,69,1,0,0,0,1,0,0,3],
    [5,0,62,1,5,0,2,0,0,0],
    [4,0,2,62,1,0,3,3,0,0],
    [1,0,15,3,55,0,0,1,0,0],
    [0,0,0,0,0,69,1,0,5,0],
    [8,0,1,2,0,0,64,0,0,0],
    [1,0,2,2,3,1,1,65,0,0],
    [0,0,0,1,0,3,0,0,70,1],
    [3,2,0,0,0,0,0,0,0,70],
])

TASK2_SVM_RBF_CM = np.array([
    [8,0,0,0,0,0,0,0,0,0],
    [0,7,0,0,0,1,0,0,0,0],
    [0,0,8,0,0,0,0,0,0,0],
    [0,0,0,6,0,0,3,0,0,0],
    [0,0,0,0,8,0,0,1,0,0],
    [0,0,0,0,0,9,0,0,0,0],
    [0,0,0,3,0,0,5,0,0,0],
    [0,0,0,0,0,0,0,9,0,0],
    [0,0,0,0,0,0,0,0,6,2],
    [0,0,0,0,0,0,0,0,2,6],
])

TASK1_RESNET_F1 = [0.77,0.94,0.77,0.84,0.79,0.93,0.85,0.88,0.93,0.94]
TASK2_SVM_F1 = [1.00,0.93,1.00,0.67,0.94,0.95,0.62,0.95,0.75,0.75]


def save_accuracy_chart(data, title, filename):
    labels = list(data.keys())
    values = list(data.values())
    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(labels, values)
    plt.ylim(0, 1.0)
    plt.ylabel("Validation accuracy")
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200)
    plt.close()


def save_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(6.4, 5.6))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(range(cm.shape[1]))
    plt.yticks(range(cm.shape[0]))
    plt.colorbar(label="Count")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200)
    plt.close()


def save_f1_chart(f1_scores, title, filename):
    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(f1_scores))
    bars = plt.bar(x, f1_scores)
    plt.ylim(0, 1.05)
    plt.xlabel("Class ID")
    plt.ylabel("F1-score")
    plt.title(title)
    plt.xticks(x)
    for bar, value in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200)
    plt.close()


save_accuracy_chart(TASK1_ACC, "Task 1 validation accuracy by model", "figure1_task1_accuracy.png")
save_confusion_matrix(TASK1_RESNET_CM, "Task 1 ResNet18 + Logistic Regression confusion matrix", "figure2_task1_resnet_confusion_matrix.png")
save_f1_chart(TASK1_RESNET_F1, "Task 1 ResNet18 + Logistic Regression per-class F1-score", "figure3_task1_resnet_f1.png")
save_accuracy_chart(TASK2_ACC, "Task 2 validation accuracy by model", "figure4_task2_accuracy.png")
save_confusion_matrix(TASK2_SVM_RBF_CM, "Task 2 SVM RBF confusion matrix", "figure5_task2_svm_confusion_matrix.png")
save_f1_chart(TASK2_SVM_F1, "Task 2 SVM RBF per-class F1-score", "figure6_task2_svm_f1.png")

print(f"Saved figures to: {os.path.abspath(OUT_DIR)}")
