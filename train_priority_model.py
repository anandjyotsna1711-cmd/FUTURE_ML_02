
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/priority_training.csv")

# Keep only usable Low and Medium labels
df = df[df["priority"].isin(["Low", "Medium"])].copy()

df["text"] = df["text"].fillna("").astype(str)

X = df["text"]
y = df["priority"]

print("PRIORITY MODEL TRAINING")
print("=" * 40)
print("\nTraining data distribution:")
print(y.value_counts())

# Word and character TF-IDF features
features = FeatureUnion([
    ("word", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True
    )),
    ("char", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=10000,
        sublinear_tf=True
    ))
])

# Model pipeline
model = Pipeline([
    ("features", features),
    ("classifier", LinearSVC(
        class_weight="balanced",
        C=1.0,
        random_state=42
    ))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print("\nHOLDOUT TEST RESULTS")
print("=" * 40)

print("Accuracy:", round(
    accuracy_score(y_test, predictions), 4
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["Low", "Medium"]
))

# Five-fold cross-validation
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n5-FOLD CROSS-VALIDATION")
print("=" * 40)
print("Scores:", cv_scores)
print("Mean accuracy:", round(cv_scores.mean(), 4))

# Save confusion matrix
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=["Low", "Medium"],
    display_labels=["Low", "Medium"],
    cmap="Blues"
)

plt.title("Priority Classification Confusion Matrix")
plt.tight_layout()
plt.savefig("results/priority_confusion_matrix.png")
plt.close()

# Train final model on all usable labelled data
model.fit(X, y)

joblib.dump(model, "models/priority_classifier.joblib")

print("\nFinal model saved successfully!")
print("Location: models/priority_classifier.joblib")
print("Confusion matrix saved in results/")
