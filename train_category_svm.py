
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/priority_labeling_v2.csv")

df = df.dropna(subset=["text", "category_truth"])
df = df[df["category_truth"] != "EOL"]

X = df["text"].astype(str)
y = df["category_truth"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

features = FeatureUnion([
    ("word", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True
    )),
    ("character", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=30000,
        sublinear_tf=True
    ))
])

model = Pipeline([
    ("features", features),
    ("classifier", LinearSVC(
        class_weight="balanced",
        C=1.0
    ))
])

print("Training Linear SVM classifier...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nSVM MODEL EVALUATION")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/category_classifier_svm.joblib")

print("\nSVM model saved successfully!")
