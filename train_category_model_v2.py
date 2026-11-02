
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/priority_labeling_v2.csv")

df = df.dropna(subset=["text", "category_truth"])
df = df[df["category_truth"] != "EOL"]

X = df["text"].astype(str)
y = df["category_truth"]

# Use the same split for a fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Combine word and character features
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

# Build model
model = Pipeline([
    ("features", features),
    ("classifier", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=2
    ))
])

# Train
print("Training improved category classifier...")
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
print("\nMODEL V2 EVALUATION")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# Save separately
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/category_classifier_v2.joblib")

print("\nV2 model saved successfully!")
