
import pandas as pd
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix
)

# Load prepared datasets
train = pd.read_csv("data/clean_train.csv")
test = pd.read_csv("data/clean_test.csv")

X_train = train["clean_text"]
y_train = train["category_truth"]

X_test = test["clean_text"]
y_test = test["category_truth"]

labels = sorted(y_train.unique())

# Create models directory
os.makedirs("models", exist_ok=True)

# Define models
models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ]),

    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("classifier", MultinomialNB(alpha=0.5))
    ])
}

# Train and evaluate
results = {}

for name, model in models.items():

    print("\n" + "=" * 55)
    print(name.upper())
    print("=" * 55)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    weighted_f1 = f1_score(y_test, predictions, average="weighted")

    results[name] = {
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1
    }

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=labels,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=labels
    ))

    filename = name.lower().replace(" ", "_")
    joblib.dump(model, f"models/{filename}.pkl")

# Compare models
print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)

comparison = pd.DataFrame(results).T
print(comparison.round(4))

print("\nMODELS SAVED SUCCESSFULLY!")
