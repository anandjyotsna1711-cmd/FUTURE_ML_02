
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Load dataset
df = pd.read_csv("data/priority_labeling_v2.csv")

df = df.dropna(subset=["text", "category_truth"])
df = df[df["category_truth"] != "EOL"]

X = df["text"].astype(str)
y = df["category_truth"]

# Five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Model 1: Word TF-IDF + Logistic Regression
model_v1 = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=20000
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

# Shared word + character features
def create_features():
    return FeatureUnion([
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

# Model 2: Word + Character TF-IDF + Logistic Regression
model_v2 = Pipeline([
    ("features", create_features()),
    ("classifier", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=2
    ))
])

# Model 3: Word + Character TF-IDF + Linear SVM
model_svm = Pipeline([
    ("features", create_features()),
    ("classifier", LinearSVC(
        class_weight="balanced",
        C=1.0
    ))
])

models = {
    "Logistic Regression V1": model_v1,
    "Logistic Regression V2": model_v2,
    "Linear SVM": model_svm
}

print("5-FOLD CROSS-VALIDATION")
print("=======================")
print("Total tickets:", len(df))
print("Number of folds: 5")

results = []

for name, model in models.items():

    print("\nEvaluating:", name)

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "f1_macro": "f1_macro"
        },
        n_jobs=1
    )

    accuracy = scores["test_accuracy"]
    f1 = scores["test_f1_macro"]

    results.append({
        "Model": name,
        "Mean Accuracy": accuracy.mean(),
        "Accuracy Std": accuracy.std(),
        "Mean Macro F1": f1.mean(),
        "F1 Std": f1.std()
    })

    print("Fold accuracies:", np.round(accuracy, 4))
    print("Mean accuracy:", round(accuracy.mean(), 4))
    print("Mean macro F1:", round(f1.mean(), 4))

# Final comparison
results_df = pd.DataFrame(results)

print("\n\nFINAL MODEL COMPARISON")
print("======================")
print(results_df.round(4).to_string(index=False))

best_model = results_df.loc[
    results_df["Mean Macro F1"].idxmax()
]

print("\nBest model based on mean macro F1:")
print(best_model["Model"])
