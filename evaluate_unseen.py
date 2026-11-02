
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report, f1_score

train = pd.read_csv("data/clean_train.csv")
test = pd.read_csv("data/clean_test.csv")

model = joblib.load("models/logistic_regression.pkl")

# Find descriptions not seen during training
train_texts = set(train["clean_text"])

unseen_test = test[
    ~test["clean_text"].isin(train_texts)
].copy()

print("TOTAL TEST RECORDS:", len(test))
print("UNSEEN TEST RECORDS:", len(unseen_test))
print("SEEN TEST RECORDS:", len(test) - len(unseen_test))

if len(unseen_test) > 0:

    predictions = model.predict(unseen_test["clean_text"])

    print("\nUNSEEN DATA EVALUATION")
    print("Accuracy:", round(
        accuracy_score(
            unseen_test["category_truth"], predictions
        ) * 100, 2
    ), "%")

    print("Macro F1:", round(
        f1_score(
            unseen_test["category_truth"],
            predictions,
            average="macro"
        ), 4
    ))

    print("\nClassification Report:")
    print(classification_report(
        unseen_test["category_truth"],
        predictions,
        zero_division=0
    ))
