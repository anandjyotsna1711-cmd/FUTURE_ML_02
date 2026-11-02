
import pandas as pd

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

train = pd.merge(X_train, y_train, on="id")
test = pd.merge(X_test, y_test, on="id")

print("TRAINING DATA DUPLICATES")
print(train["text"].duplicated().sum())

print("\nDUPLICATES WITH DIFFERENT CATEGORIES")

duplicates = train.groupby("text")["category_truth"].nunique()

print((duplicates > 1).sum())

print("\nDUPLICATE TEXTS WITH CONFLICTING LABELS")

conflicts = duplicates[duplicates > 1].index

print(train[train["text"].isin(conflicts)][
    ["text", "category_truth"]
].head(10))

print("\nTRAIN-TEST TEXT OVERLAP")

overlap = set(train["text"]) & set(test["text"])

print("Overlapping tickets:", len(overlap))
