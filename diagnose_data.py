
import pandas as pd

train = pd.read_csv("data/X_train.csv").merge(
    pd.read_csv("data/y_train.csv"), on="id"
)

test = pd.read_csv("data/X_test.csv").merge(
    pd.read_csv("data/y_test.csv"), on="id"
)

print("ORIGINAL TRAINING CATEGORIES")
print(train["category_truth"].value_counts())

print("\nORIGINAL TESTING CATEGORIES")
print(test["category_truth"].value_counts())

# Unique training descriptions
unique_train = train.drop_duplicates(subset="text")

print("\nAFTER REMOVING DUPLICATES")
print(unique_train["category_truth"].value_counts())

# Categories missing from cleaned training data
missing = set(test["category_truth"]) - set(
    unique_train["category_truth"]
)

print("\nMISSING TRAINING CATEGORIES:")
print(missing)

for category in missing:
    print("\nCATEGORY:", category)
    print(train[train["category_truth"] == category][
        ["id", "text"]
    ].head(10))

print("\nTRAIN-TEST OVERLAPPING TEXTS:")
overlap = set(train["text"]) & set(test["text"])
print(len(overlap))
