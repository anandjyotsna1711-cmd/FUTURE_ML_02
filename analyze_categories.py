
import pandas as pd
import os

path = os.path.join("data", "y_train.csv")
test_path = os.path.join("data", "y_test.csv")

train = pd.read_csv(path)
test = pd.read_csv(test_path)

all_labels = pd.concat([
    train["category_truth"],
    test["category_truth"]
])

print("TOTAL CATEGORIES:", all_labels.nunique())

print("\nCATEGORY DISTRIBUTION:")
print(all_labels.value_counts())

print("\nCATEGORY DISTRIBUTION IN PERCENTAGE:")
print((all_labels.value_counts(normalize=True) * 100).round(2))

print("\nALL UNIQUE CATEGORIES:")
print(sorted(all_labels.unique()))
