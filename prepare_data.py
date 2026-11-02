
import pandas as pd
import re
import unicodedata

# Load original datasets
train = pd.read_csv("data/X_train.csv").merge(
    pd.read_csv("data/y_train.csv"), on="id"
)

test = pd.read_csv("data/X_test.csv").merge(
    pd.read_csv("data/y_test.csv"), on="id"
)

# Identify conflicting descriptions
conflicts = train.groupby("text")["category_truth"].nunique()
conflicts = conflicts[conflicts > 1].index

# Remove only conflicting descriptions
train = train[~train["text"].isin(conflicts)].copy()

# Clean text
def clean_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("�", " ")
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = text.lower()
    text = re.sub(r"[^\w\s+#.-]", " ", text)
    text = re.sub(r"[-_.]{2,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

train["clean_text"] = train["text"].apply(clean_text)
test["clean_text"] = test["text"].apply(clean_text)

# Remove empty descriptions
train = train[train["clean_text"] != ""].copy()
test = test[test["clean_text"] != ""].copy()

# Save corrected datasets
train.to_csv("data/clean_train.csv", index=False)
test.to_csv("data/clean_test.csv", index=False)

print("CORRECTED DATA PREPARATION")
print("-" * 40)

print("Training records:", len(train))
print("Testing records:", len(test))

print("\nTraining categories:")
print(train["category_truth"].value_counts())

print("\nTesting categories:")
print(test["category_truth"].value_counts())

print("\nTraining unique categories:",
      train["category_truth"].nunique())

print("Testing unique categories:",
      test["category_truth"].nunique())

print("\nEmpty training texts:",
      train["clean_text"].eq("").sum())
