
import pandas as pd
import re
import unicodedata

# Load datasets
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

# Combine text and labels
train = pd.merge(X_train, y_train, on="id")
test = pd.merge(X_test, y_test, on="id")

# Remove conflicting training labels
conflicts = train.groupby("text")["category_truth"].nunique()
conflict_texts = conflicts[conflicts > 1].index

train = train[~train["text"].isin(conflict_texts)]

# Remove duplicate training tickets
train = train.drop_duplicates(subset="text", keep="first")

# Prevent train-test text leakage
test_texts = set(test["text"])
train = train[~train["text"].isin(test_texts)]

# Text cleaning function
def clean_text(text):
    text = str(text)

    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove corrupted characters
    text = text.replace("�", " ")

    # Remove placeholders
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Handle punctuation while preserving technical symbols
    text = re.sub(r"[^\w\s+#.-]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Clean ticket descriptions
train["clean_text"] = train["text"].apply(clean_text)
test["clean_text"] = test["text"].apply(clean_text)

# Remove empty descriptions
train = train[train["clean_text"] != ""]
test = test[test["clean_text"] != ""]

# Save processed datasets
train.to_csv("data/clean_train.csv", index=False)
test.to_csv("data/clean_test.csv", index=False)

print("PREPROCESSING COMPLETED")
print("-" * 40)
print("Training records:", len(train))
print("Testing records:", len(test))
print("Training categories:", train["category_truth"].nunique())
print("Testing categories:", test["category_truth"].nunique())
print("Empty training texts:", train["clean_text"].eq("").sum())
print("\nCleaned sample:")
print(train[["clean_text", "category_truth"]].head())
