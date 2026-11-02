
import pandas as pd

X = pd.read_csv("data/X_train.csv")
y = pd.read_csv("data/y_train.csv")

data = pd.merge(X, y, on="id")

print("SAMPLE TICKETS")
print("-" * 60)

for i, row in data.head(10).iterrows():
    print("\nCategory:", row["category_truth"])
    print("Ticket:", row["text"])
    print("-" * 60)

print("\nTEXT STATISTICS")
print("Average text length:", data["text"].str.len().mean())
print("Shortest ticket:", data["text"].str.len().min())
print("Longest ticket:", data["text"].str.len().max())

print("\nDUPLICATE TICKETS:", data["text"].duplicated().sum())
