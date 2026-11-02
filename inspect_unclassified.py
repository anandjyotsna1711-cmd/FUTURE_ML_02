
import pandas as pd

df = pd.read_csv("data/priority_labeling_v2.csv")

remaining = df[df["priority"] == "Unclassified"]

print("Unclassified tickets:", len(remaining))
print("\nTicket descriptions:\n")

for _, row in remaining.iterrows():
    print(f"ID: {row['id']}")
    print(f"Category: {row['category_truth']}")
    print(f"Description: {row['text']}")
    print("-" * 70)
