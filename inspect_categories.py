
import pandas as pd

df = pd.read_csv("data/priority_labeling_v2.csv")

categories = sorted(df["category_truth"].dropna().unique())

for category in categories:
    print("\n" + "=" * 60)
    print("CATEGORY:", category)
    print("=" * 60)

    tickets = df[df["category_truth"] == category]["text"]

    for ticket in tickets.head(15):
        print("-", ticket)
