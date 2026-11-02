
import pandas as pd

df = pd.read_csv("data/tickets_with_languages.csv")

unclassified = df[df["priority"] == "Unclassified"].copy()

print("\nUnclassified tickets by language:")
print(unclassified["language"].value_counts())

print("\nUnclassified ticket details:")

for language, group in unclassified.groupby("language"):
    print(f"\n--- {language.upper()} ({len(group)} tickets) ---")

    for _, row in group.iterrows():
        print(f"ID: {row['id']} | {row['text']}")

