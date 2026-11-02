
import pandas as pd
from pathlib import Path

original_path = Path("data/priority_labeling.csv")
review_path = Path("data/priority_review.csv")
output_path = Path("data/priority_training.csv")

df = pd.read_csv(original_path)
review = pd.read_csv(review_path)

df["id"] = df["id"].astype(str).str.strip()
review["id"] = review["id"].astype(str).str.strip()

review["verified_priority"] = (
    review["verified_priority"].fillna("").astype(str).str.strip()
)

# Only use manually verified labels
verified = review[
    review["verified_priority"].isin(["Low", "Medium", "High"])
]

# Check IDs before merging
if df["id"].duplicated().any():
    raise ValueError("Duplicate IDs in original dataset.")

if review["id"].duplicated().any():
    raise ValueError("Duplicate IDs in review dataset.")

for _, row in verified.iterrows():
    ticket_id = row["id"]
    priority = row["verified_priority"]

    match = df["id"] == ticket_id

    if not match.any():
        raise ValueError(f"Ticket {ticket_id} not found.")

    current = str(df.loc[match, "priority"].iloc[0]).strip()

    if current.lower() != "unclassified" and current != priority:
        raise ValueError(
            f"Priority conflict for ID {ticket_id}: {current} vs {priority}"
        )

    df.loc[match, "priority"] = priority

df.to_csv(output_path, index=False)

print("FINAL PRIORITY DATASET CREATED!")
print("Saved to:", output_path)

print("\nPriority distribution:")
print(df["priority"].value_counts())

print("\nTotal tickets:", len(df))
print("Unclassified tickets:",
      (df["priority"].str.lower() == "unclassified").sum())
