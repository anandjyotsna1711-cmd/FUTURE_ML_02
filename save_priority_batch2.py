
import pandas as pd
import os
import shutil

file_path = "data/priority_review.csv"
backup_path = "data/priority_review_backup_before_batch2.csv"

# Create backup before editing
if not os.path.exists(backup_path):
    shutil.copy2(file_path, backup_path)
    print("Backup created:", backup_path)

df = pd.read_csv(file_path)

# Ensure correct data types
df["id"] = df["id"].astype(str).str.strip()
df["verified_priority"] = (
    df["verified_priority"].fillna("").astype(str)
)

# Confirmed Batch 2 labels
labels = {
    "2087": "Low",
    "11383": "Medium",
    "833": "Low",
    "924": "Low",
    "852": "Medium",
    "939": "Low",
    "3807": "Medium",
    "8957": "Medium",
    "1340": "Medium",
    "6959": "Low",
    "1814": "Medium",
    "5604": "Low",
    "9535": "Low",
    "1075": "Low",
    "2850": "Low",
    "968": "Low",
    "2214": "Low",
    "1436": "Low",
    "3838": "Medium"
}

# Check IDs exist
missing_ids = set(labels) - set(df["id"])

if missing_ids:
    print("ERROR: Missing ticket IDs:", missing_ids)
    raise SystemExit

# Prevent accidental overwriting of existing labels
for ticket_id, priority in labels.items():
    current = df.loc[
        df["id"] == ticket_id, "verified_priority"
    ].iloc[0]

    if current not in ("", priority):
        print("ERROR: Existing label conflict for ID:", ticket_id)
        raise SystemExit

# Apply confirmed labels
for ticket_id, priority in labels.items():
    df.loc[
        df["id"] == ticket_id, "verified_priority"
    ] = priority

df.to_csv(file_path, index=False)

# Display progress
verified = df[df["verified_priority"] != ""]

print("\nBATCH 2 SAVED SUCCESSFULLY!")
print("New labels added:", len(labels))

print("\nOverall verified priority distribution:")
print(verified["verified_priority"].value_counts())

print("\nTotal verified:", len(verified))
print("Remaining:", len(df) - len(verified))
print("\nSaved to:", file_path)
