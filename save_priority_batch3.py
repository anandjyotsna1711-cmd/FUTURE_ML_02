
import pandas as pd
import os
import shutil

file_path = "data/priority_review.csv"
backup_path = "data/priority_review_backup_before_batch3.csv"

# Create backup
if not os.path.exists(backup_path):
    shutil.copy2(file_path, backup_path)
    print("Backup created:", backup_path)

df = pd.read_csv(file_path)

# Ensure correct data types
df["id"] = df["id"].astype(str).str.strip()
df["verified_priority"] = (
    df["verified_priority"].fillna("").astype(str)
)

# Confirmed Batch 3 labels
labels = {
    "1440": "Low",
    "1131": "Low",
    "4906": "Low",
    "1189": "Medium",
    "1458": "Medium",
    "1081": "Medium",
    "7582": "Low",
    "5905": "Low",
    "2382": "Low",
    "770": "Low",
    "3567": "Medium",
    "1015": "Low",
    "6623": "Medium",
    "1220": "Low"
}

# Check ticket IDs
missing_ids = set(labels) - set(df["id"])

if missing_ids:
    print("ERROR: Missing IDs:", missing_ids)
    raise SystemExit

# Prevent overwriting existing labels
for ticket_id, priority in labels.items():
    current = df.loc[
        df["id"] == ticket_id, "verified_priority"
    ].iloc[0]

    if current not in ("", priority):
        print("ERROR: Existing label conflict:", ticket_id)
        raise SystemExit

# Apply labels
for ticket_id, priority in labels.items():
    df.loc[
        df["id"] == ticket_id, "verified_priority"
    ] = priority

# Save
df.to_csv(file_path, index=False)

# Display progress
verified = df[df["verified_priority"] != ""]

print("\nBATCH 3 SAVED SUCCESSFULLY!")
print("New labels added:", len(labels))

print("\nOverall verified priority distribution:")
print(verified["verified_priority"].value_counts())

print("\nTotal verified:", len(verified))
print("Remaining:", len(df) - len(verified))

print("\nSaved to:", file_path)

