
import pandas as pd
import os
import shutil

file_path = "data/priority_review.csv"
backup_path = "data/priority_review_backup_before_batch1.csv"

# Create backup if it doesn't already exist
if not os.path.exists(backup_path):
    shutil.copy2(file_path, backup_path)
    print("Backup created:", backup_path)

df = pd.read_csv(file_path)

# Convert IDs and priority column to text
df["id"] = df["id"].astype(str).str.strip()

df["verified_priority"] = (
    df["verified_priority"]
    .fillna("")
    .astype(str)
)

# Confirmed labels from Batch 1
labels = {
    "4249": "Low",
    "1631": "Low",
    "5869": "Low",
    "2258": "Low",
    "246": "Medium",
    "1393": "Medium",
    "1753": "Low",
    "1591": "Low",
    "586": "Low",
    "8981": "Low",
    "1035": "Low",
    "844": "Medium",
    "3695": "Low"
}

# Verify ticket IDs
missing_ids = set(labels.keys()) - set(df["id"])

if missing_ids:
    print("ERROR: Missing ticket IDs:", missing_ids)
    raise SystemExit

# Save confirmed labels
for ticket_id, priority in labels.items():
    df.loc[df["id"] == ticket_id, "verified_priority"] = priority

df.to_csv(file_path, index=False)

# Display results
print("\nBATCH 1 SAVED SUCCESSFULLY!")
print("Verified tickets:", len(labels))

print("\nVerified priority distribution:")
print(
    df[df["verified_priority"] != ""]["verified_priority"]
    .value_counts()
)

remaining = (df["verified_priority"] == "").sum()

print("\nTickets still awaiting review:", remaining)
print("File saved:", file_path)
