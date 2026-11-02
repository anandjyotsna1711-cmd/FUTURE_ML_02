import pandas as pd
import shutil
from pathlib import Path

file_path = Path("data/priority_review.csv")
backup_path = Path("data/priority_review_backup_before_batch4.csv")

labels = {
    "1875": "Medium",
    "4543": "Low",
    "10670": "Medium",
    "10981": "Low",
    "4504": "Low",
    "1958": "Low",
    "7237": "Low",
    "12152": "Medium",
    "1370": "Low",
    "8397": "Medium",
    "8300": "Low",
    "1151": "Low",
    "1395": "Medium",
    "13861": "Low",
    "1531": "Low",
    "921": "Low",
    "902": "Medium"
}

df = pd.read_csv(file_path)

df["id"] = df["id"].astype(str).str.strip()
df["verified_priority"] = (
    df["verified_priority"].fillna("").astype(str).str.strip()
)

# Check that every ticket ID exists
missing_ids = set(labels) - set(df["id"])

if missing_ids:
    raise ValueError(f"Ticket IDs not found: {missing_ids}")

# Prevent overwriting conflicting labels
for ticket_id, priority in labels.items():
    existing = df.loc[
        df["id"] == ticket_id, "verified_priority"
    ].iloc[0]

    if existing and existing != priority:
        raise ValueError(
            f"Conflict for ID {ticket_id}: "
            f"existing={existing}, new={priority}"
        )

# Create backup before saving
if not backup_path.exists():
    shutil.copy2(file_path, backup_path)
    print("Backup created successfully.")

# Update only the confirmed tickets
for ticket_id, priority in labels.items():
    df.loc[df["id"] == ticket_id, "verified_priority"] = priority

df.to_csv(file_path, index=False)

# Display progress
verified = df[df["verified_priority"] != ""]

print("\nBATCH 4 SAVED SUCCESSFULLY!")
print("New labels added:", len(labels))

print("\nOverall verified priority distribution:")
print(verified["verified_priority"].value_counts())

print("\nTotal verified:", len(verified))
print("Remaining unverified:", len(df) - len(verified))

print("\nIDs intentionally left Unclassified: 1160, 1345, 365")