
import pandas as pd
import shutil
from pathlib import Path

file_path = Path("data/priority_review.csv")
backup_path = Path("data/priority_review_backup_before_batches5_6_7.csv")

# Confirmed labels from batches 5, 6 and 7
labels = {
    # BATCH 5
    "4313": "Low",
    "1710": "Low",
    "1408": "Low",
    "1607": "Low",
    "8652": "Medium",
    "996": "Medium",
    "1100": "Low",
    "5254": "Low",
    "7867": "Medium",
    "11551": "Low",
    "1134": "Medium",
    "2359": "Medium",
    "10177": "Low",
    "1683": "Low",
    "7164": "Low",
    "2907": "Low",
    "2737": "Low",
    "3806": "Medium",

    # BATCH 6
    "865": "Low",
    "708": "Low",
    "3663": "Low",
    "6818": "Medium",
    "791": "Low",
    "2590": "Medium",
    "5786": "Medium",
    "874": "Low",
    "638": "Medium",
    "201": "Low",
    "9936": "Low",
    "1334": "Low",
    "1154": "Low",
    "11480": "Medium",

    # BATCH 7
    "5067": "Low",
    "5511": "Low",
    "1206": "Medium",
    "1884": "Low",
    "8351": "Medium",
    "1390": "Medium",
    "1708": "Medium"
}

# These 8 tickets were reviewed but intentionally left unclassified
unclassified_ids = [
    "1231", "841",
    "820", "1140", "699", "775", "3576", "3848"
]

df = pd.read_csv(file_path)

df["id"] = df["id"].astype(str).str.strip()
df["verified_priority"] = (
    df["verified_priority"].fillna("").astype(str).str.strip()
)

# Validate ticket IDs
if df["id"].duplicated().any():
    raise ValueError("Duplicate ticket IDs found. No changes made.")

all_ids = set(df["id"])
expected_ids = set(labels) | set(unclassified_ids)

missing_ids = expected_ids - all_ids

if missing_ids:
    raise ValueError(f"Missing ticket IDs: {missing_ids}")

if len(expected_ids) != 47:
    raise ValueError("Expected exactly 47 reviewed tickets.")

# Ensure no existing labels are overwritten
for ticket_id, priority in labels.items():
    existing = df.loc[
        df["id"] == ticket_id, "verified_priority"
    ].iloc[0]

    if existing and existing != priority:
        raise ValueError(
            f"Conflict for ID {ticket_id}: "
            f"existing={existing}, new={priority}"
        )

# Ensure intentionally unclassified tickets stay blank
for ticket_id in unclassified_ids:
    existing = df.loc[
        df["id"] == ticket_id, "verified_priority"
    ].iloc[0]

    if existing:
        raise ValueError(
            f"ID {ticket_id} already has a label. "
            "Review before continuing."
        )

# Create backup before updating
if not backup_path.exists():
    shutil.copy2(file_path, backup_path)
    print("Backup created successfully.")

# Save confirmed labels only
for ticket_id, priority in labels.items():
    df.loc[df["id"] == ticket_id, "verified_priority"] = priority

df.to_csv(file_path, index=False)

# Report final progress
verified = df[df["verified_priority"] != ""]

print("\nBATCHES 5, 6 AND 7 SAVED SUCCESSFULLY!")
print("Reviewed tickets:", len(expected_ids))
print("New verified labels:", len(labels))
print("Left intentionally unclassified:", len(unclassified_ids))

print("\nOverall verified priority distribution:")
print(verified["verified_priority"].value_counts())

print("\nTotal verified:", len(verified))
print("Remaining unverified:", len(df) - len(verified))
