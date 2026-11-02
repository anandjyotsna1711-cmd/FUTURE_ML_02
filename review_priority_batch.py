
import pandas as pd

df = pd.read_csv("data/priority_review.csv")

# Change this number to review another batch
BATCH_SIZE = 20
BATCH_NUMBER = 7

start = (BATCH_NUMBER - 1) * BATCH_SIZE
end = start + BATCH_SIZE

batch = df.iloc[start:end]

print(f"PRIORITY REVIEW - BATCH {BATCH_NUMBER}")
print("=" * 60)

for _, row in batch.iterrows():

    print("\nID:", row["id"])
    print("Ticket:", row["text"])
    print("Keyword suggestion:", row["keyword_suggestion"])
    print("-" * 60)

print(f"\nShowing tickets {start + 1} to {min(end, len(df))}")
print("Total tickets:", len(df))
