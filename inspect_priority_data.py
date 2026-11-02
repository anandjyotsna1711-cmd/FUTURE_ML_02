
import pandas as pd

df = pd.read_csv("data/priority_labeling.csv")

print("PRIORITY DATA AUDIT")
print("===================")

print("\nPriority distribution:")
print(df["priority"].value_counts(dropna=False))

# Display all manually labelled High tickets
print("\n\nHIGH PRIORITY TICKETS")
print("=====================")

high = df[df["priority"] == "High"]

for _, row in high.iterrows():
    print("\nID:", row["id"])
    print("Ticket:", row["text"])

# Display sample Low and Medium tickets
for priority in ["Low", "Medium"]:

    print("\n\n", priority.upper(), "PRIORITY SAMPLES")
    print("=" * 40)

    samples = df[df["priority"] == priority].head(15)

    for _, row in samples.iterrows():
        print("\nID:", row["id"])
        print("Ticket:", row["text"])

# Show unlabelled tickets for further review
print("\n\nUNLABELLED TICKETS")
print("==================")

unlabelled = df[df["priority"].isna()]

print("Total:", len(unlabelled))

for _, row in unlabelled.head(15).iterrows():
    print("\nID:", row["id"])
    print("Ticket:", row["text"])
