
import pandas as pd

files = [
    "data/priority_labeling.csv",
    "data/priority_labeling_v2.csv"
]

for file in files:

    print("\n" + "=" * 50)
    print("FILE:", file)
    print("=" * 50)

    df = pd.read_csv(file)

    df["priority"] = (
        df["priority"]
        .fillna("Unclassified")
        .astype(str)
        .str.strip()
    )

    print("\nTotal tickets:", len(df))

    print("\nPriority distribution:")
    print(df["priority"].value_counts())

    print("\nUnclassified tickets:")
    unclassified = df[
        df["priority"].str.lower() == "unclassified"
    ]

    print("Count:", len(unclassified))

    print("\nSample unclassified tickets:")

    for _, row in unclassified.head(10).iterrows():
        print("\nID:", row["id"])
        print("Ticket:", row["text"])
