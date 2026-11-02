
import pandas as pd

original = pd.read_csv("data/priority_labeling.csv")
suggestions = pd.read_csv("data/priority_labeling_v2.csv")

# Select tickets without manually verified priority
review = original[
    original["priority"].fillna("Unclassified").str.strip() == "Unclassified"
].copy()

# Bring in keyword-generated suggestions
review = review.merge(
    suggestions[["id", "priority"]],
    on="id",
    how="left",
    suffixes=("", "_suggested")
)

# Prepare the review sheet
review = review.rename(columns={
    "priority_suggested": "keyword_suggestion"
})

review["verified_priority"] = ""

review = review[
    ["id", "text", "keyword_suggestion", "verified_priority"]
]

review.to_csv("data/priority_review.csv", index=False)

print("Priority review sheet created!")
print("Tickets to review:", len(review))
print("\nKeyword suggestions:")
print(review["keyword_suggestion"].value_counts(dropna=False))

print("\nSaved to: data/priority_review.csv")
