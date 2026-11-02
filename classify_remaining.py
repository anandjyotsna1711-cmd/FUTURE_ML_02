
import pandas as pd

df = pd.read_csv("data/priority_labeling.csv")

high_keywords = [
    "server down",
    "system outage",
    "critical failure",
    "production down",
    "network outage",
    "security breach",
    "vulnerable sql-server",
    "vulnerable sql server",
    "hostalarm",
    "up -> down",
    "service unavailable"
]

medium_keywords = [
    "unable to email",
    "cannot email",
    "email not working",
    "outlook issue",
    "fails to publish",
    "does not work",
    "not working",
    "cannot access",
    "connection issue",
    "blocked website",
    "please block",
    "spam",
    "computer troubles",
    "malfunction",
    "failed",
    "unable to access",
    "problemas",
    "problem"
]

low_keywords = [
    "create shared mailbox",
    "license request",
    "light user",
    "access to public folder",
    "file share access",
    "software installation",
    "software update",
    "scanner config",
    "configuration",
    "config.",
    "adjustment",
    "adjustments",
    "user creation",
    "new user",
    "permission request",
    "recover file",
    "mailbox creation",
    "instalação de software",
    "instalacao de software"
]

def classify_priority(description):
    text = str(description).lower()

    if any(word in text for word in high_keywords):
        return "High"

    if any(word in text for word in medium_keywords):
        return "Medium"

    if any(word in text for word in low_keywords):
        return "Low"

    return "Unclassified"


# Only classify tickets that remain unclassified
mask = df["priority"].isna() | (df["priority"] == "Unclassified")

df.loc[mask, "priority"] = (
    df.loc[mask, "text"].apply(classify_priority)
)

# Save separately to protect the original dataset
df.to_csv("data/priority_labeling_v2.csv", index=False)

print("Updated classification completed!")

print("\nPriority distribution:")
print(df["priority"].value_counts(dropna=False))

print("\nTotal tickets:", len(df))
print("Unclassified:", (df["priority"] == "Unclassified").sum())
