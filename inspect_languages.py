
import pandas as pd

df = pd.read_csv("data/tickets_with_languages.csv")

print("\nLanguage counts:")
print(df["language"].value_counts())

print("\nNon-English tickets:")
non_english = df[df["language"] != "en"]

print(
    non_english[["id", "language", "text"]]
    .to_string(index=False)
)
