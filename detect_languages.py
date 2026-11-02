
import pandas as pd
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

df = pd.read_csv("data/priority_labeling_v2.csv")

def detect_language(text):
    try:
        return detect(str(text))
    except:
        return "unknown"

df["language"] = df["text"].apply(detect_language)

print("\nLanguage distribution:")
print(df["language"].value_counts())

print("\nSample tickets:")
print(df[["id", "language", "text"]].head(20).to_string(index=False))

df.to_csv("data/tickets_with_languages.csv", index=False)

print("\nLanguage detection completed!")
