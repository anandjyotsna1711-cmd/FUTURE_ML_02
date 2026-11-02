
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

df = pd.read_csv("data/priority_labeling_v2.csv")

df = df.dropna(subset=["text", "category_truth"])
df = df[df["category_truth"] != "EOL"]

X = df["text"].astype(str)
y = df["category_truth"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = joblib.load("models/category_classifier_v2.joblib")

predictions = model.predict(X_test)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions,
    "Ticket": X_test.values
})

errors = results[results["Actual"] != results["Predicted"]]

print("\nV2 MISCLASSIFIED TICKETS")
print("------------------------")

for _, row in errors.iterrows():
    print("\nTicket:", row["Ticket"])
    print("Actual:", row["Actual"])
    print("Predicted:", row["Predicted"])

print("\nIncorrect predictions:", len(errors))
print("Correct predictions:", len(results) - len(errors))
