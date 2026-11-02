
import joblib
import re
import unicodedata

# Load trained model
model = joblib.load("models/logistic_regression.pkl")

def clean_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("�", " ")
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = text.lower()
    text = re.sub(r"[^\w\s+#.-]", " ", text)
    text = re.sub(r"[-_.]{2,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

print("=" * 45)
print("       TICKETWISE CLASSIFIER")
print("=" * 45)

while True:
    ticket = input("\nEnter support ticket (or 'exit'): ")

    if ticket.lower() == "exit":
        print("Thank you for using TICKETWISE!")
        break

    cleaned = clean_text(ticket)

    if not cleaned:
        print("Please enter a valid ticket.")
        continue

    probabilities = model.predict_proba([cleaned])[0]
    categories = model.classes_

    top_indices = probabilities.argsort()[::-1][:3]

    print("\nPREDICTION RESULTS")
    print("-" * 30)

    for i in top_indices:
        print(categories[i], ":", round(probabilities[i] * 100, 2), "%")
