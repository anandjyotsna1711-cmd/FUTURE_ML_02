import joblib

model = joblib.load("models/category_classifier.joblib")

print("SUPPORT TICKET CATEGORY PREDICTOR")
print("--------------------------------")

while True:
    ticket = input("\nEnter ticket description (or type exit): ")

    if ticket.lower() == "exit":
        break

    prediction = model.predict([ticket])

    probabilities = model.predict_proba([ticket])[0]
    confidence = max(probabilities) * 100

    print("\nPredicted Category:", prediction[0])
    print("Confidence:", round(confidence, 2), "%")