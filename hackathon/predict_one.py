import joblib

# Load model & vectorizer
clf = joblib.load("hallucination_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

def predict_sentence(text):
    X = tfidf.transform([text])
    pred = clf.predict(X)[0]
    label = "Hallucination 🚨" if pred == 1 else "Factual ✅"
    return label

if __name__ == "__main__":
    while True:
        user_inp = input("\nEnter a hypothesis (or 'q' to quit): ")
        if user_inp.lower() == 'q':
            break
        print("Prediction:", predict_sentence(user_inp))
