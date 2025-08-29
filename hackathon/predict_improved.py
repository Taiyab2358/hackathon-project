import joblib
import numpy as np

# === Load model and vectorizer ===
clf = joblib.load("hallucination_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

print("✅ Model and vectorizer loaded")

# === Helper function for prediction ===
def predict_one(hyp, tgt, threshold=0.5):
    # Transform hyp + tgt
    hyp_vec = tfidf.transform([hyp])
    tgt_vec = tfidf.transform([tgt])

    # Compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(hyp_vec, tgt_vec)[0,0]

    # Stack features (same order as training)
    from scipy.sparse import hstack
    X = hstack([hyp_vec, np.array([[sim]])])

    # Get probability
    prob = clf.predict_proba(X)[0,1]  # probability of hallucination
    pred = 1 if prob >= threshold else 0

    return pred, prob

# === Interactive loop ===
print("\n💡 Enter hypothesis + target. Type 'q' anytime to quit.")
while True:
    hyp = input("\nEnter hypothesis: ")
    if hyp.lower() == "q":
        break
    tgt = input("Enter target/reference: ")
    if tgt.lower() == "q":
        break

    pred, prob = predict_one(hyp, tgt, threshold=0.5)

    if pred == 1:
        print(f"Prediction: 🚨 Hallucination (confidence: {prob*100:.2f}%)")
    else:
        print(f"Prediction: ✅ Not hallucination (confidence: {(1-prob)*100:.2f}%)")
