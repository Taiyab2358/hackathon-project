import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
from scipy.sparse import hstack
from tqdm import tqdm   # ✅ progress bar

# === Load datasets ===
train_df = pd.read_csv("train_clean.csv")
val_df = pd.read_csv("val_clean.csv")

print("✅ Loaded datasets")
print("Train size:", len(train_df))
print("Val size:", len(val_df))

# === Vectorize hypothesis and target ===
tfidf = TfidfVectorizer(max_features=5000)

# Fit on combined hyp+tgt text
tfidf.fit(train_df['hyp'].astype(str).tolist() + train_df['tgt'].astype(str).tolist())

# Transform hyp and tgt
hyp_train = tfidf.transform(train_df['hyp'].astype(str))
tgt_train = tfidf.transform(train_df['tgt'].astype(str))
hyp_val = tfidf.transform(val_df['hyp'].astype(str))
tgt_val = tfidf.transform(val_df['tgt'].astype(str))

# === Row-wise cosine similarity (with progress bar) ===
def rowwise_cosine(a, b, desc="Cosine sims"):
    sims = []
    for i in tqdm(range(a.shape[0]), desc=desc):
        sims.append(cosine_similarity(a[i], b[i])[0, 0])
    return np.array(sims).reshape(-1, 1)

print("⚡ Computing cosine similarities...")
train_sims = rowwise_cosine(hyp_train, tgt_train, desc="Train sims")
val_sims = rowwise_cosine(hyp_val, tgt_val, desc="Val sims")

# === Stack features without converting to dense ===
X_train = hstack([hyp_train, train_sims])
X_val = hstack([hyp_val, val_sims])

y_train = train_df['label']
y_val = val_df['label']

print("✅ Features prepared")

# === Train model (faster LogisticRegression instead of huge RandomForest) ===
clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_val)
print("\n✅ Evaluation on Validation Set:")
print(classification_report(y_val, y_pred))

# === Save artifacts ===
joblib.dump(clf, "hallucination_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\n✅ Improved model and vectorizer saved as hallucination_model.pkl & tfidf_vectorizer.pkl")
