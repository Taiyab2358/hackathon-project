import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# ==== Step 1: Load Data ====
train_df = pd.read_csv("train_clean.csv")
val_df = pd.read_csv("val_clean.csv")

print("✅ Loaded datasets")
print("DEBUG: Raw train label counts:\n", train_df['label'].value_counts())
print("DEBUG: Raw val label counts:\n", val_df['label'].value_counts())

# ==== Step 2: Drop NaNs (only hyp is required) ====
train_df = train_df.dropna(subset=["hyp"])
val_df = val_df.dropna(subset=["hyp"])

train_df = train_df[train_df['label'].isin([0, 1])]
val_df = val_df[val_df['label'].isin([0, 1])]

print("\n✅ Cleaned datasets")
print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("Train label counts:\n", train_df['label'].value_counts())
print("Val label counts:\n", val_df['label'].value_counts())

# ==== Step 3: TF-IDF Features ====
tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(train_df['hyp'])
X_val = tfidf.transform(val_df['hyp'])

y_train = train_df['label']
y_val = val_df['label']

# ==== Step 4: Train Model (with class balancing) ====
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)

# ==== Step 5: Evaluate ====
y_pred = clf.predict(X_val)
print("\n✅ Evaluation on Validation Set:")
print(classification_report(y_val, y_pred, digits=4))

# ==== Step 6: Save Model & Vectorizer ====
joblib.dump(clf, "hallucination_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\n✅ Model and vectorizer saved as hallucination_model.pkl & tfidf_vectorizer.pkl")
