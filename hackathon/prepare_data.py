import json
import pandas as pd

# ==== Step 1: Load JSON ====
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Paths to datasets (make sure all 4 are in your folder!)
train_paths = ["train.model-agnostic.json", "train.model-aware.v2.json"]
val_paths = ["val.model-agnostic.json", "val.model-aware.v2.json"]

# ==== Step 2: Convert JSON to DataFrame ====
def json_to_df(data, split_name):
    records = []
    for i, item in enumerate(data):
        ref = str(item.get("ref", "")).lower()

        # Map ref to binary label
        if ref in ["hallucination", "unsupported", "src"]:
            label = 1   # hallucination
        elif ref in ["entailed", "supported", "faithful", "either", "tgt"]:
            label = 0   # factual
        else:
            label = -1  # unknown / skip

        records.append({
            "id": f"{split_name}_{i}",
            "src": item.get("src", ""),
            "tgt": item.get("tgt", ""),
            "hyp": item.get("hyp", ""),
            "task": item.get("task", ""),
            "model": item.get("model", ""),
            "label": label
        })
    return pd.DataFrame(records)

# Load & merge train
train_dfs = []
for path in train_paths:
    data = load_json(path)
    train_dfs.append(json_to_df(data, "train"))
train_df = pd.concat(train_dfs).reset_index(drop=True)

# Load & merge val
val_dfs = []
for path in val_paths:
    data = load_json(path)
    val_dfs.append(json_to_df(data, "val"))
val_df = pd.concat(val_dfs).reset_index(drop=True)

# ==== Step 3: Clean text ====
def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.strip().lower()

for col in ["src", "tgt", "hyp"]:
    train_df[col] = train_df[col].apply(clean_text)
    val_df[col] = val_df[col].apply(clean_text)

# ==== Step 4: Keep only labeled data ====
train_df = train_df[train_df['label'].isin([0, 1])].reset_index(drop=True)
val_df = val_df[val_df['label'].isin([0, 1])].reset_index(drop=True)

# ==== Step 5: Save ====
train_df.to_csv("train_clean.csv", index=False)
val_df.to_csv("val_clean.csv", index=False)

print("✅ Saved train_clean.csv and val_clean.csv")
print("Train label distribution:\n", train_df['label'].value_counts())
print("Val label distribution:\n", val_df['label'].value_counts())
