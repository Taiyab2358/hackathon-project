import json
from collections import Counter

with open("train.model-aware.v2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

refs = [str(item.get("ref", "")).lower() for item in data]
counts = Counter(refs)

print("Unique ref values in train.model-aware.v2.json:")
for k, v in counts.items():
    print(f"{k}: {v}")
