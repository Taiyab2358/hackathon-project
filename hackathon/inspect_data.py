import json

with open("train.model-agnostic.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total samples:", len(data))
print("First sample:\n", data[0].keys())  # show fields
print("\nFull first sample:\n", data[0])
