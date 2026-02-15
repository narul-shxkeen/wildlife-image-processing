import json
from collections import Counter

# === 1. Load JSON file ===
with open("../image_categories_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# === 2. Initialize a counter for all labels ===
label_counter = Counter()

# === 3. Count occurrences of each label ===
for labels in data.values():
    if labels:  # skip None or empty lists
        label_counter.update(labels)

# === 4. Print results ===
print("Label counts:\n")
for label, count in label_counter.most_common():
    print(f"{label}: {count}")

# === 5. (Optional) Save to a JSON file ===
with open("label_counts.json", "w", encoding="utf-8") as f:
    json.dump(label_counter, f, indent=2, ensure_ascii=False)

print("\nSaved counts to label_counts.json")
