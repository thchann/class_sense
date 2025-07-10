import pandas as pd

# Load labels
labels_df = pd.read_csv("data/final_labels.csv")
label_ids = set(labels_df["ClipID"].astype(str).str.replace(".avi", "", regex=False).str.replace(".mp4", "", regex=False).str.strip())

# Load train/val/test split lists
with open("data/train.txt") as f:
    train = [line.strip() for line in f]

with open("data/valid.txt") as f:
    valid = [line.strip() for line in f]

with open("data/test.txt") as f:
    test = [line.strip() for line in f]

# Combine all split IDs
split_ids = set(train + valid + test)

# Find IDs in split but missing in labels
missing_ids = split_ids - label_ids

# Save or print
print(f"❌ Missing {len(missing_ids)} video labels.")
with open("missing_ids.txt", "w") as f:
    for vid in sorted(missing_ids):
        f.write(f"{vid}\n")
