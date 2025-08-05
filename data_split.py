import pandas as pd

labels_df = pd.read_csv("data/final_labels.csv")
label_ids = set(labels_df["ClipID"].astype(str).str.replace(".avi", "", regex=False).str.replace(".mp4", "", regex=False).str.strip())

with open("data/train.txt") as f:
    train = [line.strip() for line in f]

with open("data/valid.txt") as f:
    valid = [line.strip() for line in f]

with open("data/test.txt") as f:
    test = [line.strip() for line in f]

split_ids = set(train + valid + test)

missing_ids = split_ids - label_ids

print(f"❌ Missing {len(missing_ids)} video labels.")
with open("missing_ids.txt", "w") as f:
    for vid in sorted(missing_ids):
        f.write(f"{vid}\n")
