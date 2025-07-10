import pandas as pd

# Replace this with the actual full path or relative path to AllLabels.csv
csv_path = "/Users/theodorechan/Downloads/DAiSEE/Labels/AllLabels.csv"

all_df = pd.read_csv(csv_path)

# Clean column names and extract engagement
all_df.columns = [col.strip() for col in all_df.columns]
all_df['video_id'] = all_df['ClipID'].str.replace('.avi', '', regex=False)
all_df = all_df[['video_id', 'Engagement']].rename(columns={"Engagement": "label"})

# Drop missing values just in case
all_df = all_df.dropna(subset=['label'])

# Save
all_df.to_csv("data/final_labels.csv", index=False)

print(f"✅ Saved final_labels.csv with {len(all_df)} engagement labels")
