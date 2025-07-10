import os

marlin_ids = set([os.path.splitext(f)[0] for f in os.listdir("/Users/theodorechan/Downloads/marlin_embeddings") if f.endswith('.npy')])
openface_ids = set([os.path.splitext(f)[0] for f in os.listdir("/Users/theodorechan/Downloads/openface_output") if f.endswith('.csv')])

missing_openface = sorted(marlin_ids - openface_ids)

print(f"Total MARLIN files: {len(marlin_ids)}")
print(f"Total OpenFace CSVs: {len(openface_ids)}")
print(f"Missing OpenFace CSVs: {len(missing_openface)}")
print("Examples:", missing_openface[:10])
