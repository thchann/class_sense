# Extract_features_marlin_safe and Extract_features_marlin were both in the original repo.
# Original repo link: https://github.com/engagenet/engagenet_baselines
# These are some differences between the two:

# Reads input video paths directly from todo<rank>.txt (no base_path/chunks/ prefix)
# Outputs features into subject-specific subfolders: pafe/data/<subject>/marlin_features_<feature_type>/
# Skips already-processed files using ONLY the processed log (does not check output dir)
# Feature_type is passed via command-line args (default 'small'), rank defaults to 0
# Extracts subject id with vname.split('/')[1] to organize outputs
# Includes unused functions (load_labels, multiprocessing) not called in main flow

from marlin_pytorch import Marlin
import torch.multiprocessing as mp

from tqdm import tqdm
import torch
import os
import sys
import pandas as pd

base_path = '/home/surbhi/ximi/marlin_models'

def read_file(path):
    try:
        with open(path) as f:
            dat = [i.strip('\n') for i in f.readlines()]
    except:
        return []
    return dat

def log(path, content):
    with open(path, 'a') as f:
        f.write(content)
        f.write('\n')
        
def load_model(feature_type):
    model = Marlin.from_file(f"marlin_vit_{feature_type}_ytf", f"marlin_models/marlin_vit_{feature_type}_ytf.encoder.pt")
    return model

def load_labels():

    labels = pd.read_csv('/home/surbhi/ximi/final_labels.csv')
    labels = labels[labels['label']!='SNP(Subject Not Present)']
    return labels

marlin_feature_type = 'small'
def main(marlin_feature_type, rank):
    model = load_model(marlin_feature_type)
    model = model.cuda()
    
    _todo_ = read_file(f'todo{rank}.txt')
    errors = []
    processed = read_file(f'{marlin_feature_type}_processed_{rank}.txt')

    todo = set([f for f in _todo_]) - set(processed)
    todo = list(set(todo) - set(processed))
    for vname in tqdm(todo):
        try:
            
            sub = vname.split('/')[1]
            
            features = model.extract_video(vname, crop_face=True)
            feature_dir = f'pafe/data/{sub}/marlin_features_{marlin_feature_type}'
            
            torch.save(features, f"{feature_dir}/{vname.split('/')[-1]}.pt")
            log(f'{marlin_feature_type}_processed_{rank}.txt', vname)

        except Exception as e:
            print (e)
            log(f'{marlin_feature_type}_errors_{rank}.txt', vname)
            
if __name__ == '__main__':
    args = sys.argv
    main(args[1], 0)
    
