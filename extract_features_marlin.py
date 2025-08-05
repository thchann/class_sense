from marlin_pytorch import Marlin
from tqdm import tqdm
import torch
import os
import sys

base_path = '/home/surbhi/ximi/'

def read_file(path):
    try:
        with open(path) as f:
            return [i.strip('\n') for i in f.readlines()]
    except:
        return []

def log(path, content):
    with open(path, 'a') as f:
        f.write(content + '\n')

def load_model(feature_type):
    return Marlin.from_file(f"marlin_vit_{feature_type}_ytf", f"marlin_models/marlin_vit_{feature_type}_ytf.encoder.pt")

def main(feature_type, rank):
    model = load_model(feature_type).cuda()

    _todo = read_file(f'todo{rank}.txt')
    processed = set(read_file(f'{feature_type}_processed_{rank}.txt'))
    proc_files = set(f.strip('.pt') for f in os.listdir(f'marlin_features_{feature_type}/'))

    todo = set(f'chunks/{f}' for f in _todo) - processed - set(f'chunks/{f}' for f in proc_files)

    for vname in tqdm(todo):
        try:
            path = os.path.join(base_path, vname)
            features = model.extract_video(path, crop_face=True)
            torch.save(features, f"marlin_features_{feature_type}/{os.path.basename(vname)}.pt")
            log(f'{feature_type}_processed_{rank}.txt', vname)
        except Exception as e:
            print(e)
            log(f'{feature_type}_errors_{rank}.txt', vname)

if __name__ == '__main__':
    main('large', 'ESC')
