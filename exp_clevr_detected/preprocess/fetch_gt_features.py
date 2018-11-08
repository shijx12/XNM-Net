import h5py
import numpy as np
import argparse
import pickle
import json
import os
from tqdm import tqdm
from PIL import Image


def main(args):
    scenes = json.load(open(args.scene_json))['scenes']
    scene_dict = { s['image_index']:s for s in scenes }
    size_to_feat = { 'large':0, 'small':0.6 }
    shape_to_feat = { 'cube':0, 'sphere':0.3, 'cylinder':0.6 }
    results = []
    with h5py.File(args.feature_h5, 'r') as f:
        for index in tqdm(range(len(f['features']))):
            object_feature, object_coord = [], []
            scene = scene_dict[index]
            image_filename = os.path.join(args.image_dir, scene['image_filename'])
            im = Image.open(image_filename)
            width, height = im.size
            feature = f['features'][index] # (512,28,28)
            for obj in scene['objects']:
                x, y = obj['pixel_coords'][:2]
                i = int(y/height * args.num_grid)
                j = int(x/width * args.num_grid)
                v1 = feature[:, i, j]
                v2 = np.asarray([size_to_feat[obj['size']]]*20 + [shape_to_feat[obj['shape']]]*20)
                v = np.concatenate((v1, v2), axis=0) # (512+20+20=552)
                v /= np.sqrt(np.linalg.norm(v, ord=2))
                object_feature.append(v)
                object_coord.append(obj['3d_coords'])
            results.append({
                'feature': np.asarray(object_feature),
                'coord': np.asarray(object_coord),
                })
            
    with open(args.output_pt, 'wb') as f:
        pickle.dump(results, f)

            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--feature_h5', required=True)
    parser.add_argument('--scene_json', required=True)
    parser.add_argument('--output_pt', required=True)
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--num_grid', default=28, type=int)
    args = parser.parse_args()

    main(args)
