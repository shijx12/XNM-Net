import torch
import h5py
import numpy as np
import argparse
import pickle
from tqdm import tqdm

from tbd.module_net import load_tbd_net
from utils.clevr import load_vocab
from itertools import product
from IPython import embed


def dfs(mask, flag, out, i, j, h, w):
    if flag[i, j] == 1 or mask[i, j] == 0:
        return
    flag[i,j] = 1
    out[i, j] = 1
    up, down, left, right = max(i-1,0), min(i+1,h-1), max(j-1,0), min(j+1,w-1)
    for x, y in ((up,j), (down,j), (i,left), (i,right)):
        dfs(mask, flag, out, x, y, h, w)


def search(mask, h, w):
    results = []
    flag = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            out = np.zeros((h, w))
            dfs(mask, flag, out, i, j, h, w)
            if np.sum(out) > 1:
                results.append(out)
    return results


def main(args, module_net, device):
    color_modules = [ module for name, module in module_net.function_modules.items() if 'filter_color' in name ]
    shape_modules = [ module for name, module in module_net.function_modules.items() if 'filter_shape' in name ]
    size_modules = [ module for name, module in module_net.function_modules.items() if 'filter_size' in name ]
    material_modules = [ module for name, module in module_net.function_modules.items() if 'filter_material' in name ]

    stem = module_net.stem
    ones = torch.ones((1, 1, args.num_grid, args.num_grid)).to(device)
    coord_matrix = np.asarray([[[i,j] for j in range(args.num_grid)] for i in range(args.num_grid)])
    results = []
    with h5py.File(args.input_h5, 'r') as f:
        for index in tqdm(range(len(f['features']))):
            object_feature = []
            object_coord = []
            object_mask = []
            feature_np = f['features'][index] # (512,28,28)
            feature = torch.FloatTensor(feature_np).to(device).unsqueeze(0) # (1,512,28,28)
            feature = stem(feature)
            for modules in product(color_modules, shape_modules, size_modules, material_modules):
                attention = ones
                for module in modules:
                    attention = module(feature, attention)
                attention = attention.squeeze().cpu().data.numpy()
                binary_attention = (attention > args.threshold).astype(np.uint8)
                if np.sum(binary_attention) <= 1:
                    continue
                #print(binary_attention)
                #print(index)
                #embed()
                obj_masks = search(binary_attention, args.num_grid, args.num_grid)
                for mask in obj_masks:
                    weight_mask = np.reshape(mask * attention, (1, args.num_grid, args.num_grid)) # (1,28,28)
                    obj = np.sum(weight_mask * feature_np, axis=(1,2)) / np.sum(weight_mask)  # (512,)
                    object_feature.append(obj)
                    weight_mask = np.reshape(weight_mask, (args.num_grid, args.num_grid, 1)) # (28,28,1)
                    coord = np.sum(weight_mask * coord_matrix, axis=(0,1)) / np.sum(weight_mask) # (2,)
                    object_coord.append(coord)
                    # object_mask.append(mask)
            assert len(object_feature) > 0
            object_feature = np.asarray(object_feature)
            object_coord = np.asarray(object_coord)
            object_mask = np.asarray(object_mask)
            results.append({
                # 'mask': object_mask,
                'feature': object_feature,
                'coord': object_coord,
                })
            
    with open(args.output_pt, 'wb') as f:
        pickle.dump(results, f)

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_h5', required=True)
    parser.add_argument('--output_pt', required=True)
    parser.add_argument('--num_grid', default=28, type=int)
    parser.add_argument('--threshold', default=0.4, type=float)
    args = parser.parse_args()

    device = 'cuda'
    model_path = 'models/clevr-reg-hres.pt'
    vocab_path = 'data/vocab.json'
    module_net = load_tbd_net(model_path, load_vocab(vocab_path))
    main(args, module_net, device)
