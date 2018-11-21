# According to [https://github.com/Cyanogenoid/vqa-counting/blob/master/vqa-v2/preprocess-features.py]

import sys
import argparse
import base64
import os
import csv
import itertools

csv.field_size_limit(sys.maxsize)

import h5py
import torch.utils.data
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_h5', required=True)
    parser.add_argument('--input_tsv_folder', required=True, help='path to trainval_36 or test2015_36')
    parser.add_argument('--test', action='store_true', help='specified when processing test2015_36')
    args = parser.parse_args()
    assert os.path.isdir(args.input_tsv_folder)

    FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

    features_shape = (
        82783 + 40504 if not args.test else 81434,  # number of images in trainval or in test
        2048, # dim_vision,
        36, # 36 for fixed case, 100 for the adaptive case
    )
    boxes_shape = (
        features_shape[0],
        4,
        36,
    )

    path = args.output_h5
    with h5py.File(path, libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float32')
        coco_ids = fd.create_dataset('ids', shape=(features_shape[0],), dtype='int32')
        boxes = fd.create_dataset('boxes', shape=boxes_shape, dtype='float32')
        widths = fd.create_dataset('widths', shape=(features_shape[0],), dtype='int32')
        heights = fd.create_dataset('heights', shape=(features_shape[0],), dtype='int32')

        readers = []
        for filename in os.listdir(args.input_tsv_folder):
            if not '.tsv' in filename:
                continue
            full_filename = os.path.join(args.input_tsv_folder, filename)
            fd = open(full_filename, 'r')
            reader = csv.DictReader(fd, delimiter='\t', fieldnames=FIELDNAMES)
            readers.append(reader)

        reader = itertools.chain.from_iterable(readers)
        for i, item in enumerate(tqdm(reader, total=features_shape[0])):
            coco_ids[i] = int(item['image_id'])
            widths[i] = int(item['image_w'])
            heights[i] = int(item['image_h'])

            buf = base64.decodestring(item['features'].encode('utf8'))
            array = np.frombuffer(buf, dtype='float32')
            array = array.reshape((-1, 2048)).transpose()
            features[i, :, :array.shape[1]] = array

            buf = base64.decodestring(item['boxes'].encode('utf8'))
            array = np.frombuffer(buf, dtype='float32')
            array = array.reshape((-1, 4)).transpose()
            boxes[i, :, :array.shape[1]] = array


if __name__ == '__main__':
    main()
