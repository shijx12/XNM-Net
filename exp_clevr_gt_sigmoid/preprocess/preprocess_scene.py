# process scene graph annotations of CLEVR
# each attribute value (e.g. 'blue') is considered as a **special node**, whose v_h is one-hot vector
# in every scene, att nodes are after object nodes and index from len(objects)
# each attribute field/type (e.g. 'color') is considered as a special relationship
# object attributes are considered as edges from object nodes to att nodes

import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import json
import pickle 
import argparse
import numpy as np

att_fields = ['size', 'color', 'material', 'shape']
att_values = ["small", "large", "gray", "blue", "brown", "yellow", "red", "green", "purple", "cyan", "rubber", "metal", "cube", "sphere", "cylinder"]
edge_values = ['left', 'right', 'front', 'behind']


def get_graph_matrix(edge2idx, objects, relations):
    '''
    Args:
    '''
    triples = []
    for cat, ships in relations.items(): # 4 categories of relationship
        for i, js in enumerate(ships):
            for j in js:
                # NOTE: (i, left, j) if j is on the left of i
                triples.append((i, edge2idx[cat], j))
    # count numbers of relationships between each pair
    rel_count = {}
    for i in range(len(triples)):
        pair = (triples[i][0], triples[i][2])
        rel_count[pair] = rel_count.get(pair, 0) + 1
    # max number of relationship. For those pairs which have less relationships, we will pad with 0
    num_rel = max(list(rel_count.values()))
    # edge matrix. edge_M[i][j]=[k1, k2, ...] means that the types of the edge between i and j are [k1, k2, ...], where 0 is placeholder.
    n = len(objects)
    edge_M = np.zeros((n, n, num_rel))
    rel_count = { k:0 for k in rel_count } # record the number of occurrence of each pair
    for i in range(len(triples)):
        a, b = triples[i][0], triples[i][2]
        edge_M[a][b][rel_count[(a, b)]] = triples[i][1]
        rel_count[(a, b)] += 1
    return edge_M

def get_object_attributes(objects, att2idx):
    res = []
    for obj in objects:
        values = [att2idx[obj[field]] for field in att_fields] # all attribute values of current object
        res.append(values)
    res = np.asarray(res) # (n, 4)
    return res


def get_descriptions(objects, relations):
    # get description list for all object nodes and attribute nodes
    # for visualize
    descriptions = []
    # object nodes
    for i, obj in enumerate(objects):
        description = '%d: ' % i
        description += ', '.join([obj[field] for field in att_fields]) + '. '
        for r in ['left', 'right', 'front', 'behind']:
            description += ' %s:%s. ' % (r, ','.join([str(j) for j in relations[r][i]]))
        descriptions.append(description)
    return descriptions



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-scene', required=True, help='scene graph json file')
    parser.add_argument('--vocab-json', required=True, help='vocab file')
    parser.add_argument('--output-scene', required=True, help='output file')
    args = parser.parse_args()

    print('Loading data')
    with open(args.input_scene, 'r') as f:
        ori_scenes = json.load(f)['scenes']

    if not os.path.exists(args.vocab_json):
        raise Exception("must give vocab.json produced by questions")
    vocab = json.load(open(args.vocab_json))
    # NOTE: we add all attribute values and edge categories into one common vocab
    for v in att_values:
        if v not in vocab['question_token_to_idx']:
            vocab['question_token_to_idx'][v] = len(vocab['question_token_to_idx'])
    for v in edge_values:
        if v not in vocab['question_token_to_idx']:
            vocab['question_token_to_idx'][v] = len(vocab['question_token_to_idx'])
    att2idx = { v: vocab['question_token_to_idx'][v] for v in att_values }
    edge2idx = { v: vocab['question_token_to_idx'][v] for v in edge_values }
    print("Update existed vocab")
    with open(args.vocab_json, 'w') as f:
        json.dump(vocab, f, indent=4)

    print('Construct')
    edge_matrixes = {}
    vertex_vectors = {} # np array
    scene_descs = {} # list of str
    for scene in ori_scenes:
        image_index = scene['image_index']
        edges = get_graph_matrix(edge2idx, scene['objects'], scene['relationships'])
        vertexes = get_object_attributes(scene['objects'], att2idx)
        scene_desc = get_descriptions(scene['objects'], scene['relationships']) # desc of each node for debug
        
        assert image_index not in scene_descs
        edge_matrixes[image_index] = edges
        vertex_vectors[image_index] = vertexes
        scene_descs[image_index] = scene_desc

    #print('Simple test...')
    #assert(np.any(scenes['train'][0]['edge_triples'][0:4] == [[0,4,7], [0,5,9], [0,6,16], [0,7,18]]))
    #assert(np.any(scenes['train'][0]['v_h'][0:2] == [[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]]))

    print('Writing output')
    with open(args.output_scene, 'wb') as f:
        # cannot convert to np.array due to the difference between matrix shapes
        pickle.dump(edge_matrixes, f)
        pickle.dump(vertex_vectors, f)
        pickle.dump(scene_descs, f)




if __name__ == '__main__':
    main()
