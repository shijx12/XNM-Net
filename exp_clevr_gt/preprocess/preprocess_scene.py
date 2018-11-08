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
att_nodes = ["small", "large", "gray", "blue", "brown", "yellow", "red", "green", "purple", "cyan", "rubber", "metal", "cube", "sphere", "cylinder"]
att_nodes2idx = { n:i for i,n in enumerate(att_nodes) } # idx should plus len(objects)
edge_wtoi = {
    '<NULL>': 0, # no edge
    'left': 1, 
    'right': 2, 
    'front': 3, 
    'behind': 4,
    'size': 5,
    'color': 6,
    'material': 7,
    'shape': 8,
    }

def get_edge_triple(edge_wtoi, objects, relations):
    M = []
    num_obj = len(objects)
    for i, obj in enumerate(objects):
        for field in att_fields: # 4 attribute edges
            j = num_obj + att_nodes2idx[obj[field]] # append attribute nodes after object nodes
            M.append((i, edge_wtoi[field], j))
    for cat, ships in relations.items(): # 4 categories of relationship
        for i, js in enumerate(ships):
            for j in js:
                # NOTE: (i, left, j) if j is on the left of i
                M.append((i, edge_wtoi[cat], j))
    return M

def get_graph_matrix(edge_wtoi, objects, relations):
    '''
    Args:
    '''
    triple = get_edge_triple(edge_wtoi, objects, relations)
    n = len(objects) + len(att_nodes) # node number
    # connectivity matrix. there is no self-edge. conn_M[i][j]=1 if there is an edge between i and j.
    conn_M = np.zeros((n, n))
    # count numbers of relationships between each pair
    rel_count = {}
    for i in range(len(triple)):
        pair = (triple[i][0], triple[i][2])
        rel_count[pair] = rel_count.get(pair, 0) + 1
    # max number of relationship. For those pairs which have less relationships, we will pad with 0
    num_rel = max(list(rel_count.values()))
    # edge matrix. edge_M[i][j]=[k1, k2, ...] means that the types of the edge between i and j are [k1, k2, ...], where 0 is placeholder.
    edge_M = np.zeros((n, n, num_rel))
    rel_count = { k:0 for k in rel_count } # record the number of occurrence of each pair
    for i in range(len(triple)):
        a, b = triple[i][0], triple[i][2]
        conn_M[a][b] = conn_M[b][a]= 1
        edge_M[a][b][rel_count[(a, b)]] = triple[i][1]
        rel_count[(a, b)] += 1
    return conn_M, edge_M

def get_onehot_attributes_object(obj):
    res = []
    att_values = [obj[field] for field in att_fields] # all attribute values of current object
    res = [int(att in att_values) for att in att_nodes] # one-hot vector
    res = np.asarray(res)
    return res

def get_onehot_attributes_objects(objects):
    # NOTE: assign zero vector to object nodes, so they make no contributions to the prediction
    object_nodes_onehot = np.zeros((len(objects), len(att_nodes)))
    # one-hot vectors of attribute nodes form an identity matrix
    att_nodes_onehot = np.eye(len(att_nodes))
    res = np.concatenate([object_nodes_onehot, att_nodes_onehot], axis=0)
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
    # att nodes
    for att in att_nodes:
        descriptions.append(att)
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

    # relationship category label <-> index
    vocab = {
        'edge_token_to_idx': edge_wtoi,
    }
    if args.vocab_json:
        if os.path.exists(args.vocab_json):
            old_vocab = json.load(open(args.vocab_json))
            vocab.update(old_vocab)
            print("Update existed vocab")
        print("Write vocab to %s" % args.vocab_json)
        with open(args.vocab_json, 'w') as f:
            json.dump(vocab, f, indent=4)

    print('Construct')
    conn_matrixes = {} 
    edge_matrixes = {}
    vertex_vectors = {} # np array
    scene_descs = {} # list of str
    for scene in ori_scenes:
        image_index = scene['image_index']
        conn_M, edge_M = get_graph_matrix(edge_wtoi, scene['objects'], scene['relationships'])
        scene_vertex_vector = get_onehot_attributes_objects(scene['objects'])
        scene_desc = get_descriptions(scene['objects'], scene['relationships']) # desc of each node for debug
        
        assert image_index not in scene_descs
        conn_matrixes[image_index] = conn_M
        edge_matrixes[image_index] = edge_M
        vertex_vectors[image_index] = scene_vertex_vector
        scene_descs[image_index] = scene_desc

    #print('Simple test...')
    #assert(np.any(scenes['train'][0]['edge_triples'][0:4] == [[0,4,7], [0,5,9], [0,6,16], [0,7,18]]))
    #assert(np.any(scenes['train'][0]['v_h'][0:2] == [[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]]))

    print('Writing output')
    with open(args.output_scene, 'wb') as f:
        # cannot convert to np.array due to the difference between matrix shapes
        pickle.dump(conn_matrixes, f)
        pickle.dump(edge_matrixes, f)
        pickle.dump(vertex_vectors, f)
        pickle.dump(scene_descs, f)




if __name__ == '__main__':
    main()
