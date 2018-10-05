# process scene graph annotations of VQA
# each attribute value (e.g. 'blue') is considered as a **special node**, whose v_h is one-hot vector
# in every scene, att nodes are after object nodes and index from len(objects)
# each unique attribute value is considered as a special relationship
# object attributes are considered as edges from object nodes to att nodes

import os
import json
import pickle 
import argparse
import numpy as np
from tqdm import tqdm
from IPython import embed


def get_edge_triple(token_wtoi, objects, relations):
    object_id_to_index = { obj['object_id']:i for i,obj in enumerate(objects) }
    M = []
    num_obj = len(objects)
    current_attributes = {}
    current_relations = {'<NULL>':0, 'name':1}
    for i, obj in enumerate(objects):
        if 'attributes' not in obj:
            continue
        for token in obj['attributes']: # attribute edges
            if token not in token_wtoi:
                continue
            if token not in current_attributes:
                current_attributes[token] = num_obj + len(current_attributes) # attribute node index, start from num_obj
            if token not in current_relations:
                current_relations[token] = len(current_relations)
            M.append((i, current_relations[token], current_attributes[token]))
        for token in obj['names']:
            if token not in token_wtoi:
                continue
            if token not in current_attributes: # name is a special attribute
                current_attributes[token] = num_obj + len(current_attributes)
            M.append((i, current_relations['name'], current_attributes[token]))
    for rel in relations:
        if rel['predicate'] not in token_wtoi:
            continue
        token = rel['predicate']
        i = object_id_to_index[rel['subject_id']]
        j = object_id_to_index[rel['object_id']]
        if token not in current_relations:
            current_relations[token] = len(current_relations)
        M.append((i, current_relations[token], j))

    inverted_current_attributes = { i:w for w,i in current_attributes.items() }
    inverted_current_relations = { i:w for w,i in current_relations.items() }
    # attribute indexes of current graph nodes, including object nodes and attribute nodes
    # will be fed into attribute embedding layer, to get the vertex embedding matrix
    v_indexes = [0] * (num_obj + len(current_attributes)) # 0 for object nodes
    for i in range(num_obj, num_obj+len(current_attributes)):
        token = inverted_current_attributes[i]
        v_indexes[i] = token_wtoi[token] # fetch index of attribute nodes
    e_indexes = [token_wtoi[inverted_current_relations[i]] for i in range(len(inverted_current_relations))]
    assert e_indexes[0] == token_wtoi['<NULL>']
    return M, v_indexes, e_indexes


def get_graph_matrix(token_wtoi, objects, relations):
    triple, v_indexes, e_indexes = get_edge_triple(token_wtoi, objects, relations)
    n = len(v_indexes) # node number
    # connectivity matrix. there is no self-edge. conn_M[i][j]=1 if there is an edge between i and j.
    conn_M = np.zeros((n, n))
    # edge matrix. edge_M[i][j]=r means that the type of the edge between i and j is e_indexes[r].
    edge_M = np.zeros((n, n, 1))
    for a,r,b in triple:
        conn_M[a][b] = conn_M[b][a]= 1
        edge_M[a][b][0] = r
    return conn_M, edge_M, v_indexes, e_indexes


def get_descriptions(objects, relations, token_itow, v_indexes):
    # get description list for all object nodes and attribute nodes
    # for visualize
    descriptions = []
    object_id_to_index = { obj['object_id']:i for i,obj in enumerate(objects) }
    # object nodes
    for i, obj in enumerate(objects):
        description = '%d: ' % i
        if 'attributes' in obj:
            description += ', '.join(obj['names'] + obj['attributes']) + '. '
        else:
            description += ', '.join(obj['names']) + '. '
        for rel in relations:
            if rel['subject_id'] == obj['object_id']:
                description += ' %s:%d. ' % (rel['predicate'], object_id_to_index[rel['object_id']])
        descriptions.append(description)
    # att nodes
    for i in range(len(objects), len(v_indexes)):
        token = token_itow[v_indexes[i]]
        descriptions.append(token)
    return descriptions



def main(args):
    print('Loading data')
    ori_scenes = json.load(open(args.input_scene_json, 'r'))

    print('Loading vocab')
    assert args.output_vocab_json and os.path.exists(args.output_vocab_json)
    vocab = json.load(open(args.output_vocab_json))
    assert 'question_token_to_idx' in vocab and vocab['question_token_to_idx']['<NULL>']==0

    if args.mode == 'train':
        print("train mode, update vocab")
        token_cnt = {}
        for scene in ori_scenes:
            for obj in scene['objects']:
                if 'attributes' not in obj:
                    obj['attributes'] = []
                for token in obj['attributes']:
                    token_cnt[token] = token_cnt.get(token, 0) + 1
                for name in obj['names']: 
                    token_cnt[token] = token_cnt.get(token, 0) + 1
            for rel in scene['relationships']:
                token = rel['predicate']
                token_cnt[token] = token_cnt.get(token, 0) + 1

        question_token_num = len(vocab['question_token_to_idx']) 
        token_wtoi = vocab['question_token_to_idx'] # 0 for object nodes, merge attribute and relation into question tokens
        if 'name' not in token_wtoi:
            token_wtoi['name'] = len(token_wtoi)
        for token, count in token_cnt.items():
            if count >= args.token_min_count and token not in token_wtoi:
                token_wtoi[token] = len(token_wtoi)
        
        print("Write vocab to %s" % args.output_vocab_json)
        with open(args.output_vocab_json, 'w') as f:
            json.dump(vocab, f, indent=4)
        print('new tokens: %d' % (len(token_wtoi) - question_token_num))
    else:
        token_wtoi = vocab['question_token_to_idx']

    print('Construct')
    image_indexes = []
    conn_matrixes = [] 
    edge_matrixes = []
    vertex_indexes = []
    edge_indexes = []
    scene_descs = {} # list of str
    token_itow = { i:w for w,i in token_wtoi.items() }
    cnt = 0
    for scene in tqdm(ori_scenes):
        if len(scene['objects']) == 0:
            cnt += 1
            conn_M = np.zeros((1,1))
            edge_M = np.zeros((1,1,1))
            v_indexes = [0]
            e_indexes = [0]
            scene_desc = []
        else:
            conn_M, edge_M, v_indexes, e_indexes = get_graph_matrix(token_wtoi, scene['objects'], scene['relationships'])
            scene_desc = get_descriptions(scene['objects'], scene['relationships'], token_itow, v_indexes) # desc of each node for debug
        
        coco_id = int(scene['image_id'])
        assert coco_id not in scene_descs
        image_indexes.append(coco_id)
        conn_matrixes.append(conn_M)
        edge_matrixes.append(edge_M)
        vertex_indexes.append(v_indexes)
        edge_indexes.append(e_indexes)
        scene_descs[coco_id] = scene_desc
    print('number of scenes without objects: %d' % cnt)

    glove_matrix = None
    if args.mode == 'train':
        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)


    print('Writing output')
    with open(args.output_scene_pt, 'wb') as f:
        pickle.dump(image_indexes, f)
        pickle.dump(conn_matrixes, f)
        pickle.dump(edge_matrixes, f)
        pickle.dump(vertex_indexes, f)
        pickle.dump(edge_indexes, f)
        pickle.dump(scene_descs, f)
        pickle.dump(glove_matrix, f)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_scene_json', required=True, help='scene graph json file')
    parser.add_argument('--output_vocab_json', required=True, help='output vocab file')
    parser.add_argument('--token_min_count', default=3, type=int)
    parser.add_argument('--mode', choices=['train', 'val'], help='val mode vocab will not be updated')
    parser.add_argument('--output_scene_pt', required=True, help='output file')
    parser.add_argument('--glove_pt', default='/data/sjx/glove.840B.300d.py36.pkl', help='glove pickle file')
    args = parser.parse_args()
    main(args)
