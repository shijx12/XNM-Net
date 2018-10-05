import json
from tqdm import tqdm
import os
import argparse


def vg_intersect_coco():
    results = []
    f = json.load(open(vg_meta_file))
    for img in f:
        if img['coco_id']:
            results.append(img)
    return results

def vqa_intersect_vg(vqa_annotation_file, vqa_question_file, vg_scenegraph, output_scene_json, output_question_json):
    results = []
    f = json.load(open(vqa_annotation_file))
    questions = { q['question_id']:q for q in json.load(open(vqa_question_file))['questions'] }
    # q = {'question': str, 'question_id': int, 'image_id': int}
    vg_annotations = vg_intersect_coco()
    vg_id_to_scenegraph = { scene['image_id']:scene for scene in vg_scenegraph  }
    # exclude those scenes that miss objects
    vg_coco_ids = { a['coco_id']:a for a in vg_annotations if len(vg_id_to_scenegraph[a['image_id']]['objects']) > 0 }
    for ann in tqdm(f['annotations']):
        if ann['image_id'] in vg_coco_ids:
            ann['vg_id'] = vg_coco_ids[ann['image_id']]['image_id']
            ann['question'] = questions[ann['question_id']]['question'] # add 'question'
            assert questions[ann['question_id']]['image_id'] == ann['image_id']
            results.append(ann)
    print('%s has %d overlap images, containing %d questions, with vg' % (vqa_annotation_file, len({_['image_id'] for _ in results}), len(results)))
    # filter vg scene graph
    vg_id_to_coco_id = { _['vg_id']:_['image_id'] for _ in results }
    used_vg_scenegraph = []
    for s in vg_scenegraph:
        if s['image_id'] in vg_id_to_coco_id:
            s['image_id'] = vg_id_to_coco_id[s['image_id']] # convert vg_id to coco_id
            used_vg_scenegraph.append(s)
    json.dump(used_vg_scenegraph, open(output_scene_json, 'w'))
    json.dump(results, open(output_question_json, 'w'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vg_home', default='/data/sjx/dataset/visualgenome/')
    parser.add_argument('--vqa_home', default='/data/sjx/dataset/vqa-dataset/')
    parser.add_argument('--output_json_dir', required=True, help='a folder to store scene json and question json files')
    args = parser.parse_args()

    vg_meta_file = os.path.join(args.vg_home, 'image_data.json')
    vg_scenegraph_file = os.path.join(args.vg_home, 'scene_graphs.json')

    vqa_train_annotation_file = os.path.join(args.vqa_home, 'Annotations/mscoco_train2014_annotations.json')
    vqa_val_annotation_file = os.path.join(args.vqa_home, 'Annotations/mscoco_val2014_annotations.json')
    vqa_train_question_file = os.path.join(args.vqa_home, 'Questions/OpenEnded_mscoco_train2014_questions.json')
    vqa_val_question_file = os.path.join(args.vqa_home, 'Questions/OpenEnded_mscoco_val2014_questions.json')

    if not os.path.exists(args.output_json_dir):
        os.mkdir(args.output_json_dir)
    vg_scenegraph = json.load(open(vg_scenegraph_file))
    vqa_intersect_vg(vqa_train_annotation_file, vqa_train_question_file, vg_scenegraph, 
        os.path.join(args.output_json_dir, 'train_scene.json'), os.path.join(args.output_json_dir, 'train_question.json'))
    vg_scenegraph = json.load(open(vg_scenegraph_file)) # the old one has changed
    vqa_intersect_vg(vqa_val_annotation_file, vqa_val_question_file, vg_scenegraph,
        os.path.join(args.output_json_dir, 'val_scene.json'), os.path.join(args.output_json_dir, 'val_question.json'))
