
## pipeline to preprocess data

1. train questions and create vocab
```
python3.6 preprocess_vg_questions.py --input_questions_json /data/sjx/VQA-Exp/raw_vg_data/train_question.json --output_pt /data/sjx/VQA-Exp/data/vg_train_questions.pt --vocab_json /data/sjx/VQA-Exp/data/vg_vocab.json --mode train

python3.6 preprocess_questions.py --input_questions_json /data/sjx/dataset/vqa-dataset/Questions/v2_OpenEnded_mscoco_train2014_questions.json --input_annotations_json /data/sjx/dataset/vqa-dataset/Annotations/v2_mscoco_train2014_annotations.json --output_pt /data/sjx/VQA-Exp/data/train_questions.pt --vocab_json /data/sjx/VQA-Exp/data/vocab.json --mode train

python3.6 preprocess_questions.py --input_questions_json /data1/jiaxin/dataset/vqa/v2_OpenEnded_mscoco_train2014_questions.json --input_annotations_json /data1/jiaxin/dataset/vqa/v2_mscoco_train2014_annotations.json --output_pt /data1/jiaxin/exp/vqa/data/train_questions.pt --vocab_json /data1/jiaxin/exp/vqa/data/vocab.json --mode train
```

2. val questions. load generated vocab and will not change it
```
python3.6 preprocess_vg_questions.py --input_questions_json /data/sjx/VQA-Exp/raw_vg_data/val_question.json --output_pt /data/sjx/VQA-Exp/data/vg_val_questions.pt --vocab_json /data/sjx/VQA-Exp/data/vg_vocab.json --mode val

python3.6 preprocess_questions.py --input_questions_json /data/sjx/dataset/vqa-dataset/Questions/v2_OpenEnded_mscoco_val2014_questions.json --input_annotations_json /data/sjx/dataset/vqa-dataset/Annotations/v2_mscoco_val2014_annotations.json --output_pt /data/sjx/VQA-Exp/data/val_questions.pt --vocab_json /data/sjx/VQA-Exp/data/vocab.json --mode val

python3.6 preprocess_questions.py --input_questions_json /data1/jiaxin/dataset/vqa/v2_OpenEnded_mscoco_val2014_questions.json --input_annotations_json /data1/jiaxin/dataset/vqa/v2_mscoco_val2014_annotations.json --output_pt /data1/jiaxin/exp/vqa/data/val_questions.pt --vocab_json /data1/jiaxin/exp/vqa/data/vocab.json --mode val
```

3. train scenes. update vocab with edge_token_to_idx and merge attribute tokens into question_token_to_idx
```
python3.6 preprocess_vg_scene.py --input_scene_json /data/sjx/VQA-Exp/raw_vg_data/train_scene.json --output_vocab_json /data/sjx/VQA-Exp/data/vg_vocab.json --output_scene_pt /data/sjx/VQA-Exp/data/vg_train_sg.pt --mode train

python3.6 preprocess_scene.py --input_scene_json /data/sjx/VQA-Exp/raw_data/coco_train_sg.json --output_vocab_json /data/sjx/VQA-Exp/data/vocab.json --output_scene_pt /data/sjx/VQA-Exp/data/coco_train_sg.pt --mode train

python3.6 preprocess_scene.py --input_scene_json /data1/jiaxin/exp/vqa/data/coco_train_sg.json --output_vocab_json /data1/jiaxin/exp/vqa/data/vocab.json --output_scene_pt /data1/jiaxin/exp/vqa/data/coco_train_sg.pt --mode train
```

4. val scenes. 
```
python3.6 preprocess_vg_scene.py --input_scene_json /data/sjx/VQA-Exp/raw_vg_data/val_scene.json --output_vocab_json /data/sjx/VQA-Exp/data/vg_vocab.json --output_scene_pt /data/sjx/VQA-Exp/data/vg_val_sg.pt --mode val

python3.6 preprocess_scene.py --input_scene_json /data/sjx/VQA-Exp/raw_data/coco_val_sg.json --output_vocab_json /data/sjx/VQA-Exp/data/vocab.json --output_scene_pt /data/sjx/VQA-Exp/data/coco_val_sg.pt --mode val

python3.6 preprocess_scene.py --input_scene_json /data1/jiaxin/exp/vqa/data/coco_val_sg.json --output_vocab_json /data1/jiaxin/exp/vqa/data/vocab.json --output_scene_pt /data1/jiaxin/exp/vqa/data/coco_val_sg.pt --mode val
```


## preprocess features
```
python3.6 preprocess_features.py --input_tsv_folder /data/sjx/dataset/vqa-dataset/ --output_h5 /data/sjx/VQA-Exp/data/trainval_feature.h5

python3.6 preprocess_features.py --input_tsv_folder /data1/jiaxin/dataset/coco_trainval_36 --output_h5 /data1/jiaxin/exp/vqa/data/trainval_feature.h5

```
