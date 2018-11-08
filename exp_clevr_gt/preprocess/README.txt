
## pipeline to preprocess data

1. train questions and create vocab
```
python3.6 preprocess_questions.py --input_questions_json /data/sjx/CLEVR_v1.0/questions/CLEVR_train_questions.json --output_pt_file /data/sjx/CLEVR-Exp/data/train_questions.pt --output_vocab_json /data/sjx/CLEVR-Exp/data/vocab.json

```

2. val questions. load generated vocab and will not change it
```
python3.6 preprocess_questions.py --input_questions_json /data/sjx/CLEVR_v1.0/questions/CLEVR_val_questions.json --output_pt_file /data/sjx/CLEVR-Exp/data/val_questions.pt --input_vocab_json /data/sjx/CLEVR-Exp/data/vocab.json

```

3. train scens. update vocab with edge_token_to_idx
```
 python3.6 preprocess_scene.py --input-scene /data/sjx/CLEVR_v1.0/scenes/CLEVR_train_scenes.json --vocab-json /data/sjx/CLEVR-Exp/data/vocab.json --output-scene /data/sjx/CLEVR-Exp/data/train_scenes.pt
```

4. val scenes. update vocab with edge_token_to_idx, which will actually not change it.
```
 python3.6 preprocess_scene.py --input-scene /data/sjx/CLEVR_v1.0/scenes/CLEVR_val_scenes.json --vocab-json /data/sjx/CLEVR-Exp/data/vocab.json --output-scene /data/sjx/CLEVR-Exp/data/val_scenes.pt
```
