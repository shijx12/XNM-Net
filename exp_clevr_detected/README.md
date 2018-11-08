
# pipeline to preprocess data

### preprocess CLEVR train questions and obtain two output files: train_questions.pt and vocab.json
```
cd preprocess
python preprocess_questions.py --input_questions_json /data4/CLEVR_v1.0/questions/CLEVR_train_questions.json --output_pt_file /data4/jiaxin/exp/CLEVR/data_clevr/train_questions.pt --output_vocab_json /data4/jiaxin/exp/CLEVR/data_clevr/vocab.json
```

### preprocess CLEVR val questions. Load generated vocab.json and will not change it
```
cd preprocess
python preprocess_questions.py --input_questions_json /data4/CLEVR_v1.0/questions/CLEVR_val_questions.json --output_pt_file /data4/jiaxin/exp/CLEVR/data_clevr/val_questions.pt --input_vocab_json /data4/jiaxin/exp/CLEVR/data_clevr/vocab.json
```

### detect salient objects with the help of [tbd-net](https://github.com/davidmascharka/tbd-nets).
1. clone their repo
2. download their pretrained high-resolution checkpoint by the following command
```
python utils/download_pretrained_models.py -m hres
```

3. extract CLEVR image features with the instructions [here](https://github.com/facebookresearch/clevr-iep/blob/master/TRAINING.md#preprocessing-clevr)
```
python scripts/extract_features.py \
    --input_image_dir data/CLEVR_v1.0/images/train \
    --output_h5_file data/train_features_28x28.h5 \
    --model_stage 2
python scripts/extract_features.py \
    --input_image_dir data/CLEVR_v1.0/images/val \
    --output_h5_file data/val_features_28x28.h5 \
    --model_stage 2
```
Note: `--model_stage 2` is necessary to extract high-resolution features.

**For detected setting**:

4. copy my `utils/find-salient.py` into the main directory of tbd project.

5. detect salient objects and coordinates using following commands:
```
python find-salient.py --input_h5 /data1/jiaxin/exp/CLEVR/data/train_features_28x28.h5 --output_pt /data1/jiaxin/exp/CLEVR/data/train_features_salient_thres0.4.pt
python find-salient.py --input_h5 /data1/jiaxin/exp/CLEVR/data/val_features_28x28.h5 --output_pt /data1/jiaxin/exp/CLEVR/data/val_features_salient_thres0.4.pt

```

**For gt grounding with visual features**:

4. fetch visual features of gt objects
```
python fetch_gt_features.py --feature_h5 /data4/jiaxin/exp/CLEVR/data_clevr/train_features_28x28.h5 --scene_json /data4/CLEVR_v1.0/scenes/CLEVR_train_scenes.json --image_dir /data4/CLEVR_v1.0/images/train --output_pt /data4/jiaxin/exp/CLEVR/data_clevr/train_gt_features.pt
python fetch_gt_features.py --feature_h5 /data4/jiaxin/exp/CLEVR/data_clevr/val_features_28x28.h5 --scene_json /data4/CLEVR_v1.0/scenes/CLEVR_val_scenes.json --image_dir /data4/CLEVR_v1.0/images/val --output_pt /data4/jiaxin/exp/CLEVR/data_clevr/val_gt_features.pt
```
**Note**: When using gt features, --edge_class must be 'dense'.


### Finally, please check whether you have prepared all these files:
- vocab.json
- train_questions.pt
- val_questions.pt
- train_features_salient_thres0.4.pt or train_gt_features.pt
- val_features_salient_thres0.4.pt or val_gt_features.pt


# Train
You need to put these five files into one folder, and train our model by following command:
```
python train.py --input_dir /the/directory/containing/five/files --save_dir /the/directory/to/save/checkpoints/and/logs
```


# Validate
- with ground truth programs
    ```
    python validate.py --ckpt /path/to/checkpoint --mode val --program gt
    ```
- with predicted programs via a seq2seq model which is trained by David from tbd-net 
    ```
    python validate.py --ckpt /path/to/checkpoint --mode val --program david
    ```

