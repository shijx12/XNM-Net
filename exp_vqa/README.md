
# pipeline to preprocess data

1. preprocess VQA2.0 train questions and obtain two output files: train_questions.pt and vocab.json.
```
python preprocess_questions.py --input_questions_json /your/path/to/v2_OpenEnded_mscoco_train2014_questions.json --input_annotations_json /your/path/to/v2_mscoco_train2014_annotations.json --output_pt /your/output/path/train_questions.pt --vocab_json /your/output/path/vocab.json --mode train
```

2. preprocess VQA2.0 val questions
```
python preprocess_questions.py --input_questions_json /your/path/to/v2_OpenEnded_mscoco_val2014_questions.json --input_annotations_json /your/path/to/v2_mscoco_val2014_annotations.json --output_pt /your/output/path/val_questions.pt --vocab_json /your/output/path/vocab.json --mode val
```


3. download grounded features from the [repo](https://github.com/peteanderson80/bottom-up-attention) of paper [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)
```
wget https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip
```

4. unzip it and preprocess features
```
python preprocess_features.py --input_tsv_folder /your/path/to/trainval_36/ --output_h5 /your/output/path/trainval_feature.h5
```

Before training, make sure your have following files in the same folder:
- vocab.json
- train_questions.pt
- trainval_feature.h5
- val_questions.pt (optional)


# Training
```
python train.py --input_dir /your/path/containing/preprocessed/files --save_dir /path/for/checkpoint --val
```
The option `--val` means to validate the trained model after each training epoch.

