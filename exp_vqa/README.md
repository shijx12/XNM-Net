
# XNM experiments on VQA2.0

### Pipeline to preprocess data
1. Download [glove pretrained 300d word vectors](http://nlp.stanford.edu/data/glove.840B.300d.zip) and process it into a pickle file, whose content should be such a map:
```
{
    "word1": numpy.ndarray,
    "word2": numpy.ndarray,
    ...
}
```

2. Preprocess VQA2.0 train questions and obtain two output files: train_questions.pt and vocab.json.
```
python preprocess_questions.py --glove_pt </path/to/generated/glove/pickle/file> --input_questions_json </your/path/to/v2_OpenEnded_mscoco_train2014_questions.json> --input_annotations_json </your/path/to/v2_mscoco_train2014_annotations.json> --output_pt </your/output/path/train_questions.pt> --vocab_json </your/output/path/vocab.json> --mode train
```
> To combine the official train set and val set for training, just use : to join multiple json files. For example, `--input_questions_json train2014_questions.json:val2014_questions.json`

3. (optional) Preprocess VQA2.0 val questions. Note `--vocab_json` must be the one that is generated last step.
```
python preprocess_questions.py --input_questions_json </your/path/to/v2_OpenEnded_mscoco_val2014_questions.json> --input_annotations_json </your/path/to/v2_mscoco_val2014_annotations.json> --output_pt </your/output/path/val_questions.pt> --vocab_json </just/generated/vocab.json> --mode val
```

4. (optional) Preprocess VQA2.0 test questions, using the same `vocab.json`.
```
python preprocess_questions.py --input_questions_json </your/path/to/v2_OpenEnded_mscoco_test2015_questions.json> --output_pt </your/output/path/test_questions.pt> --vocab_json </just/generated/vocab.json> --mode test
```

5. Download grounded features from the [repo](https://github.com/peteanderson80/bottom-up-attention) of paper [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)
```
wget https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip
```

6. Unzip it and preprocess features
```
python preprocess_features.py --input_tsv_folder /your/path/to/trainval_36/ --output_h5 /your/output/path/trainval_feature.h5
```

7. (optional) Download and preprocess test2015 features following step 5 and 6

Before training, make sure your have following files in the same folder:
- vocab.json
- train_questions.pt
- trainval_feature.h5
- val_questions.pt (optional)
- test_questions.pt (optional)
- test_feature.h5 (optional)


### Train
```
python train.py --input_dir </your/path/containing/preprocessed/files> --save_dir </path/for/checkpoint> --val
```
The option `--val` means to validate the trained model after each training epoch.

### Validate
```
python validate.py --input_dir </path/containing/preprocessed/files> --ckpt </path/to/checkpoint> --mode val
```

### Test
```
python validate.py --input_dir </path/containing/preprocessed/files> --ckpt </path/to/checkpoint> --mode test --output_file </file/to/store/predictions> --test_question_json <your/path/to/v2_OpenEnded_mscoco_test2015_questions.json>
```

### Visualization
Startup `visualize.ipynb` and follow the instructions.
