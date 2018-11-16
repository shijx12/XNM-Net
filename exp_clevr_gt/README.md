
## pipeline to preprocess data

1. Preprocess CLEVR train questions and create vocab
```
cd preprocess
python3.6 preprocess_questions.py --input_questions_json /your/path/to/CLEVR_train_questions.json --output_pt_file /your/output/path/train_questions.pt --output_vocab_json /your/output/path/vocab.json

```

2. Preprocess CLEVR val questions with generated vocab
```
python3.6 preprocess_questions.py --input_questions_json /your/path/to/CLEVR_val_questions.json --output_pt_file /your/output/path/val_questions.pt --input_vocab_json /your/output/path/vocab.json

```

3. Preprocess CLEVR train scenes. This will reload vocab.json and update it with edge labels.
```
python3.6 preprocess_scene.py --input-scene /your/path/to/CLEVR_train_scenes.json --vocab-json /your/output/path/vocab.json --output-scene /your/output/path/train_scenes.pt
```

4. Preprocess CLEVR val scenes.
```
python3.6 preprocess_scene.py --input-scene /your/path/to/CLEVR_val_scenes.json --vocab-json /your/output/path/vocab.json --output-scene /your/output/path/val_scenes.pt
```

Note that all mentioned output path should be the same one, containing all of your processed files.



## Training
To reproduce our experiments, you only need to run the following command:
```
python train.py --input_dir /your/path/containing/processed/files --save_dir /path/for/checkpoint
```
The validation accuracy will be evaluated and printed after each training epoch.


## Visulization
Startup our jupyter notebook `visualize-output.ipynb` and follow the instructions.
