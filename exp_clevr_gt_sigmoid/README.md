
# XNM experiments on the CLEVR gt setting - concat features, then sigmoid
In this setting, we concat attribute value features as node features, and use sigmoid as the attention activation function. Edge attention is similar.

### Pipeline to preprocess data

1. Preprocess CLEVR train questions and obtain two output files: train_questions.pt and vocab.json
```
cd preprocess
python preprocess_questions.py --input_questions_json </path/to/CLEVR_train_questions.json> --output_pt_file </path/to/output/train_questions.pt> --output_vocab_json <path/to/output/vocab.json>
```

2. Preprocess CLEVR val questions. Note the vocab json file should be the one generated last step. We will load the vocabulary of train set and reuse it for val set.
```
python preprocess_questions.py --input_questions_json </path/to/CLEVR_val_questions.json> --output_pt_file </path/to/output/val_questions.pt --input_vocab_json </path/to/produced/vocab.json>

```

3. Preprocess CLEVR train scenes. The vocab json file should still be the last one. We will update it with attribute values and relationship categories.
```
python preprocess_scene.py --input-scene </path/to/CLEVR_train_scenes.json> --vocab-json </path/to/produced/vocab.json> --output-scene </path/to/output/train_scenes.pt>
```

4. Preprocess CLEVR val scenes.
```
python preprocess_scene.py --input-scene </path/to/CLEVR_val_scenes.json> --vocab-json </path/to/produced/vocab.json> --output-scene </path/to/output/val_scenes.pt>
```

5. Finally, please move these generated files into one folder and make sure they are named as following:
- vocab.json
- train_questions.pt
- val_questions.pt
- train_scenes.pt
- val_scenes.pt

> NOTE: For CLEVR-CoGenT experiments, the preprocessing is totally same. You just need to change the input path at each step.


### Train
To reproduce our experiments, you only need to run the following command:
```
python train.py --input_dir </dir/containing/generated/files> --save_dir </dir/for/checkpoint>
```
The validation accuracy will be evaluated and printed after each training epoch.
If your file names are different from above mentioned, you must specify them using `--train_question_pt`, `--train_feature_pt`, and etc. Please refer to `train.py` for all arguments.


### Validate
- with ground truth programs
    ```
    python validate.py --input_dir </dir/containing/generated/files> --ckpt </path/to/checkpoint> --program gt
    ```
- with predicted programs via a seq2seq model which is trained by David from tbd-net 
    ```
    python validate.py --input_dir </dir/containing/generated/files> --ckpt </path/to/checkpoint> --program david
    ```


### Visulization
Refer to our jupyter notebook `../exp_clevr_gt_softmax/visualize-output.ipynb`.


### Test on test set
Since scene graph annotations of CLEVR test set are not available, we can't conduct this experiment on test set. To predict answers for test set, you can move to our detected setting, whose implementation is in `../exp_clevr_detected`.


