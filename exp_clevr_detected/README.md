
# XNM experiments on the CLEVR detected setting
In this experiment, we detect objects from CLEVR images using the trained detector of [tbd-net](https://github.com/davidmascharka/tbd-nets). Object-level visual features are regarded as node features, and coordinate differences between object pairs are regarded as edge features.

### Pipeline to preprocess data

1. Preprocess CLEVR train questions and obtain two output files: train_questions.pt and vocab.json
```
cd preprocess
python preprocess_questions.py --input_questions_json </path/to/CLEVR_train_questions.json> --output_pt_file <path/to/output/train_questions.pt> --output_vocab_json </path/to/output/vocab.json>
```

2. Preprocess CLEVR val questions. Note the vocab json file should be the one generated last step. We will load the vocabulary of train set and reuse it for val set.
```
python preprocess_questions.py --input_questions_json </path/to/CLEVR_val_questions.json> --output_pt_file <path/to/output/val_questions.pt> --input_vocab_json </path/to/produced/vocab.json>
```

3. Detect salient objects with the help of [tbd-net](https://github.com/davidmascharka/tbd-nets).
    1. clone their repo and enter the directory
    2. download their pretrained high-resolution checkpoint by the following command
    ```
    python utils/download_pretrained_models.py -m hres
    ```

    3. extract CLEVR image features with the following commands
    ```
    python scripts/extract_features.py \
        --input_image_dir </path/to/CLEVR/images/train> \
        --output_h5_file </path/to/train_features.h5> \
        --model_stage 2
    python scripts/extract_features.py \
        --input_image_dir </path/to/CLEVR/images/val> \
        --output_h5_file </path/to/val_features.h5> \
        --model_stage 2
    ```
    Note: `--model_stage 2` is necessary to extract high-resolution features.

    4. copy our `utils/find-salient.py` into the main directory of tbd project.

    5. enter the main directory of tbd-net, detect salient objects and coordinates using following commands:
    ```
    python find-salient.py --input_h5 </path/to/train_features.h5> --output_pt </path/to/train_features.pt>
    python find-salient.py --input_h5 </path/to/val_features.h5> --output_pt </path/to/val_features.pt>

    ```

4. Finally, please move these generated files into one folder and rename them as following:
- vocab.json
- train_questions.pt
- val_questions.pt
- train_features.pt
- val_features.pt

> NOTE: For CLEVR-CoGenT experiments, the preprocessing is totally same. You just need to change the input path at each step.


### Train
Now you can easily train our XNM model by the following command:
```
python train.py --input_dir </dir/containing/generated/files> --save_dir </dir/for/checkpoint>
```
If your file names are different from above mentioned, you must specify them using `--train_question_pt`, `--train_feature_pt`, and etc. Please refer to `train.py` for all arguments.


### Validate
- with ground truth programs
    ```
    python validate.py --input_dir </dir/containing/val/pt/files> --ckpt </path/to/checkpoint> --mode val --program gt
    ```
- with predicted programs via a seq2seq model which is trained by David from tbd-net 
    ```
    python validate.py --input_dir </dir/containing/val/pt/files> --ckpt </path/to/checkpoint> --mode val --program david
    ```


### Visualization
Startup our jupyter notebook `visualize-output.ipynb` and follow the instructions.

### Test on test set
You should preprocess test questions following Step 2, and detect objects of test images following Step 3. After `test_questions.pt` and `test_features.pt` are prepared well, you can run following command to obtain predictions:
```
python validate.py --input_dir </dir/containing/test/pt/files> --ckpt </path/to/checkpoint> --mode test --output_file </path/to/store/test/predictions>
```
During testing, David's program generator will be used since we don't have access to the ground truth programs of CLEVR test set.

