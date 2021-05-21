# Supplementary step by step guide for extrinsic evaluation in this project

These are the steps to train and test a Deidentify model with the dummy synthetic data. 

You can find the original instructions in the [deidentify developer readme](https://github.com/nedap/deidentify#running-experiments-and-training-models).


***

## Set-up

1. Clone deidentify repo (example uses ssh key), go into folder:  
```bash 
git clone git@github.com:nedap/deidentify.git
cd deidentify 
```

2. Set up environment with environment.yml 
```bash 
conda env create -f environment.yml 
conda activate deidentify && export PYTHONPATH="${PYTHONPATH}:$(pwd)" 
```

## Prepare corpus

3. Use synthetic data from project repo and split:

For this example, will be using data from this project in 
/generating-synthetic-EHRnotes/output/dummy/pytorch_lstm_40ep_tied/deidentify_data, which should contain .ann and .txt files we prepared for deidentify (make sure you have created these with the previous steps in the pipeline). 

We need to create train/dev/test split for deidentify. Let’s call the new corpus dummysynth:
(Make sure you use the correct path from your root folder ~/\<the path\>)

```bash 
cd ~/deidentify
python deidentify/dataset/brat2corpus.py DUMMYSYNTH ~/generating-synthetic-EHRnotes/output/dummy/pytorch_lstm_40ep_tied/deidentify_data
```
You can now find your corpus in data/corpus/DUMMYSYNTH.

## Train model on synth data

4. To train, we will use the bilstmcrf, which has a range of arguments.  To see options use --help. DEMORUN is the (arbitrary) run-id that the output will be saved under. E.g.:

```bash
python deidentify/methods/bilstmcrf/run_bilstmcrf.py --help
python deidentify/methods/bilstmcrf/run_bilstmcrf.py DUMMYSYNTH DEMORUN --pooled_contextual_embeddings --train_with_dev
```

After training (you can also interrupt with ctrl+c and the best model is saved), you should have a file:
`output/predictions/DUMMYSYNTH/bilstmcrf_DEMORUN/flair/final-model.pt`

**OPTIONAL**: It is possible to train models on fractions of the corpus, e.g. 25%, to compare how f1-scores relate to training set size. Let's call the example run 'FRACRUN':
```bash
python deidentify/methods/bilstmcrf/run_bilstmcrf_training_sample.py DUMMYSYNTH FRACRUN --train_sample_frac 0.25 --corpus_lang nl --save_final_model
```
_As the training ends with an evaluation of the model on the test set in the corpus, you may not want to save the model (e.g. for space reasons) and can leave out the last argument --save\_final\_model. However, in our case we want to evaluate model performance on a test set of a different corpus, so this happens separately in the next step and we need to save the model._

### Continuing training on a pre-trained model
```bash 
python deidentify/methods/bilstmcrf/run_bilstmcrf.py TRAININGDATA FINETUNE-RUN --continue_training PATH-TO-PRETRAINED-MODEL --learning_rate FLOAT --pooled_contextual_embeddings --train_with_dev
```
The pre-trained model is any final-model.pt file you want to use. If you don't set the initial learning rate, it will start with 0.1. Consider setting it to a much smaller float value to avoid 'overwriting' (and forgetting) the pre-trained model's weights. 

## Evaluate model on gold annotated corpus

5. First we need to generate predictions for the real test data, but using the new model. To do this, make sure your real dataset (with 'gold annotations') exists as deidentify corpus in data/corpus/ (use brat2corpus.py as in step 3). Let's call it 'REALDATA'

```bash
python deidentify/methods/bilstmcrf/run_bilstmcrf.py REALDATA DEMORUN-EVAL --model_file output/predictions/DUMMYSYNTH/bilstmcrf_DEMORUN/flair/final-model.pt
```

This will create a new folder `output/predictions/REALDATA`, which contains the predictions.


6. Finally, we want to know if those predictions are actually good! To compare them to the ground-truth annotations in REALDATA:

```bash
python deidentify/evaluation/evaluate_run.py nl data/corpus/REALDATA/test/ data/corpus/REALDATA/test/ output/predictions/REALDATA/DEMORUN-EVAL/test
```

If you created multiple predictions with the same corpus, you can evaluate all of them by running:

```bash
python deidentify/evaluation/evaluate_corpus.py REALDATA 
```

 
