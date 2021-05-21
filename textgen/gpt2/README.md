## Finetune GPT-2 for Dutch

### Set-up gpt2 environment

For now, we have a separate environment for GPT-2. We might merge this later with the base environment.

```sh
conda env update -f textgen/gpt2/environment.yml
conda activate textgen-gpt2
```

### 1. Making the Tokenizer more Dutch & adjusting embedding matrix of model

```bash
python -m textgen.gpt2.train_tokenizer --corpus dummy
```
* --corpus: specify corpus name, default is dummy
* --run_test True: (default False) returns quick encoding test on example sentence
* --cuda: use CUDA (if available)

You can now load the model like this:

```python
from transformers import GPT2HeadModel
model_nl = GPT2LMHeadModel.from_pretrained('output/dummy/gpt2_nl/model')
```

### 2. Making the model even more Dutch (Fine-tuning on Dutch corpus)

To finetune model:

```bash
python -m textgen.gpt2.finetune --corpus dummy
```
* --corpus: specify corpus with text to train/validate model. Will use all.txt file from corpus, the test/validate split is done in the script!
* --cuda: use CUDA (if available)

If you have previously fitted the model, the checkpoints should be saved in your model directory as .pth files. You can also load a checkpoint and validate the model. This prints loss, accuracy and perplexity:

```bash
python -m textgen.gpt2.finetune --corpus dummy --load_finetuned finetuned
```
* --corpus: specify corpus used to finetune model, default is dummy
* --load_finetuned: use the checkpoint name without .pth.
* --cuda: use CUDA (if available)

### 3. Generate synthetic text!

You can use beam-search, top-k sampling or top-p (nucleus) sampling. You have to specify at least one of these as argument, but you can also do all three and the results of each will be printed to the commandline.

```bash
python -m textgen.gpt2.generate --input_text 'the first words ...' --corpus dummy --checkpoint \<CHECKPOINT-NAME\> --beam 8 --max_length 100
```
* --input_text: The conditional input for generation, so basically the first words of the text you will generate.
* --input_file: You can read conditional input from a file, each line is treated as new input, for which synthetic text is generated. Can only be used in combination with beam search (--beam) or p-sampling (--p). The synthetic text is automatically saved to a file in output/<corpus>/gpt2_dutch/synthetic_text. You can't use this option together with --input_text.
* --corpus: specify corpus used to finetune model, default is dummy.
* --checkpoint: use the checkpoint name without .pth (checkpoints are saved in the model folder).
* --k: Sampling with k, should be a positive integer value, e.g. 10.
* --p: Sampling with p (nucleus sampling), should be a float between 0 and 1, e.g. 0.95.
* --beam: Beam search, give number of beams. Should be positive integer value, e.g. 8.
* --max_length: Maximum number of words generated. Should be positive integer value, default is 100.
* --cuda: use CUDA (if available)
* --verbose: you can print text as it is being generated
