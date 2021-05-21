# Generating Synthetic Training Data for Supervised De-Identification of Electronic Health Records

Python code to create synthetic text data with privacy annotations (using a LSTM or GPT-2 based language model), including scripts for automatic pre-processing, post-processing and evaluation.

This repository shares the code developed in the following paper:

> Libbi, C.A.; Trienes, J.; Trieschnigg, D.; Seifert, C. Generating Synthetic Training Data for Supervised De-Identification of Electronic Health Records. Future Internet 2021, 13, 136. https://doi.org/10.3390/fi13050136

The PDF version of the paper can be downloaded using this link: https://www.mdpi.com/1999-5903/13/5/136/pdf

## Summary

> A major hurdle in the development of natural language processing (NLP) methods for Electronic Health Records (EHRs) is the lack of large, annotated datasets. Privacy concerns prevent the distribution of EHRs, and the annotation of data is known to be costly and cumbersome. Synthetic data presents a promising solution to the privacy concern, if synthetic data has comparable utility to real data and if it preserves the privacy of patients. However, the generation of synthetic text alone is not useful for NLP because of the lack of annotations. In this work, we propose the use of neural language models (LSTM and GPT-2) for generating artificial EHR text jointly with annotations for named-entity recognition. Our experiments show that artificial documents can be used to train a supervised named-entity recognition model for de-identification, which outperforms a state-of-the-art rule-based baseline. Moreover, we show that combining real data with synthetic data improves the recall of the method, without manual annotation effort. We conduct a user study to gain insights on the privacy of artificial text. We highlight privacy risks associated with language models to inform future research on privacy-preserving automated text generation and metrics for evaluating privacy-preservation during text generation.

## Requirements

Install packages:

```sh
conda env update -f environment.yml
```

Download [`deidentify`](https://github.com/nedap/deidentify) model:

```sh
python -m deidentify.util.download_model model_bilstmcrf_ons_fast-v0.1.0
```
Have a look [here](https://github.com/nedap/deidentify/blob/master/docs/04_hsdm2020_surrogate_generation.md#setup) to make sure you have the required locales setup.


Download Dutch spaCy model:

```sh
python3 -m spacy download 'nl_core_news_lg'
```

## Prepare dataset

The data used in this study is privacy sensitive and was not published. Instead, a dummy raw data file file is included: data/external/dummy_data.csv.
When you replace this file with your custom dataset, make sure the csv format is the same.

The text in the dummy dataset is taken from ‘dutch_book_reviews_train_14k.csv’ from the Kaggle dataset by [kashnitsky](https://www.kaggle.com/kashnitsky/exploring-transfer-learning-for-nlp). You can run the whole pipeline with this data or prepare your own dataset in a similar csv file, executing:

```bash
python -m textgen.data.make_dataset --dummy data/external/dummy_data.csv
```

### Preprocessing and evaluation splits

Takes data from data/interim and creates four text files in data/preprocessed:
* all.txt
* train.txt
* valid.txt
* test.txt

These will be used as input for the textgen model and contain the in-text annotations from the previous step.

```bash
python -m textgen.data.preprocessing  --in_dir data/interim/annotated_dummy.jsonl --out_dir data/preprocessed/dummy
```
You can restrict the amount of data you preprocess (i.e. decide how many words your dataset should have) with the argument: `--n_words <desired size of dataset>`


## Train the model and generate synthetic EHR notes

### Get list of low-frequency tokens
_This step is only necessary if you plan to train the LSTM (word-language-model), which uses the output when loading data to reduce the vocabulary size._

Takes training set (train.txt) and returns a file with all tokens occurring less than n times. You can specify n, or leave the default n=3.

* Output: data/preprocessed/CORPUSNAME/lowfrequency_tokens.txt

```bash
python -m textgen.data.get_lowfreqs dummy --min_wordfreq 3
```

### Training and generating text with the LSTM
We adapted the pytorch word_language_model in [this repository](https://github.com/pytorch/examples/tree/master/word_language_model). Changes are inspired by [Melamud & Shivade (2019)](https://www.aclweb.org/anthology/W19-1905.pdf).

**Training**

1. If you want to run the pytorch LSTM (default), then the output filepath will have MODELNAME = pytorch_lstm
2. If you want to run the pytorch Transformer or a different model (remember to specify), make sure to create a directory with the appropriate sub-directories in output/

Training produces:
* output/MODELNAME/models/MODELSPECS.pt
* output/MODELNAME/models/MODELSPECS.pt.log

Note: Ideally, the name of the specific trained model (MODELSPECS) should include some hints to training parameters that distinguish different training runs with the same model.

Example with dummy dataset:
```bash
python textgen/word_language_model/main.py --data data/preprocessed/dummy --save output/dummy/pytorch_lstm_40ep_tied/models/40ep_tied.pt --epochs 40 --tied --nhid 650 --emsize 650 --cuda
```
**Text Generation**
You can either use temperature or p-sampling with the LSTM.

The generator takes these arguments:
```
--data (location of the data corpus)
--checkpoint (location of the trained model you want to use)
--outf (location and name of the output file)
--words (number of words to generate)
--max_length (maximum length of the sequences you want to generate)
--input_file (a file to generate input prompts, i.e. the starts of the sequences you will generate)
--temperature (for temperature sampling, a higher temperature will increase diversity. t=1 is in the middle.)
--top_p (for nucleus/p-sampling, p=0.95 is a good value)
--cuda (will run on CUDA if you have it)
```

Generation produces:
* output/MODELNAME/synthetic_data/MODELNAME.txt

If you trained the dummy LSTM, you can generate synthetic data like this:

Without input file & with temperature sampling:
```bash
python textgen/word_language_model/generate.py  --data data/preprocessed/dummy \
                                                --checkpoint output/dummy/pytorch_lstm_40ep_tied/models/40ep_tied.pt \
                                                --outf output/dummy/pytorch_lstm_40ep_tied/synthetic_data/synth500w.txt \
                                                --words 500 \
                                                --max_length 100\
                                                --temperature 1\
                                                --cuda
```

With input file:

1. Create an input file. You can do this by sampling sub-sequences from the testset in your corpus, e.g.:
```bash
python textgen/data/get_inputseqs.py --corpus dummy --sample_len 3 --sample_from_start
```
This will produce a file: 'data/preprocessed/dummy/textgen_inputs_len_3_fromstart.txt'

2. Then use this to generate synthetic data, let's use p-sampling this time:
```bash
python textgen/word_language_model/generate.py  --data data/preprocessed/dummy \
                                                --checkpoint output/dummy/pytorch_lstm_40ep_tied/models/40ep_tied.pt \
                                                --outf output/dummy/pytorch_lstm_40ep_tied/synthetic_data/synth500w_p095_with_input.txt \
                                                --input_file data/preprocessed/dummy/textgen_inputs_len_3_fromstart.txt
                                                --words 500 \
                                                --max_length 100\
                                                --top_p 0.95\
                                                --cuda
```

More info about the model can be found in [instructions for the textgen model](https://github.com/nedap/mdpi2021-textgen/blob/main/textgen/word_language_model/README.md).

### Finetuning GPT2 and generating text with it
This implementation uses the huggingface transformers library and fastai. We fine-tune OpenAI's English pre-trained (small) gpt2 model (and tokenizer), closely following the implementation by Pierre Guillou (find it [here](https://medium.com/@pierre_guillou/faster-than-training-from-scratch-fine-tuning-the-english-gpt-2-in-any-language-with-hugging-f2ec05c98787)).

Click [here](https://github.com/nedap/mdpi2021-textgen/blob/main/textgen/word_language_model/README.md) to find the full instructions for this part and other options for generating text.

The steps in short:

1. Set up gpt2 environment
```bash
conda env update -f textgen/gpt2/environment.yml
conda activate textgen-gpt2
```
2. Train Tokenizer
```bash
python -m textgen.gpt2.train_tokenizer --corpus dummy
```
3. Finetune
```bash
python -m textgen.gpt2.finetune --corpus dummy --save_model YOUR-MODELNAME
```
4. Calculate test perplexity
```bash
python -m textgen.gpt2.get_test_perplexity --corpus dummy --modelpath output/dummy/gpt2_nl/model/YOUR-MODELNAME
```
5. Generate synthetic text
```bash
python -m textgen.gpt2.generate --input_text 'the first words ...' --corpus dummy --checkpoint \<CHECKPOINT-NAME\> --beam 8 --max_length 100
```
or with the input file:
```bash
python -m textgen.gpt2.generate --input_file data/preprocessed/dummy/textgen_inputs_len_3_fromstart.txt --corpus dummy --checkpoint \<CHECKPOINT-NAME\> --beam 8 --max_length 100
```
you can also use p/nucleus-sampling instead of beam search:
```bash
python -m textgen.gpt2.generate --input_file data/preprocessed/dummy/textgen_inputs_len_3_fromstart.txt --corpus dummy --checkpoint \<CHECKPOINT-NAME\> --p 0.95 --max_length 100
```


## Postprocessing: checking well-formed annotations
Before using the model-generated annotations in the synthetic data, we need to make sure that the model generated well-formed annotations. This means that there should be no case where e.g. <DateSTART> isn't followed by a <DateEND> before any other annotation token. We also want to avoid end-tokens that are not preceeded by a start-token, empty annotations or start-end-pairs of different types.

```bash
conda activate textgen
python -m textgen.evaluation.postprocessing.annotation_check output/dummy/pytorch_lstm_40ep_tied/synthetic_data/synth500w.txt
```

The inputfile is the synthetic data, which is in the output directory.

The output is saved automatically in the same directory as the input data:
* .phi_cleaned.txt contains the new version of synthetic text without badly formed annotations
* .phi_cleaned.log contains a count of bad vs. good annotations per note and overall.


## Intrinsic evaluation
* The test perplexity of the LSTM is saved in the model log (model.pt.log)
* You can calculate test perplexity of the GPT2 model like this:

```bash
conda activate textgen-gpt2
python textgen/gpt2/get_test_perplexity.py --corpus dummy --modelpath <PATH-TO-CHECKPOINT> --cuda
```

* The previous step evaluated how often the model produces well-formed annotations (although this doesn't say anything about whether the annotations are actually appropriate)

_(This repository also contains a file textgen/word_language_model/reconstruct_sequency.py which you can use to check whether your LSTM has overfitted and will be able to re-create certain input sequences from the training set if you give it the initial tokens. This script was not used in this study and it was not fully tested with the final version of the model, but it was kept in the repository in case it can be useful for future work.)_

### PHI tag distribution
To compare the distribution of PHI tags in the original data vs. the synthetic data.
The script automatically takes the original input data, but you can specify which model's synthetic data you want to compare.
From this directory, all files ending in .phi_cleaned.txt will be included.
```sh
python -m textgen.evaluation.phi_tag_distribution --datadir 'data/preprocessed/dummy' --synthdir 'output/dummy/pytorch_lstm_40ep_tied/synthetic_data'
```
The output can be found in `output/MODELNAME/evaluation` with the ending `_phi-eval.csv`


## Privacy evaluation
To check whether data has leaked from the real data into the synthetic data, we did the following:

1. Use ROUGE-N (measure for n-gram overlap) recall to rank synthetic data according to which text has the highest similarity to a text in the training data + retrieve that real text.

2. For each synthetic text, use BM25 (tf-idf based document similarity measure usually used in information retrieval) scores to retrieve another option as potential "most similar" text from the training data.

3. You can compile these real-fake document pairs into a table and use the questionnaire (we provided a Jupyter notebook that will run the questionnaire using Voila and the table as input) to ask users to check each pair for potential privacy leaks.


### ROUGE-N
This step creates a csv file where each row contains a synthetic document with the real document that was most similar to it (according to ROUGE-N recall) and the ROUGE-N scores of this match. This means every synthetic note is compared with every real note, which takes a while, depending on the size of your datasets. The script we used parallelizes this process, you may have to adjust the parameters depending on how many cores you can work with.

```bash
python textgen/evaluation/rouge_similarity_ranking.py --realpath data/preprocessed/dummy/train.txt \
                                                      --synthpath output/dummy/pytorch_lstm_40ep_tied/synthetic_data/synth500w.txt \
                                                      --outdir output/dummy/pytorch_lstm_40ep_tied/evaluation \
                                                      --n_jobs 1 \
                                                      --batch_size 32
```
* `--n_jobs` is the number of synthetic documents processed in parallel. Set this higher if you can and have a lot of data.
* `--batch_size` is the number of synthetic documents to process before saving results


### BM25 (using an Elasticsearch index)
Taking an EHR note as query, we can search the whole corpus of real notes (used in model training) for the most similar ones by building a search index with Elasticsearch. In this case, we use BM25 to calculate document similarity.

**Set-up environment (with Docker)**

Install Docker like [this](https://gist.github.com/rstacruz/297fc799f094f55d062b982f7dac9e41#getting-docker). Set up and start Docker with Elasticsearch and Kibana like this:

```sh
docker-compose up -d
```

**Build search index**

```sh
python textgen/elasticsearch/make_search_index.py --data '/output/dummy/pytorch_lstm_40ep_tied/synthetic_data/synth.txt'
```
With `--indexname` you can specify the name of your index, the default is 'ehr-data-index'
(This may take a while if you are indexing many files.)

**Run set of notes as queries and retrieve similar notes**

To navigate and query your index via a visual interface, go to Kibana in your browser: http://localhost:5601
* Navigate to DevTools in the sidebar
* Use the console to run queries, e.g.:

```curl
GET /_search
{
  "query": {
    "match": {
      "content": "<your query>"
  }
}
```

To automatically run all your queries and save the results, run the script `run_match_queries.py`, which takes a few arguments:
* --queries takes the location of the file that contains the EHR notes that you want to run as queries. This could be a manually created sample or an existing txt file in the corpus.
* --outf is the path to the output file you want to create. Put this in output/... and give the file a name that relates to your --queries input file.
* --n_hits takes an integer >= 1 and dictates how many hits per query will be written to results. Default is 1, so only the most similar note found is saved, but you could e.g. save the top 3 best etc.

```sh
python textgen/elasticsearch/run_match_queries.py \
  --queries 'data/preprocessed/dummy/train.txt' \
  --outf '/output/dummy/pytorch_lstm_40ep_tied/evaluation/query_results_dummy.txt' \
  --n_hits 3
```

If you want to test that everything runs correctly, you can run some debug queries with curl like [this](https://github.com/nedap/mdpi2021-textgen/blob/main/textgen/elasticsearch/README.md).


### User study (questionnaire environment)
The questionnaire is an interactive Jupyter notebook served as a standalone application via [`voila`](https://github.com/voila-dashboards/voila). You can use below command to start Jupyter locally. Then, open http://localhost:8895 in your browser. When being prompted for a password, use `privacy-evaluation`.

```sh
cd questionnaire && docker-compose up
```
Note that we did not include code to compile the table of document-pairs that you need as input for the questionnaire. You can decide how you would like to combine, filter and batch your data. The csv should have the format required as input for the notebook in 'questionnaire/notebooks/Questionnaire.ipynb', but you can edit the notebook to read a table with different headers/columns etc.

### Interpretation and user study

_The similarity scores or 'matches' between real and synthetic notes do not intrinsically have an interesting meaning. This is why we set up a user study to interpret this similarity in a few samples. Depending on the question, this can help understanding whether synthetic notes are (1) copies or near-copies of original text or (2) similar in the sense that personal information has been leaked into synthetic data, causing privacy concerns. On the other hand, a conclusion might be that (3) similarity merely reflects the generation of popular phrases or word patterns in the original data, which makes synthetic data look very realistic or appropriate for the domain/style._

## Extrinsic evaluation with downstream NLP-task: deidentify

We need to adjust the format of text and annotations for `deidentify`, which takes each note as .txt file with a separate .ann file that contains annotations in stand-off format. You will need 'phi_cleaned' synthetic data to do this step!

You need to specify input_file OR input_dir, not both (depends on whether you want to use all generated text or just specific files):
 * --input_file takes a single synth file
 * --input_dir takes all .phi_cleaned.txt files in the given folder

Decide on max. how many batches you want, e.g. 10: `--max_batch 10`

Optionally, you can activate a filter that will remove all notes outside of the length-range of 50-1000 tokens. This is to closely follow the instructions for training the deidentify model as specified in the research paper. To activate the filter, use --lengthfilter

```sh
python -m textgen.evaluation.postprocessing.prepdata4deidentify_batched --input_dir output/dummy/pytorch_lstm_40ep_tied/synthetic_data --output_dir output/dummy/pytorch_lstm_40ep_tied/deidentify_data --max_batch 3
```

To (make a corpus,) train and test the deidentify model, follow further instructions here: https://github.com/nedap/deidentify

The wiki also includes a step-by-step guide to reproduce how the model was trained and tested in this project [here](https://github.com/nedap/mdpi2021-textgen/blob/main/wiki_docs/SupplementaryDeidentifyGuide.md).


## Citation

If you use the resources presented in this repository, please cite:

```bibtex
@article{Libbi:2021:GST,
  author={Libbi, Claudia Alessandra and Trienes, Jan and Trieschnigg, Dolf and Seifert, Christin},
  title = {Generating Synthetic Training Data for Supervised De-Identification of Electronic Health Records},
  journal = {Future Internet},
  volume = {13},
  year = {2021},
  number = {5},
  article-number = {136},
  url = {https://www.mdpi.com/1999-5903/13/5/136},
  issn = {1999-5903},
  doi = {10.3390/fi13050136}
}
```

## Acknowledgements

_We thank Michel Glintmeijer for the valuable discussions!_

## Contact

For questions about this research, you are welcome to send an email to Claudia Libbi at: alelib29@gmail.com
