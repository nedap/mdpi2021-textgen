# Word-level language modeling RNN

A few changes were made to the example word-level pytorch LM for text generation. The dataloader now requires the corpus to contain a file with low-frequency tokens that will automatically be replaced with <unk> during dataloading when main.py is run. We also changed the generator script and extended it with p-sampling and the option to use start-prompts, i.e. setting the sequence start. You can follow the new instructions in this project's main README.

### Training
For training, we introduced logging and backoff for learning rate (and minimum learning rate). 

From dir: generating-synthetic-EHRnotes
With example params:
```bash
python textgen/word_language_model/main.py --epochs 40 --nhid 650 --log-interval 2000 --data data/preprocessed/dummy/ --emsize 650 --save output/dummy/pytorch_lstm_40ep_tied/models/40ep_tied.pt --cuda --tied
```
Make sure to choose the correct path to save your model: If --model Transformer, save in output/pytorch_transformer/...

Optional args:
  * Model type --model RNN_TANH, RNN_RELU, LSTM, GRU, Transformer (default: LSTM)
    * If using Transformer, you can change number of attention heads: --nhead (default:2)
  * Size of word embeddings --emsize (default: 200)
  * Number of hidden units per layer --nhid (default:200)
  * Number of layers --nlayers (default:2)
  * Learning rate --lr (default:20)
  * Learning rate reduce factor --lr_backoff (default: 4, i.e. reduction by factor of 4 per epoch if validation loss did not go down)
  * Learning rate minimum --lr_min (default: 0.1)
  * Gradient clipping --clip (default: 0.25)
  * Epochs --epochs (default: 40)
  * Batch size --batch_size (default:20)
  * Sequence length --bptt (default: 35)
  * Dropout applied to layers --dropout (default:0.5, dropout=0 means no dropout)
  * Tie word embedding and softmax weights --tied 
  * Random seed for reproducibility --seed (default: 1111)
  * Run on GPU if cuda enabled --cuda
  * --onnx-export can be used to export final model as onnx and --dry-run allows verifying code and model


***


***



## Original: from pytorch/examples/word_language_model

This example trains a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task.
By default, the training script uses the Wikitext-2 dataset, provided.
The trained model can then be used by the generate script to generate new text.

```bash 
python main.py --cuda --epochs 6           # Train a LSTM on Wikitext-2 with CUDA
python main.py --cuda --epochs 6 --tied    # Train a tied LSTM on Wikitext-2 with CUDA
python main.py --cuda --epochs 6 --model Transformer --lr 5   
                                           # Train a Transformer model on Wikitext-2 with CUDA
python main.py --cuda --tied               # Train a tied LSTM on Wikitext-2 with CUDA for 40 epochs
python generate.py                         # Generate samples from the trained LSTM model.
python generate.py --cuda --model Transformer
                                           # Generate samples from the trained Transformer model.
```

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`)
which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --model MODEL         type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU,
                        Transformer)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --bptt BPTT           sequence length
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --cuda                use CUDA
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --onnx-export ONNX_EXPORT
                        path to export the final model in onnx format
  --nhead NHEAD         the number of heads in the encoder/decoder of the
                        transformer model
```

With these arguments, a variety of models can be tested.
As an example, the following arguments produce slower but better models:

```bash
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40           
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied    
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40        
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied 
```
