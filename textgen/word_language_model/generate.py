# sampling from LM using just temperature or with nucleus/top-p filtering

import argparse
from tqdm import tqdm
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from textgen.word_language_model import data

parser = argparse.ArgumentParser(description='Pytorch Language Model Generator')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./output/52miow_sr/lstm/models/50ep_tied_2ndrun.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--input_file',
                    help='path to file with input prompts')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--top_p', type = float, default=None,
                    help='If used, will sample with nucleus sampling and p=0.95')
parser.add_argument('--max_length', type = int, default=500,
                    help='Generator does not produce sequences longer than 500 tokens.')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")
    
if args.top_p and (args.top_p > 1 or args.top_p < 0):
    parser.error("--p has to be between 0 and 1")

# get model, print (trainable) parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

                    
# helper function for nucleus sampling
def top_p_filtering(logits, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1 # batch size 1 for now - could be updated for more but the code would be less clear

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        softmax = nn.Softmax(dim=-1)
        cumulative_probs = torch.cumsum(softmax(sorted_logits), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        
    return logits


# import prompts from file
def get_prompts(input_path):
    # get list of input texts, encode them
    with open(input_path, 'r') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip() != '']
        prompts = [p.split(' ') for p in prompts]

    print(prompts[:5])
    return prompts


# GENERATOR 
    # can sample with temperature and optionally filter using p 
    # can use prompts from input file or print one result starting from <eos>
def generate(corpus, model, hidden, args):
    print('Generating with temperature = {}'.format(args.temperature))
    if args.top_p:
        print('Filtering with p = {} (nucleus/top-p sampling)'.format(args.top_p))
    
    unk_ind = corpus.dictionary.word2idx['<unk>']
    
    # get prompt list or <eos> starter
    if args.input_file:
        print('Reading prompts from input file')
        prompts = get_prompts(Path(args.input_file))
    else:
        print('Using <eos> as prompt')
        prompts = [['<eos>']]
        result_sequence = []
    
    with open(args.outf, 'w') as outf:
        
        # for each prompt...
        for p in tqdm(prompts):
            count=0
            restart = False
            
            # clear memory
            with torch.no_grad(): 
                
                # deal with prompt
                for word in p:
                    # write word to file/commandline & update wordcount
                    if args.input_file:
                        outf.write(word+' ')
                    else:
                        result_sequence.append(word)
                    count+=1
                    # update model
                    word_ind = corpus.dictionary.word2idx.get(word, unk_ind)
                    input = torch.ones(1,1).mul(word_ind).long().to(device)
                    output, hidden = model(input, hidden)
                
                # continue to sequence generation from prompt & write to file
                while (not restart) and (count<args.max_length):
                    
                    if args.top_p: # with p
                        # continue to sequence generation from prompt & write to file
                        logits = output.data.view(-1).div(args.temperature)
                        filtered_logits = top_p_filtering(logits, top_p=args.top_p)
                        # Sample from the filtered distribution
                        word_weights = F.softmax(filtered_logits, dim=-1)
                    
                    else: # with temperature
                        word_weights = output.squeeze().div(args.temperature).exp().cpu()
                                            
                    # update model
                    next_token_idx = torch.multinomial(word_weights, 1)[0]
                    input.fill_(next_token_idx)
                    output, hidden = model(input, hidden)
                                        
                    
                    # get word from picked idx, update wordcount
                    word = corpus.dictionary.idx2word[next_token_idx]
                    count += 1
                    # write word or newline if eos or max words reached
                    if word != '<eos>':
                        if args.input_file:
                            outf.write(word + ' ')
                        else:
                            result_sequence.append(word)
                        restart = False
                    else:
                        if args.input_file:
                            outf.write('\n')
                        else:
                            print(' '.join(result_sequence))
                        restart = True


################################################################################

# GET MODEL:
with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
print('Number of model parameters: {}'.format(count_parameters(model)))
model.eval()

# GET DICTIONARY
corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

# ADJUST MODEL SETTINGS
is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)

# GENERATE AND WRITE TO FILE:
generate(corpus, model, hidden, args)
