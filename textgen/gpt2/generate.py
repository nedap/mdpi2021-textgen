# GENERATOR
# https://huggingface.co/blog/how-to-generate
# https://huggingface.co/transformers/main_classes/model.html?highlight=generate

import argparse
import re
import sys
from pathlib import Path
from tqdm import tqdm
from random import randrange

import torch
from fastai.text.all import CrossEntropyLossFlat, Learner, Perplexity, accuracy
from transformers import GPT2LMHeadModel

from .finetune import DropOutput, splitter
from .train_tokenizer import get_text, load_gpt2tokenizer

from tqdm import tqdm

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--input_text',
                    help='Input text for conditional generation')
parser.add_argument('--input_file',
                    help='Takes filepath to input file and saves all to output/<corpus>/gpt2/synthetic_data')
parser.add_argument('--corpus',
                    default='dummy',
                    help='name of corpus used to finetune model')
parser.add_argument('--checkpoint',
                    default='GPT2_nl_5epoch_lre-4',
                    help='name of last saved learner object (ends in .pth)')
parser.add_argument('--k',
                    type=int,
                    help='Sampling with k, should be int value.')
parser.add_argument('--p',
                    type=float,
                    help='Sampling with p (nucleus sampling), should be float between 0 and 1.')
parser.add_argument('--beam',
                    type=int,
                    help='Using beam search, arg is number of beams')
parser.add_argument('--max_length',
                    default=100,
                    help='Maximum length of text to be generated.')
parser.add_argument('--verbose',
                    action='store_true',
                    help='Print text as it is being generated.')


################################################################################


def generator_p(model_nl, input_ids, p, max_length, eos_token_id):
    p_output = model_nl.generate(
        input_ids,
        # Explicitly set pad to eos. This is the default anyways, but setting it silences message
        pad_token_id=eos_token_id,
        eos_token_id=eos_token_id,
        min_length=randrange(20,200),
        max_length=max_length,
        top_p=p,
        top_k=0,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    return p_output


def generator_k(model_nl, input_ids, k, max_length):
    k_output = model_nl.generate(
        input_ids,
        do_sample=True,
        max_length=max_length,
        top_k=k
    )
    return k_output


def generator_beam(model_nl, input_ids, num_beams, max_length, eos_token_id):
    beam_output = model_nl.generate(
        input_ids,
        # Explicitly set pad to eos. This is the default anyways, but setting it silences message
        pad_token_id=eos_token_id,
        eos_token_id=eos_token_id,
        min_length=randrange(50,200),
        max_length=max_length,
        num_beams=num_beams,
        no_repeat_ngram_size=3,
        num_return_sequences=num_beams,  # 5 looks pretty good
        early_stopping=True
    )
    return beam_output


def preproc(text):
    # add later, probably with spacy: remove anything after last fullstop
    text = text.replace('<PAR>', '\n')
    text = text.replace('<|endoftext|>', '')
    preprocessed = re.sub('<[A-Za-z_]*START>[^a-zA-Z0-9_]*<[A-Za-z_]*END>', '<unk>', text)
    phinum = len(re.findall(r'<[A-Za-z\_]*(END)>', preprocessed))
    return preprocessed, phinum


def pick_max_phi(output_beam, tokenizer_nl):
    chosen_one = ''
    maxphinum = 0
    maxphi_textlen = 0
    for i, beam_output in enumerate(output_beam):
        # decode output, remove empty phi and incomplete endsentence, then return processed text and number of phi
        text, phinum = preproc(tokenizer_nl.decode(beam_output, skip_special_tokens=False))
        # if phinum is higher than current chosen one, replace it & update phinum
        if phinum > maxphinum:
            chosen_one = text
            maxphinum = phinum
            maxphi_textlen = len(text)
        # if phinum is tied, take longer one
        elif maxphinum == 0:
            if len(text) > maxphi_textlen:
                chosen_one = text
                maxphinum = phinum
                maxphi_textlen = len(text)
    return chosen_one


def generate_and_write_beam(args_beam, input_file, args_corpus, tokenizer_nl, model_nl, max_length):

    # get list of input texts, encode them
    input_path = Path(input_file)
    with open(input_path, 'r') as f:
        prompts = ['<|endoftext|> '+line.strip() for line in f.readlines() if line.strip() != '']
        print(prompts[:5])
    id_prompts = [tokenizer_nl.encode(p_text, return_tensors='pt').to(device) for p_text in prompts]

    print('Reading from input file.')
    print('BEAM SEARCH: \nSaving results of {} text prompts to text file.'.format(len(prompts)))

    # get output path and filename
    output_dir = Path(__file__).parent.parent.parent / 'output' / \
        str(args_corpus) / 'gpt2_nl' / 'synthetic_data'
    fname = 'synth_' + str(input_path.stem) + '_' + str(max_length) + \
        'maxlen_' + str(args_beam) + 'beams.txt'

    # for each input text, generate text with beam search and print to file
    with open(output_dir / fname, 'w') as f:
        for input_ids in tqdm(id_prompts):
            #pad_token_id = tokenizer_nl.pad_token_id
            eos_token_id = tokenizer_nl.convert_tokens_to_ids(['<|endoftext|>'])[0]
            output_beam = generator_beam(model_nl, input_ids, args_beam, max_length, eos_token_id)
            # from output_beam, pick version with most PHI:
            text = pick_max_phi(output_beam, tokenizer_nl)
            text = text.replace('\n', '<PAR>')
            f.write(text.strip())
            f.write('\n')

            if args.verbose:
                print(text)
                
                
def generate_and_write_p(args_p, input_file, args_corpus, tokenizer_nl, model_nl, max_length):
    # get list of input texts, encode them
    input_path = Path(input_file)
    with open(input_path, 'r') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip() != '']
        print(prompts[:5])
    id_prompts = [tokenizer_nl.encode(p_text, return_tensors='pt').to(device) for p_text in prompts]

    print('Reading from input file.')
    print('P/NUCLEUS-SAMPLING: \nSaving results of {} text prompts to text file.'.format(len(prompts)))

    # get output path and filename
    output_dir = Path(__file__).parent.parent.parent / 'output' / \
        str(args_corpus) / 'gpt2_nl' / 'synthetic_data'
    fname = 'synth_' + str(input_path.stem) + '_' + str(max_length) + \
        'maxlen_p' + str(args_p) + '.txt'

    # for each input text, generate text with beam search and print to file
    with open(output_dir / fname, 'w') as f:
        for input_ids in tqdm(id_prompts):
            #pad_token_id = tokenizer_nl.pad_token_id
            eos_token_id = tokenizer_nl.convert_tokens_to_ids(['<|endoftext|>'])[0]
            output_p = generator_p(model_nl, input_ids, args_p, max_length, eos_token_id)
            text = tokenizer_nl.decode(output_p[0], skip_special_tokens=False)
            text = text.replace('<|endoftext|>', '')
            f.write(text.strip())
            f.write('\n')

            if args.verbose:
                print(text)
                

################################################################################

def main():

    # 1. Import nl model & tokenizer
    specialtoks = get_text(Path(__file__).parent.parent.parent /
                           'data' / 'external' / 'special_tokens.txt')
    tokenizer_nl = load_gpt2tokenizer(dir_path, specialtoks)
    # attempt at making early termination on encounter of eos work:
    model_nl = GPT2LMHeadModel.from_pretrained(dir_path).to(device)
    #model_nl = GPT2LMHeadModel.from_pretrained(dir_path, pad_token = tokenizer_nl.eos_token)
    print("Imported nl gpt2 tokenizer and model")

    # 2. Load dataloaders:
    dls = torch.load(dir_path / 'dls_nl_tokenizerGPT2.pkl')

    # 3. Load learner
    if device == "cuda":
        learn = Learner(dls, model_nl, loss_func=CrossEntropyLossFlat(),
                        splitter=splitter,
                        cbs=[DropOutput],
                        metrics=[accuracy, Perplexity()]).to_fp16()
    else:
        learn = Learner(dls, model_nl, loss_func=CrossEntropyLossFlat(),
                        splitter=splitter,
                        cbs=[DropOutput],
                        metrics=[accuracy, Perplexity()])

    learner = learn.load(dir_path / str(args.checkpoint), device=device)

    # 4. Generate
    # if reading list of input prompts from file, save outputs to file (currently only with beam search)
    if args.input_file and args.beam:
        print('reading input file from path: ' + str(args.input_file))
        generate_and_write_beam(args.beam, args.input_file, args.corpus,
                           tokenizer_nl, model_nl, int(args.max_length))
    
    elif args.input_file and args.p:
        print('reading input file from path: ' + str(args.input_file))
        generate_and_write_p(args.p, args.input_file, args.corpus,
                           tokenizer_nl, model_nl, int(args.max_length))

    else:  # single input, output of any specified sampling strategies will be printed to commandline
        input_text = args.input_text
        print('INPUT TEXT:\n{}'.format(input_text))
        input_ids = tokenizer_nl.encode(input_text, return_tensors='pt').to(device)

        max_length = int(args.max_length)

        if args.beam:
            beams = args.beam  # e.g. 5
            eos_token_id = tokenizer_nl.convert_tokens_to_ids(['<|endoftext|>'])[0]
            output_beam = generator_beam(model_nl, input_ids, beams, max_length, eos_token_id)
            print('BEAM SEARCH')
            print("Output:\n" + 100 * '-')
            text = pick_max_phi(output_beam, tokenizer_nl)
            print(text)
            #for i, beam_output in enumerate(output_beam):
                #text = tokenizer_nl.decode(beam_output, skip_special_tokens=False)
                #print("{}: {}".format(i, text))
                #print()

        if args.p:
            p = args.p  # e.g. 0.95
            eos_token_id = tokenizer_nl.convert_tokens_to_ids(['<|endoftext|>'])[0]
            output_p = generator_p(model_nl, input_ids, p, max_length, eos_token_id)
            print('TOP-P/NUCLEUS SAMPLING')
            print("Output:\n" + 100 * '-')
            print(tokenizer_nl.decode(output_p[0], skip_special_tokens=False))
            print()

        if args.k:
            k = args.k  # e.g. 20
            output_k = generator_k(model_nl, input_ids, k, max_length)
            print('TOP-K SAMPLING')
            print("Output:\n" + 100 * '-')
            print(tokenizer_nl.decode(output_k[0], skip_special_tokens=False))
            print()


################################################################################

if __name__ == '__main__':

    args = parser.parse_args()

    # Handle missing or incorrect combinations of arguments:
    if not (args.k or args.p or args.beam):
        print('No sampling strategy specified, please add argument k, p or beams.')
        sys.exit(1)

    if args.input_file and not(args.beam or args.p):
        print('Can currently only use beam or p search if reading input text from file.')
        print('For --input_file, please use --beam or --p as generation strategy.')
        sys.exit(1)

    if args.input_file and args.input_text:
        print('Warning: You can not use --input_file and --input_text at the same time.')
        print('Output will be printed to file.')

    # enable gpu (or not)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")

    # set model path
    dir_path = Path(__file__).parent.parent.parent / 'output' / \
        str(args.corpus) / 'gpt2_nl' / 'model'

    main()
