# Inspired by:
# Article: https://medium.com/@pierre_guillou/faster-than-training-from-scratch-fine-tuning-the-english-gpt-2-in-any-language-with-hugging-f2ec05c98787
# Ipynb:https://colab.research.google.com/drive/1B3rgV5maqb-5ZabRBm9eN4wMrSQw0Uni?usp=sharing#scrollTo=XEe0F5shdOw7
# Additional resource: https://www.analyticsvidhya.com/blog/2020/06/hugging-face-tokenizers-nlp-library/


import argparse
import os
from pathlib import Path

import torch
from fastai.text.all import TitledStr, Transform, nn, tensor
from tokenizers import AddedToken, ByteLevelBPETokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer  # , GPT2TokenizerFast

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--corpus',
                    default='dummy',
                    help='Name of corpus with text to train tokenizer.')
parser.add_argument('--run_test',
                    type=bool,
                    default=False,
                    help='Set \'True\' to encode example sentence after training.')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')


################################################################################
# UTILITY FUNCTIONS: TRAINING

def get_text(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [l.replace('\n', '') for l in lines]
    return lines


def retrain_tokenizer(tokenizer_en, training_text, specialtoks, enable_trunc: bool):
    ByteLevelBPE_tokenizer_nl_vocab_size = tokenizer_en.vocab_size
    ByteLevelBPE_tokenizer_nl = ByteLevelBPETokenizer()
    ByteLevelBPE_tokenizer_nl.add_prefix_space = True
    ByteLevelBPE_tokenizer_nl.is_split_into_words = True
    #ByteLevelBPE_tokenizer_nl.additional_special_tokens = get_text(Path('textgen/gpt2/special_tokens.txt'))

    # Get list of paths to corpus files (in this case we have one)
    # and customize training with special/custom tokens
    data_paths = [str(training_text)]
    ByteLevelBPE_tokenizer_nl.train(files=data_paths,
                                    vocab_size=ByteLevelBPE_tokenizer_nl_vocab_size,
                                    min_frequency=2,
                                    special_tokens=specialtoks
                                    )

    # Get sequence length max of 1024 (is it necessary?)
    if enable_trunc:
        ByteLevelBPE_tokenizer_nl.enable_truncation(max_length=1024)

    return ByteLevelBPE_tokenizer_nl


def test_encoding(ByteLevelBPE_tokenizer_nl):
    sentence = "<NameSTART> Chrystal <NameEND> gaf aan haar medicatie nu . Volgens <NameSTART> Chrystal <NameEND> gaat dit goed ."
    enc = ByteLevelBPE_tokenizer_nl.encode(sentence)
    print(enc.tokens)


# UTILITY FUNCTIONS/CLASSES: MODEL CONFIG

def load_gpt2tokenizer(path, specialtoks):
    # import the pre-trained GPT2TokenizerFast tokenizer with the tokenizer_nl config files
    tokenizer_nl = GPT2Tokenizer.from_pretrained(
        str(path),
        pad_token='<|endoftext|>',
        add_prefix_space=True,
        model_max_length=1024
    )

    # `transformers` distinguishes between two types of tokens: "str" and "AddedToken".
    # str tokens are by default stripped on left and right side. AddedToken (see tokenizer package)
    # are *not* stripped. See:
    # 1) https://github.com/huggingface/transformers/blob/v3.3.1/src/transformers/tokenization_utils.py#L287-L315
    # 2) https://github.com/huggingface/tokenizers/blob/python-v0.9.2/bindings/python/py_src/tokenizers/__init__.pyi#L424-L437
    tokenizer_nl.add_special_tokens({
        'additional_special_tokens': [AddedToken(token) for token in specialtoks]
    })

    return tokenizer_nl


class TransformersTokenizer(Transform):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encodes(self, x):
        toks = self.tokenizer.tokenize(x)
        return tensor(self.tokenizer.convert_tokens_to_ids(toks))
        #ids = self.tokenizer.encode(x.split(), is_split_into_words=True)
        #toks = self.tokenizer.convert_ids_to_tokens(ids)
        # return tensor(self.tokenizer.convert_tokens_to_ids(toks))

    def decodes(self, x):
        return TitledStr(self.tokenizer.decode(x.cpu().numpy()))


def check_tokenizer_fastai(tokenizer_fastai_nl):
    text = "<NameSTART> Chrystal <NameEND> gaf aan haar medicatie nu . Volgens <NameSTART> Chrystal <NameEND> gaat dit goed ."
    tokens_ids = tokenizer_fastai_nl.encodes(text)
    tokens = tokenizer_fastai_nl.tokenizer.convert_ids_to_tokens(tokens_ids)

    print('input text:', TitledStr(text))
    print('text tokens:', TitledStr(tokens))
    print('text tokens_ids:', TitledStr(tokens_ids))
    print('output text:', TitledStr(tokenizer_fastai_nl.decodes(tokens_ids)))


def setup_new_embeddings(model_en, tokenizer_fastai_nl, tokenizer_fastai_en, dir_path):
    # Change vocab embedding in the GPT-2 pre-trained model to adapt to the Dutch vocab

    # Check atual weight of wte and lm_head and if wte = lm_head
    #tens_a = model_en.transformer.wte.weight
    #tens_b = model_en.lm_head.weight
    # print(model_en.transformer.wte.weight,model_en.lm_head.weight,torch.all(tens_a.eq(tens_b)))

    # 1. Get old weights, mean of..
    old_wgts = model_en.transformer.get_input_embeddings().weight.clone().detach()
    wgts_m = old_wgts.mean(0)

    # 2. Initialize new wte
    new_vocab_size = tokenizer_fastai_nl.tokenizer.vocab_size
    new_wgts = old_wgts.new_zeros(new_vocab_size, old_wgts.size(1))

    # 3. Get the new wte
    # keep the embedding vectors of tokens found in both vocabs
    # new vocab gets mean embedding vector of old
    old_vocab = tokenizer_fastai_en.tokenizer.get_vocab()
    new_vocab = tokenizer_fastai_nl.tokenizer.get_vocab()
    same_tokens_list = list()
    different_tokens_list = list()

    for w, idx_new in new_vocab.items():
        idx_old = old_vocab.get(w, -1)
        if idx_old >= 0:
            new_wgts[idx_new] = old_wgts[idx_old]
            same_tokens_list.append((w, idx_new))
        else:
            new_wgts[idx_new] = wgts_m
            different_tokens_list.append((w, idx_new))

    # 4. put new wte in model
    new_wte = nn.Embedding(new_vocab_size, old_wgts.size(1))
    new_wte.weight.data = new_wgts
    model_en.transformer.set_input_embeddings(new_wte)

    # 5. save new weights, same and different tokens lists
    torch.save(new_wgts, dir_path + '/new_wte_wgts.pt')
    torch.save(same_tokens_list, dir_path + '/same_tokens_list.pt')
    torch.save(different_tokens_list, dir_path + '/different_tokens_list.pt')

    return model_en


################################################################################

def main():

    # TRAINING TOKENIZER #######################################################

    # 1. Get pre-trained GPT2 Tokenizer
    pretrained_weights = 'gpt2'
    tokenizer_en = GPT2Tokenizer.from_pretrained(pretrained_weights)
    tokenizer_en.pad_token = tokenizer_en.eos_token

    # 2. Train Byte Level BPE tokenizer on Dutch dataset
    training_text = Path(__file__).parent.parent.parent / 'data' / \
        'preprocessed' / str(args.corpus) / 'all.txt'
    specialtoks = get_text(Path(__file__).parent.parent.parent /
                           'data' / 'external' / 'special_tokens.txt')
    tokenizer_nl = retrain_tokenizer(tokenizer_en, str(
        training_text), specialtoks, enable_trunc=True)

    # (Optional) Encoding test
    if args.run_test:
        print('Encoding test: Retrained BPE Tokenizer')
        test_encoding(tokenizer_nl)

    # 3. Save tokenizer
    tokenizer_nl.save_model(dir_path)

    # ADJUST VOCABULARY EMBEDDING OF MODEL #####################################

    # 1. Load pre-trained (en) GPT2 model & tokenizer
    pretrained_weights = 'gpt2'
    model_en = GPT2LMHeadModel.from_pretrained(pretrained_weights)
    tokenizer_en = load_gpt2tokenizer(pretrained_weights, specialtoks=[])
    tokenizer_fastai_en = TransformersTokenizer(tokenizer_en)

    # 2. Setup new embedding matrix (change vocab and vocab embeddings)
    # (To do so, re-load Dutch tokenizer)
    tokenizer_nl = load_gpt2tokenizer(dir_path, specialtoks)
    tokenizer_fastai_nl = TransformersTokenizer(tokenizer_nl)
    model_nl = setup_new_embeddings(model_en, tokenizer_fastai_nl, tokenizer_fastai_en, dir_path)

    # (Optional) Encoding/decoding test
    if args.run_test:
        print('Encoding/decoding test: fastai GPT2 Tokenizer')
        check_tokenizer_fastai(tokenizer_fastai_nl)

    # 3. Changing lm_head weights with the new embedding
    model_nl.lm_head.weight = model_nl.transformer.wte.weight
    print(model_nl.lm_head)

    # 4. Pickle final model
    model_nl.save_pretrained(dir_path)


################################################################################

if __name__ == '__main__':

    args = parser.parse_args()

    # set model path: if doesn't exist yet, make directory
    dir_path = str(Path(__file__).parent.parent.parent / 'output' /
                   str(args.corpus) / 'gpt2_nl' / 'model')
    print(dir_path)
    if not os.path.exists(dir_path):
        print('Creating output directory for Dutch GPT2 model')
        os.makedirs(dir_path)
    else:
        print('Using existing direcotry for Dutch GPT2 model.\nAttention: existing files will be replaced.\n')

    # enable gpu (or not)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")

    main()
