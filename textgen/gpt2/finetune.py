# Inspired by:
# Article: https://medium.com/@pierre_guillou/faster-than-training-from-scratch-fine-tuning-the-english-gpt-2-in-any-language-with-hugging-f2ec05c98787
# Ipynb:https://colab.research.google.com/drive/1B3rgV5maqb-5ZabRBm9eN4wMrSQw0Uni?usp=sharing#scrollTo=XEe0F5shdOw7


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sys

from fastai.text.all import (Callback, CrossEntropyLossFlat, L, Learner,
                             LMDataLoader, Perplexity, TfmdLists, accuracy, nn,
                             params)
from textgen.gpt2.train_tokenizer import (TransformersTokenizer, get_text,
                                          load_gpt2tokenizer)
from transformers import GPT2LMHeadModel

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--corpus',
                    default='dummy',
                    help='Name of corpus with text to train tokenizer.')
parser.add_argument('--load_finetuned',
                    help='name of finetuned file, is \'Learner\' object')
parser.add_argument('--save_model',
                    type=str,
                    help='name under which model should be saved')

################################################################################
# UTILITY FUNCTIONS


ROOT = Path(__file__).parent.parent.parent


def df_from_path(path):
    with open(path) as f:
        lines = (line.strip() for line in f.readlines())
        documents = (t for t in lines if t != '')
        docs_with_endtok = (d + ' <|endoftext|>' for d in documents)
    df = pd.DataFrame(docs_with_endtok, columns=['text'])
    print('Prepared dataset:')
    print(df.head(2))
    return df


def prep_dataset(df_train, df_dev):
    print('Split: {} notes in Train, {} notes in Validation'.format(len(df_train), len(df_dev)))
    df = pd.concat([df_train, df_dev], ignore_index=True)
    idxs = np.arange(start=0, stop=len(df))
    idxs_train = idxs[:len(df_train)]
    idxs_val = idxs[len(df_train):]
    all_texts = np.concatenate([df.iloc[idxs_train].text.values, df.iloc[idxs_val].text.values])
    splits = [list(idxs_train), list(idxs_val)]
    return all_texts, splits


class DropOutput(Callback):
    # Necessary for using Huggingface with fastai.
    # Model will return tuple of outputs (predictions, activations)
    # Need to drop those. Instead, we just leave the first element of pred:
    def after_pred(self): self.learn.pred = self.pred[0]


def splitter(model):
    "Split a GPT2 `model` in 3 groups for differential learning rates."
    # creating 4 layers groups:
    # 3 layers groups of 4 decoder blocks and
    # one embeddings group with the wte and wpe matrices.

    # First layers group : decoder blocks from 0 to 3
    modules = []
    for i in range(4):
        modules.append(model.transformer.h[i])
    groups = [nn.Sequential(*modules)]

    # Second layers group : decoder blocks from 4 to 7
    modules = []
    for i in range(4, 8, 1):
        modules.append(model.transformer.h[i])
    groups = L(groups + [nn.Sequential(*modules)])

    # Third layers group : decoder blocks from 8 to 11
    modules = []
    for i in range(8, 12, 1):
        modules.append(model.transformer.h[i])
    groups = L(groups + [nn.Sequential(*modules)])

    # Fourth layers group : embeddings matrices wte and wpe + LayerNorm at the model output
    groups = L(groups + [nn.Sequential(model.transformer.wte,
                                       model.transformer.wpe, model.transformer.ln_f)])

    return groups.map(params)


def finetune(dls, model_nl, model_path, export_model_name):
    print('Starting finetuning.')

    learn = Learner(dls, model_nl, loss_func=CrossEntropyLossFlat(),
                    splitter=splitter,
                    cbs=[DropOutput],
                    metrics=[accuracy, Perplexity()])

    if torch.cuda.is_available():
        # If we are on a GPU, use mixed precision training for smaller memory footprint.
        learn = learn.to_fp16()

    learn.freeze()
    #suggested_lr = learn.lr_find()
    #lr = suggested_lr.lr_steep
    #print(lr)
    lr = 2e-3
    learn.fit_one_cycle(1, lr)
    print('Froze all, fitted one cycle.')
    # learn.recorder.plot_loss()
    learn.save(model_path / 'GPT2_nl_1epoch_lr2e-3')
    learn.freeze_to(-2)  # freeze all layers but last 2 layer groups
    learn.fit_one_cycle(1, slice(1e-3 / (2.6**4), 1e-3))
    # learn.recorder.plot_loss()
    learn.save(model_path / 'GPT2_nl_2epoch_lr1e-3')
    learn.freeze_to(-3)  # freeze all layers but last 3 layer groups
    learn.fit_one_cycle(1, slice(5e-4 / (2.6**4), 5e-4))
    # learn.recorder.plot_loss()
    learn.save(model_path / 'GPT2_nl_3epoch_lr5e-3')
    learn.unfreeze()  # unfreeze all layers
    learn.fit_one_cycle(2, slice(1e-4 / (2.6**4), 1e-4))  # +2 epochs
    # learn.recorder.plot_loss()
    learn.save(model_path / 'GPT2_nl_5epoch_lre-4')

    # can continue fine-tuning after unfreezing all:
    learn.fit_one_cycle(10, lr)
    learn.save(model_path / 'finetuned')

    # save model properly to later load with from_pretrained:
    if export_model_name:
        print('saving model as {}'.format(export_model_name))
        newmodelpath = model_path/export_model_name
        newmodelpath.mkdir(exist_ok=True, parents=True)
        # get pytorch model from learner
        model_finetuned = learn.model
        # save tokenizer and model
        model_finetuned.save_pretrained(f"{str(newmodelpath)}")
        tokenizer_nl.save_pretrained(f"{str(newmodelpath)}")


def validate_model(dls, model_nl, model_path, learner_name, tokenizer_nl, export_model_name):
    print('loading finetuned learner')

    learn = Learner(dls, model_nl, loss_func=CrossEntropyLossFlat(),
                    splitter=splitter,
                    cbs=[DropOutput],
                    metrics=[accuracy, Perplexity()])

    if torch.cuda.is_available():
        # If we are on a GPU, use mixed precision training for smaller memory footprint.
        learn = learn.to_fp16()

    learn = learn.load(model_path / str(learner_name))
    
    print('Model summary:')
    print(learn.summary())

    print('validating...')
    l, a, p = learn.validate()
    print('Model validation with given checkpoint:')
    print('Loss: {}'.format(l))
    print('Accuracy: {}'.format(a))
    print('Perplexity: {}'.format(p))
    
    # save model properly to later load with from_pretrained:
    if export_model_name:
        print('saving model as {}'.format(export_model_name))
        newmodelpath = model_path/export_model_name
        newmodelpath.mkdir(exist_ok=True, parents=True)
        # get pytorch model from learner
        model_finetuned = learn.model
        # save tokenizer and model
        model_finetuned.save_pretrained(f"{str(newmodelpath)}")
        tokenizer_nl.save_pretrained(f"{str(newmodelpath)}")


################################################################################

def main(args):
    model_path = ROOT / 'output' / args.corpus / 'gpt2_nl' / 'model'
    export_model_name = args.save_model if args.save_model else None

    # 1. Import nl model & tokenizer
    specialtoks = get_text(ROOT / 'data' / 'external' / 'special_tokens.txt')
    tokenizer_nl = load_gpt2tokenizer(model_path, specialtoks)
    model_nl = GPT2LMHeadModel.from_pretrained(model_path)
    print("Imported nl gpt2 tokenizer and model")

    # NO FINETUNING - JUST VALIDATE FROM CHECKPOINT
    if args.load_finetuned:
        # 2. Load dataloaders:
        dls = torch.load(model_path / 'dls_nl_tokenizerGPT2.pkl')

        # 3. Load learner:
        validate_model(dls, model_nl, model_path, args.load_finetuned, tokenizer_nl, export_model_name)
        return

    ######
    # 2. Import all data
    path_train = ROOT / 'data' / 'preprocessed' / args.corpus / 'train.txt'
    path_dev = ROOT / 'data' / 'preprocessed' / args.corpus / 'valid.txt'
    df_train = df_from_path(path_train)
    df_dev = df_from_path(path_dev)

    # 3. Recreate split within one dataframe
    all_texts, splits = prep_dataset(df_train, df_dev)

    # 4. Create fastai Datasets & Dataloaders
    tls = TfmdLists(all_texts, TransformersTokenizer(
        tokenizer_nl), splits=splits, dl_type=LMDataLoader)
    # specify batch size and sequence length:
    # GPT-2 model was trained with sequences of size 1024,
    # so we use this sequence length. Changing it will affect perplexity.

    # bs,sl = 8,1024 # CUDA limit
    bs, sl = 2, 1024
    dls = tls.dataloaders(bs=bs, seq_len=sl)
    print('Dataloaders setup done.')

    # (save dataloaders)
    torch.save(dls, model_path / 'dls_nl_tokenizerGPT2.pkl')

    # 5. Fine tuning
    finetune(dls, model_nl, model_path, export_model_name)


################################################################################

if __name__ == '__main__':
    main(parser.parse_args())
