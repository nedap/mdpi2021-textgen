# get test perplexity of gpt2 model
# requires model saved as {modelname}.pt (use --save_model 'modelname' in finetune.py)
# Implementation from: https://huggingface.co/transformers/perplexity.html#example-calculating-perplexity-with-gpt-2-in-transformers

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--corpus',
                    default ='dummy',
                    help='Name of corpus with text to train tokenizer.')
parser.add_argument('--modelpath',
                    help='path to model (directory) of which to calculate perplexity')


def get_ppl(model, encodings, cuda):
    max_length = model.config.n_positions
    stride = 512

    lls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        
        if cuda:
            input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
        else:
            input_ids = encodings.input_ids[:,begin_loc:end_loc]
            
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl


def main():
    # 1. Import nl model & tokenizer
    model_nl = GPT2LMHeadModel.from_pretrained(args.modelpath)
    tokenizer_nl = GPT2Tokenizer.from_pretrained(args.modelpath)
    if args.cuda:
        model_nl.to(device)
        
    print("Imported nl gpt2 tokenizer and model")
    model_nl.eval()

    # 2. Import test set & encode
    testpath = Path(__file__).parent.parent.parent /'data'/'preprocessed'/str(args.corpus)/'test.txt'
    with open(testpath,'r') as fin:
        lines = fin.readlines()
    test_text = '\n\n'.join([line.strip() for line in lines])
    encoded_testset = tokenizer_nl(test_text, return_tensors='pt')
    if args.cuda:
        encoded_testset.to(device)

    # 3. Get test ppl
    ppl = get_ppl(model_nl, encoded_testset, args.cuda)
    print(ppl)


if __name__ == '__main__':

    args = parser.parse_args()

    # enable gpu (or not)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")

    main()
