# Sample GPT2 input sequences


import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--corpus',
                    help='name of corpus to sample from (takes test.txt)')
parser.add_argument('--sample_len',
                    type=int,
                    help='number of tokens to sample')
parser.add_argument('--sample_from_start',
                    default=False,
                    action='store_true',
                    help='if set to False, samples from sentence starts anywhere in notes.')


def get_first_n(text, n):
    try:
        toks = text.split(' ')
        if len(toks) >= n:
            return ' '.join(toks[:n])
        else:
            return ''
    except:
        return ''


def main():
    inpath = DATAPATH / 'test.txt'
    outpath = DATAPATH / outfile

    with open(inpath, 'r') as fin:
        testset = fin.readlines()

    # sample from start of sequence
    if args.sample_from_start:
        print('Sampling only from start of EHRs')
        startseqs = [get_first_n(text, args.sample_len) for text in testset]
    # take first n tokens of each sentence in the sequence
    else:
        print('Sampling from any sentence start in EHRs')
        print('Attention: program currently splits on punctuation and falsely assigns e.g. Mr. as end of sentence.')
        sentlists = [t.split(' .') for t in testset]
        sentences = [s for sents in sentlists for s in sents if s != '\n']
        startseqs = [get_first_n(text, args.sample_len) for text in sentences]

    # write to file
    with open(outpath, 'w') as fout:
        for s in startseqs:
            if s != '':
                fout.write(s)
                fout.write('\n')
            else:
                pass


if __name__ == '__main__':
    args = parser.parse_args()

    DATAPATH = Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / str(args.corpus)
    if args.sample_from_start:
        outfile = 'inputs_len' + str(args.sample_len) + '_fromstart.txt'
    else:
        outfile = 'inputs_len' + str(args.sample_len) + '_notfromstart.txt'

    main()
