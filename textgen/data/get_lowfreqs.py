import re
from collections import Counter
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('corpus',
                    help='Will return low frequency tokens for this corpus.')
parser.add_argument('--min_wordfreq',
                    type=int,
                    help='Minimum word frequency for word to be included in LSTM training.')

#############################################################################

def path2toks(path):
    # Takes all lines in text file & concats them with ' '
    # Tokenizes merged text by splitting on ' '
    # Returns list with tokens
    with open(path) as fin:
        raw = fin.readlines()
        lines = [l.strip() for l in raw if l.strip()]
    merged_text = ' '.join(lines)
    tokens = merged_text.split(' ')
    return tokens

def get_lowfreq(tokens,min_wordfreq):
    # exclude special tokens
    specialtok_pattern = r'<[A-Za-z\_]+((START)|(END))>'
    words = [t for t in tokens if not re.match(specialtok_pattern,t)]
    # get words with min. freq. 3
    wordfreqs = Counter(words)
    lowfreq_words = [token for token, count in wordfreqs.items() if count < min_wordfreq]
    return lowfreq_words

#############################################################################


def main(args):
    corpuspath = Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / str(args.corpus)
    trainingset_filepath = corpuspath / 'train.txt'

    print("Minimum word frequency set to {}".format(args.min_wordfreq))
    print("Finding low frequency tokens for corpus: {}".format(args.corpus))
    tokenized = path2toks(trainingset_filepath)
    lowfreqtoks = get_lowfreq(tokenized, args.min_wordfreq)
    lowfreqtoks = set(tok for tok in lowfreqtoks if tok.strip())
    print("{} of {} tokens are low frequency (less than {} occurrences):".format(len(lowfreqtoks), len(set(tokenized)), args.min_wordfreq))
    
    # remaining vocab:
    new_vocab = list(set(tokenized) - set(lowfreqtoks))
    print("Resulting vocab size: {}".format(len(new_vocab)))
        
    # print new lstm_vocabulary to output file, i.e. just tokens we want to INCLUDE
    vocab_filepath = corpuspath / 'lstm_vocabulary.txt'
    with open(vocab_filepath,'w') as fout:
        for tok in new_vocab:
            fout.write("{}\n".format(tok))
    print("Written tokens to be included in LSTM vocabulary to corpus folder.")


if __name__ == "__main__":
    main(parser.parse_args())
