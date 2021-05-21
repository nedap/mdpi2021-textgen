# Preprocessing. Get all, training, valid, test set as txt files.
# Run with: python -m textgen.data.preprocessing  --in_dir data/interim/annotated_dummy.jsonl --out_dir data/preprocessed/NAME


from pathlib import Path
import jsonlines
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Token
from tqdm import tqdm
import numpy as np
import argparse

###########################################################################


#args how many words to process:
parser = argparse.ArgumentParser()
parser.add_argument('--n_words', 
                    type = int,
                    action='store',
                    default=False,
                    help='Number of words to be preprocessed. Default: no size reduction, all notes are processed.')
parser.add_argument('--in_dir',
                    help='Location of your data to be preprocessed: data/interim/...')
parser.add_argument('--out_dir',
                    help='Target directory for preprocessed dataset: data/preprocessed/your_dataset_dir')
arg = parser.parse_args()
    
#DATA_PATH = Path(__file__).parent.parent.parent / 'data' /'interim'/'annotated_ehr.jsonl'
DATA_PATH = Path(arg.in_dir)
PREPROC_PATH = Path(arg.out_dir)

###########################################################################

class merge_special(object):
    def __init__(self, nlp):
        # Register a new token extension to flag bad HTML
        Token.set_extension("special_tok", default=False, force=True)
        self.matcher = Matcher(nlp.vocab)
        self.matcher.add(
            "SPECIALTOK",
            None,
            [{"ORTH": "<"}, {"TEXT": {"REGEX": "\w+(?:START|END)"}}, {"ORTH": ">"}],
            [{"ORTH": "<"}, {"ORTH": "PAR"}, {"ORTH": ">"}],
        )

    def __call__(self, doc):
        # This method is invoked when the component is called on a Doc
        matches = self.matcher(doc)
        spans = []  # Collect the matched spans here
        for match_id, start, end in matches:
            spans.append(doc[start:end])
        with doc.retokenize() as retokenizer:
            for span in spans:
                retokenizer.merge(span)
                for token in span:
                    token._.special_tok = True  # Mark token as bad HTML
        return doc

    
def tokens2file(filepath,tokenized_reports):
    with open(filepath,'w') as f:
        for report in tokenized_reports:
            text = ' '.join([t.text for t in report if t.text.strip()])
            f.write(text.strip())
            f.write("\n")

def read_document(data_path):
    print('reading data')
    data = []
    f = jsonlines.open(data_path)
    for line in f.iter(skip_empty=True, skip_invalid=True):
        data.append(line['annotated_text'])
    print('Found {} documents.'.format(len(data)))
    return data

def analysis(tokenized_reports):
    # check corpus and vocab size
    tokens = [tok for report in tokenized_reports for tok in report if tok.text!=' ']
    words = [tok.text for tok in tokens]
        
    print('Without whitespace tokens, number of tokens in corpus: {}'.format(len(tokens)))
    
    vocab = set(words)
    print('Vocabulary size: {}'.format(len(vocab)))

def adjust_datasize(reports, n_words):
    n = 0
    smallset = []
    for r in reports:
        if n < n_words:
            replen = len(r.split(' '))
            if n+replen <= n_words:
                smallset.append(r)
                n += replen
            else:
                break
        else:
            break
    print('Dataset was shuffled and cut to contain {} words and {} reports of {}.'.format(n,len(smallset),len(reports)))
    return smallset

###########################################################################


def main():
    
    # import data
    data = read_document(DATA_PATH)
    
    #### OPTIONAL: Constrict dataset to n tokens or max possible ####
    if arg.n_words:
        print('Taking subset of reports with a total of max. {} words from shuffled original corpus'.format(arg.n_words))
        n_words = arg.n_words
        random_state = np.random.RandomState(seed=42)
        random_state.shuffle(data)
        data = adjust_datasize(data, n_words)
    else:
        print('Whole input corpus will be preprocessed.')
    
    # load tokenizer and add special tokens, tokenize
    print('tokenizing')
    nlp = spacy.load('nl_core_news_lg')
    special_merger = merge_special(nlp)
    nlp.add_pipe(special_merger, last=True)  # Add component to the pipeline
    tokenized = [nlp(report) for report in tqdm(data)]
    
    #### Analysis ####
    analysis(tokenized)
    
    #### Randomly sample and split into datasets ####
    random_state = np.random.RandomState(seed=42)
    random_state.shuffle(tokenized)
    
    #sample
    train_until_i = round(len(tokenized)*.8)
    valid_until_i = train_until_i + round(len(tokenized)*.1)
    
    training = tokenized[:train_until_i]
    valid = tokenized[train_until_i:valid_until_i]
    test = tokenized[valid_until_i:]
    
    print('Of {} records (in all.txt), setsizes are:\nTrain - {}\nValid - {}\nTest - {}\n'.format(
        len(tokenized),
        len(training),
        len(valid),
        len(test)
        ))
    
    #save to text files, concat tokens with simple whitespace
    tokens2file(PREPROC_PATH/'all.txt', tokenized)
    tokens2file(PREPROC_PATH/'train.txt', training)
    tokens2file(PREPROC_PATH/'valid.txt', valid)
    tokens2file(PREPROC_PATH/'test.txt', test)
    
    print('Saved shuffled, preprocessed data to files.')


if __name__=="__main__":
    main()
    
