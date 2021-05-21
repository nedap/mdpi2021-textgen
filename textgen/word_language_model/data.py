# Applied adaptations of melamud&shivade to updated word_language_model from pytorch/examples
import os
from io import open

import torch
from tqdm import tqdm


class Dictionary():
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus():
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.vocab = self.load_vocab(os.path.join(path, 'lstm_vocabulary.txt'))
        print('Processing validation set')
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        print('Processing test set')
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        print('Processing training set')
        self.train = self.tokenize(os.path.join(path, 'train.txt'))

    def load_vocab(self, path):
        assert os.path.exists(path)
        with open(path) as f:
            lines = f.readlines()
            vocab = [w.strip() for w in lines if w.strip()]
        print('Loaded vocabulary of size {}. Tokens not in vocabulary will be replaced with <unk>'.format(len(vocab)))

        # add vocab to dictionary
        self.dictionary.add_word('<eos>')
        self.dictionary.add_word('<unk>')
        for word in vocab:
            self.dictionary.add_word(word)
        return vocab

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Tokenize file content
        unk_idx = self.dictionary.word2idx['<unk>']
        with open(path) as f:
            idss = []
            for line in tqdm(f):
                words = [w.strip() for w in line.split()] + ['<eos>']
                ids = [self.dictionary.word2idx.get(word, unk_idx) for word in words]
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        #print(idss[0])
        #print([self.dictionary.idx2word[i] for i in idss[0]])
        #print()

        return ids

    # partly from m&s: save vocab
    def dump_dict(self, path):
        with open(path, 'w') as f:
            for word in self.dictionary.idx2word:
                f.write(word + '\n')
