import os
import torch
import collections


class Dictionary(object):

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = collections.Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1

        return token_id

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):

    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        self.num_tokens = len(self.dictionary)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split()
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        all_lines = []
        with open(path, 'r') as f:
            #ids = torch.LongTensor(tokens)
            for line in f:
                scriptline = []
                words = line.split()
                for word in words:
                    token = self.dictionary.word2idx[word]
                    scriptline.append(token)
                all_lines.append(scriptline)

        return all_lines
