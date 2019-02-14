import spacy
from torchtext.data import Field
from torchtext.data import TabularDataset 


try:
    spacy_en = spacy.load('en')
except:
    print('Spacy requires its English tokenization library!')
    print('Install with `python -m spacy download en`')
    break


def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


class DataSet:

    def __init__(self, path):
        self.path = path
        self.textfield = Field(sequential=True, tokenize=tokenizer, lower=True)
        self.labelfield = Field(sequential=False, use_vocab=True)

    def __repr__(self):
        return f'Competition dataset at {self.path}'

    def load_splits(self):
        print(f'Tokenizing data...')
        train, val, test = TabularDataset.splits(
            path=self.path, 
            train='train.csv',
            validation='val.csv', 
            test='test.csv', 
            format='csv',
            fields=[
                ('id', None),
                ('text', self.textfield),
                ('genre', self.labelfield),
                ('labels', None)
            ]
        )

        self.textfield.build_vocab(train, val, test, vectors='glove.6B.100d')
        self.labelfield.build_vocab(train, val)
        return train, val, test
