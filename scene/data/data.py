import spacy
from torchtext.data import Field
from torchtext.data import TabularDataset 


def tokenizer(text):
    spacy_en = spacy.load('en')
    return [tok.text for tok in spacy_en.tokenizer(text)]


class DataSet:

    def __init__(self, path):
        self.path = path
        self.textfield = Field(sequential=True, tokenize=tokenizer, lower=True)
        self.labelfield = Field(sequential=False, use_vocab=False)

    def __repr__(self):
        return f'Competition dataset at {self.path}'

    def load_splits(self):
        train, val, test = TabularDataset.splits(
            path=self.path, 
            train='train.csv',
            validation='val.csv', 
            test='test.csv', 
            format='csv',
            fields=[
                ('id', None),
                ('text', self.textfield),
                ('genre', None),
                ('labels', self.labelfield)
            ]
        )
        return train, val, test