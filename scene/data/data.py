from torchtext.data import TabularDataset 


class DataSet:

    def __init__(self, path):
        self.path = path

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
                ('text', TEXT),
                ('genre', None),
                ('labels', LABEL)
            ]
        )
        return train, val, test