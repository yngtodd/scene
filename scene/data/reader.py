import os
import numpy as np
import pandas as pd

from allennlp.data.fields import TextField 
from allennlp.data.fields import MetadataField
from allennlp.data.fields import ArrayField

from allennlp.data.dataset_readers import DatasetReader


class DataReader(DatasetReader):

    def __init__(self, tokenizer, token_indexers=None, max_seq_len=1000):
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_seq_len = max_seq_len

    @overrides
    def text_to_instance(self, tokens, id=None, labels=None):
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}
        
        id_field = MetadataField(id)
        fields["id"] = id_field
        
        if labels is None:
            labels = np.zeros(1)

        label_field = ArrayField(array=labels)
        fields["label"] = label_field

        return Instance(fields)
    
    @overrides
    def _read(self, path, split):
        """Read in the data

        Parameters
        ----------
        path : str
            Path to directory containing csv files.

        split : str
            Which data split to load.
            Must be 'train', 'val', or 'test'.
        """
        assert split in ('train', 'val', 'test')

        split = split + '.csv'
        filepath = os.path.join(path, split)
        df = pd.read_csv(file_path)
        
        if split == 'test':
            for i, row in df.iterrows():
                yield self.text_to_instance(
                    [Token(x) for x in self.tokenizer(row["text"])],
                    row["id"]                    
                )
        else:
            for i, row in df.iterrows():
                yield self.text_to_instance(
                    [Token(x) for x in self.tokenizer(row["text"])],
                    row["id"], row["labels"].values,
                )