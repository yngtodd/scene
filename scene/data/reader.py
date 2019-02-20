import os
import logging
import numpy as np
import pandas as pd

from allennlp.common import Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import Registrable

from overrides import overrides
from allennlp.data import Instance
from allennlp.data.fields import TextField 
from allennlp.data.fields import MetadataField
from allennlp.data.fields import LabelField

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.dataset_readers.dataset_reader import _LazyInstances

from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer


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

        label_field = LabelField(labels)
        fields["labels"] = label_field

        return Instance(fields)

    @overrides
    def read(self, path, split):
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

        lazy = getattr(self, 'lazy', None)
        if lazy is None:
            logger.warning("DatasetReader.lazy is not set, "
                           "did you forget to call the superclass constructor?")

        if lazy:
            return _LazyInstances(lambda: iter(self._read(path, split)))
        else:
            instances = self._read(path, split)
            if not isinstance(instances, list):
                instances = [instance for instance in Tqdm.tqdm(instances)]
            if not instances:
                raise ConfigurationError("No instances were read from the given filepath {}. "
                                         "Is the path correct?".format(path))
            return instances
    
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
        df = pd.read_csv(filepath)
        
        if split == 'test.csv':
            for i, row in df.iterrows():
                yield self.text_to_instance(
                    [Token(x) for x in self.tokenizer(row["text"])],
                    row["id"]                    
                )
        else:
            for i, row in df.iterrows():
                yield self.text_to_instance(
                    [Token(x) for x in self.tokenizer(row["text"])],
                    row["id"], row["genre"]
                )