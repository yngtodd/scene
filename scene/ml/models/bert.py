from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from overrides import overrides
from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder


class BertSentencePooler(Seq2VecEncoder):

    def __init__(self, vocab, embedding_dim):
        super().__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim


    def forward(self, embs, mask=None):
        # extract first token tensor
        return embs[:, 0]
    
    @overrides
    def get_output_dim(self) -> int:
        return self.embedding_dim 


class Conv2dEncoderConfig:
    # Block0
    in_channels0 = 1
    out_channels0 = 100
    kernel_size0 = 3
    pool_size0 = 100
    # Block1
    in_channels1 = 100
    out_channels1 = 100
    kernel_size1 = 4
    pool_size1 = 100
    # Block2
    in_channels2 = 100
    out_channels2 = 1
    kernel_size2 = 5
    pool_size2 = 10


class ConvBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, pool_size, dropout=0.2):
        super(ConvBlock, self).__init__()
        self.add_module('conv2d', nn.Conv2d(in_channels, out_channels, kernel_size))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('adaptive_maxpool', nn.AdaptiveMaxPool2d(pool_size))
        self.add_module('dropout', nn.Dropout(p=dropout))

    def forward(self, x):
        return super(ConvBlock, self).forward(x)


class Conv2dEncoder(Seq2VecEncoder):

    def __init__(self, config=Conv2dEncoderConfig()):
        super().__init__()
        self.conf = config
        self.block0 = ConvBlock(
            self.conf.in_channels0, 
            self.conf.out_channels0, 
            self.conf.kernel_size0, 
            self.conf.pool_size0
        )
        self.block1 = ConvBlock(
            self.conf.in_channels1, 
            self.conf.out_channels1, 
            self.conf.kernel_size1, 
            self.conf.pool_size1
        )
        self.block1 = ConvBlock(
            self.conf.in_channels2, 
            self.conf.out_channels2, 
            self.conf.kernel_size2, 
            self.conf.pool_size2
        )

    def forward(self, x):
        print_shape('x input encoder', x)
        x = self.block0(x)
        print_shape('block 0 out', x)
        x = self.block1(x)
        print_shape('block 1 out', x)
        x = self.block2(x)
        return x

    @overrides
    def get_output_dim(self) -> int:
        return self.conf.pool_size2**2 


class BertModel2D(Model):

    def __init__(self, word_embeddings, vocab, bertpooler, encoder, n_classes):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.bertpooler = bertpooler
        self.encoder = encoder 
        self.projection = nn.Linear(self.encoder.get_output_dim(), n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = CategoricalAccuracy()

    def forward(self, tokens, id, label):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        state = self.bertpooler(embeddings, mask)
        print_shape('state prior', state) 
        state = torch.unsqueeze(state, 1)
        print_shape('shape post', state)
        features = self.encoder(state)
        features = features.view(features.size(0), -1)
        logits = self.projection(features)
        output = {"class_logits": logits}

        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.criterion(logits, label.long())

        return output

    def get_metrics(self, reset=False):
        return {"accuracy": self.accuracy.get_metric(reset)}

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict


def print_shape(name, x):
    print(f'{name} has shape {x.shape}')