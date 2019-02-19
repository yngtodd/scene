import torch
import torch.nn as nn
import torch.nn.functional as F

from overrides import overrides
from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder


class YoonKimConfig:
    # Block0
    in_channels0 = 1
    out_channels0 = 100 
    kernel_size0 = 3
    pool_size0 = 1
    # Block1
    in_channels1 = 1
    out_channels1 = 100 
    kernel_size1 = 4
    pool_size1 = 1
    # Block2
    in_channels2 = 1
    out_channels2 = 100 
    kernel_size2 = 5
    pool_size2 = 1


class ConvBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, pool_size, dropout=0.2):
        super(ConvBlock, self).__init__()
        self.add_module('conv1d', nn.Conv1d(in_channels, out_channels, kernel_size))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('adaptive_maxpool', nn.AdaptiveMaxPool1d(pool_size))
        self.add_module('dropout', nn.Dropout(p=dropout))

    def forward(self, x):
        return super(ConvBlock, self).forward(x)


class YoonKimConv1DEncoder(Seq2VecEncoder):

    def __init__(self, config=YoonKimConfig()):
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
        self.block2 = ConvBlock(
            self.conf.in_channels2, 
            self.conf.out_channels2, 
            self.conf.kernel_size2, 
            self.conf.pool_size2
        )

    def forward(self, x):
        conv_features = []
        conv_features.append(self.block0(x))
        conv_features.append(self.block1(x))
        conv_features.append(self.block2(x)) 
        x = torch.cat(conv_features, 1)
        return x

    @overrides
    def get_output_dim(self) -> int:
        return self.conf.out_channels0 + self.conf.out_channels1 + self.conf.out_channels2


class BertYoonKim(Model):

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
        state = torch.unsqueeze(state, 1)
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