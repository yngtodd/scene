import torch.nn as nn
import torch.nn.functional as F

from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper


class BaselineModel(Model):

    def __init__(self, word_embeddings, vocab, encoder, n_classes):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.projection = nn.Linear(self.encoder.get_output_dim(), n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = CategoricalAccuracy()
        
    def forward(self, tokens, id, label):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        state = self.encoder(embeddings, mask)
        logits = self.projection(state)
        output = {"class_logits": logits}

        if label is not None:
            print(f'logits shape: {logits.shape}')
            self.accuracy(F.softmax(logits, dim=1), label, mask)
            output["loss"] = self.criterion(logits, label.long())

        return output

    def get_metrics(self, reset=False):
        return {"accuracy": self.accuracy.get_metric(reset)}