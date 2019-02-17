import torch.nn as nn

from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper


class BaselineModel(Model):

    def __init__(self, word_embeddings, vocab, encoder, n_classes):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.projection = nn.Linear(self.encoder.get_output_dim(), n_classes)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, tokens, id, label):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        state = self.encoder(embeddings, mask)
        logits = self.projection(state)
        
        output = {"class_logits": logits}
        output["loss"] = self.criterion(logits, label)

        return output