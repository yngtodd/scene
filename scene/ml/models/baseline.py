import numpy as np
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from overrides import overrides
from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy


class BaselineModel(Model):

    def __init__(self, word_embeddings, vocab, encoder, n_classes):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.projection = nn.Linear(self.encoder.get_output_dim(), n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = CategoricalAccuracy()
        
    def forward(self, tokens, id, labels=None):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        state = self.encoder(embeddings, mask)
        logits = self.projection(state)
        output = {"logits": logits}

        if labels is not None:
            self.accuracy(logits, labels)
            output["loss"] = self.criterion(logits, labels.long())

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