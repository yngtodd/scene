import torch
import torch.nn as nn

from allennlp.models import Model


class Hparams:
    def __init__(self, hidden_dim=50, emb_dim=300, 
                 dropout=0.1, num_layers=2, num_linear=1):
        self.hidden_dim = hidden_dim 
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_linear = num_linear

    def __repr__(self):
        return str(Hparams().__dict__)


class BiLSTM(Model):

    def __init__(self, word_embeddings, n_classes, hparams=Hparams()):
        super().__init__()
        """AllenNLP version of our BiLSTM.

        Parameters
        ----------

        word_embeddings : allennlp.modules.text_field_embedders.TextFieldEmbedder

        n_classes : int

        hparams : cls 
        """
        self.hparams = hparams
        self.embedding = word_embeddings
        self.encoder = self._get_encoder() 
        self.linear_layers = self._get_linear_layers()
        self.predictor = nn.Linear(self.hparams.hidden_dim * 2, n_classes)
        self.criterion = nn.CrossEntropyLoss()

    def _get_encoder(self):
        encoder = nn.LSTM(
            self.hparams.emb_dim,
            self.hparams.hidden_dim,
            num_layers = self.hparams.num_layers,
            dropout = self.hparams.dropout,
            bidirectional=True
        )
        return encoder

    def _get_linear_layers(self):
        linear_layers = []
        for _ in range(self.hparams.num_linear-1):
            linear_layers.append(
                nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim)
            )
        linear_layers = nn.ModuleList(linear_layers)
        return linear_layers
    
    def forward(self, tokens, id, label):
        """AllenNLP models are slightly different from nn.Modules.

        Parameters
        ----------
        tokens : Dict[str, torch.Tensor]

        id : Any

        label : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        embedding = self.embedding(tokens)
        hdn, _ = self.encoder(embedding)
        feature = hdn[-1, :, :]

        for layer in self.linear_layers:
            feature = layer(feature)

        logits = self.predictor(feature)
        output = {"class_logits": logits}
        output["loss"] = self.criterion(logits, label)

        return output