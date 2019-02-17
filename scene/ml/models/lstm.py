import torch
import torch.nn as nn


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


class BiLSTM(nn.Module):

    def __init__(self, num_vocab, n_classes, hparams=Hparams()):
        super().__init__()
        self.hparams = hparams
        self.embedding = nn.Embedding(num_vocab, self.hparams.emb_dim)
        self.encoder = self._get_encoder() 
        self.linear_layers = self._get_linear_layers()
        self.predictor = nn.Linear(self.hparams.hidden_dim * 2, n_classes)

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
    
    def forward(self, seq):
        embedding = self.embedding(seq)
        hdn, _ = self.encoder(embedding)
        feature = hdn[-1, :, :]

        for layer in self.linear_layers:
            feature = layer(feature)

        preds = self.predictor(feature)
        return preds