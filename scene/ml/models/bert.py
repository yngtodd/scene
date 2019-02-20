from overrides import overrides
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder


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