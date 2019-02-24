import torch
import torch.nn as nn
from parser import parse_args

import pandas as pd

from scene.data.reader import DataReader
from scene.data.tokenizers import spacy_word_tokenizer

from scene.ml.predictor import Predictor
from scene.ml.models import BaselineModel
from allennlp.modules.seq2vec_encoders import CnnEncoder

from allennlp.training.trainer import Trainer
from allennlp.data.vocabulary import Vocabulary

from allennlp.data.iterators import BasicIterator
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:3" if use_cuda else "cpu")

    reader = DataReader(
        tokenizer=spacy_word_tokenizer,
    )

    testdata = reader.read(args.datapath, 'test')
    iterator = BasicIterator(batch_size=28)

    vocab = Vocabulary.from_files("./experiments/augment_sans_other/final_save/vocabulary")
    iterator.index_with(vocab)

    token_embedding = Embedding(
        num_embeddings=vocab.get_vocab_size('tokens'),
        embedding_dim=args.embedding_dim
    )

    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    encoder = CnnEncoder(
        embedding_dim=word_embeddings.get_output_dim(),
        num_filters=100,
        ngram_filter_sizes=(2, 3, 4, 5)
   )

    model = BaselineModel(
        word_embeddings,
        vocab,
        encoder,
        n_classes=9
    )

    with open("./experiments/augment_sans_other/saves/best.th", 'rb') as f:
        model.load_state_dict(torch.load(f))

    predictor = Predictor(model, iterator)
    out = predictor.predict(testdata)

    outfile = 'submission.csv'
    print(f'Creating submission file - {outfile}')
    decoded = model.decode(out)
    submission = {'id': decoded['id'], 'genre': decoded['label']}
    submission = pd.DataFrame(submission)
    submission.to_csv(outfile, index=False)


if __name__=='__main__':
    main()
