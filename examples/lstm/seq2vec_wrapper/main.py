import torch
import torch.nn as nn
import torch.optim as optim
from parser import parse_args

from scene.data.reader import DataReader
from scene.data.tokenizers import spacy_word_tokenizer

from scene.ml.models import BaselineModel

from allennlp.training.trainer import Trainer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper

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

    iterator = BucketIterator(
        batch_size=args.batch_size,
        sorting_keys=[("tokens", "num_tokens")]
    )

    traindata = reader.read(args.datapath, 'train')
    valdata = reader.read(args.datapath, 'val')
    testdata = reader.read(args.datapath, 'test')

    vocab = Vocabulary.from_instances(traindata + valdata + testdata)
    iterator.index_with(vocab)

    token_embedding = Embedding(
        num_embeddings=vocab.get_vocab_size('tokens'),
        embedding_dim=args.embedding_dim
    )

    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    lstm = PytorchSeq2VecWrapper(
        nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=25,
            batch_first=True,
            dropout=0.5,
            bidirectional=False
        )
    )

    model = BaselineModel(
        word_embeddings,
        vocab,
        lstm,
        n_classes=9
    ).to(device)

    optimizer = optim.Adam(model.parameters())

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        validation_metric='+accuracy',
        serialization_dir=args.serialization_dir,
        iterator=iterator,
        train_dataset=traindata,
        validation_dataset=valdata,
        validation_iterator=iterator,
        cuda_device=3,
        patience=15,
        num_epochs=args.num_epochs,
    )

    metrics = trainer.train()

    with open("./final_save/model.th", 'wb') as f:
        torch.save(model.state_dict(), f)

    vocab.save_to_files("./final_save/vocabulary")


if __name__=='__main__':
    main()
