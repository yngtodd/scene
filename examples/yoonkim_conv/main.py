import torch
import torch.nn as nn
import torch.optim as optim
from parser import parse_args

from scene.data.reader import DataReader
from scene.data.tokenizers import spacy_word_tokenizer

from scene.ml.models import BertYoonKim 
from scene.ml.models import YoonKimConv1DEncoder
from scene.ml.models import BertSentencePooler

from allennlp.training.trainer import Trainer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator

from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder


def tokenizer(s: str):
    return token_indexer.wordpiece_tokenizer(s)[:1000 - 2]


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")

    bert_embedder = PretrainedBertEmbedder(
        pretrained_model="bert-base-uncased",
        top_layer_only=True, # conserve memory
    )

    global token_indexer
    token_indexer = PretrainedBertIndexer(
        pretrained_model="bert-base-uncased",
        max_pieces=1000,
        do_lowercase=True,
    )

    word_embeddings = BasicTextFieldEmbedder(
        {"tokens": bert_embedder},
        # we'll be ignoring masks so we'll need to set this to True
        allow_unmatched_keys = True
    )

    reader = DataReader(
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer}
    )

    iterator = BucketIterator(
        batch_size=args.batch_size,
        sorting_keys=[("tokens", "num_tokens")]
    )

    traindata = reader.read(args.datapath, 'train')
    valdata = reader.read(args.datapath, 'val')
    testdata = reader.read(args.datapath, 'test')

    vocab = Vocabulary()
    iterator.index_with(vocab)

    embedding_dim = word_embeddings.get_output_dim()
    bert_pooler = BertSentencePooler(vocab, embedding_dim)
    encoder = YoonKimConv1DEncoder()

    model = BertYoonKim(
        word_embeddings,
        vocab,
        bert_pooler,
        encoder,
        n_classes=9
    )#.to(device)

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
        cuda_device=-1,
        patience=10,
        num_epochs=args.num_epochs,
    )

    metrics = trainer.train()
    print(metrics)


if __name__=='__main__':
    main()
