import torch
import torch.optim as optim
from parser import parse_args

from scene.data.reader import DataReader
from scene.data.tokenizers import spacy_word_tokenizer
from scene.ml.models import AllenBiLSTM

from allennlp.training.trainer import Trainer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator

from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    token_indexer = ELMoTokenCharactersIndexer()

    reader = DataReader(
        tokenizer=spacy_word_tokenizer,
        token_indexers={"tokens": token_indexer}
    )

    iterator = BucketIterator(
        batch_size=args.batch_size, 
        sorting_keys=[("tokens", "num_tokens")],
    )

    vocab = Vocabulary()
    iterator.index_with(vocab)

    traindata = reader.read(args.datapath, 'train')

    elmo_embedder = ElmoTokenEmbedder(args.options_file, args.weight_file)
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})

    model = AllenBiLSTM(word_embeddings, vocab, n_classes=9)#.to(device)
    optimizer = optim.Adam(model.parameters())

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=traindata,
        cuda_device=-1,
        num_epochs=args.num_epochs,
    )

    metrics = trainer.train()
    print(metrics)


if __name__=='__main__':
    main()