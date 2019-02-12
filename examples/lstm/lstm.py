import tqdm
from parser import parse_args

import torch.nn as nn
import torch.optim as optim

from scene.data import DataSet
from torchtext.data import Iterator
from scene.data.loaders import BatchWrapper, BucketLoader

from scene.models import BiLSTM


def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for data, labels in tqdm.tqdm(loader):
        optimizer.zero_grad()
        preds = model(data)
        loss = loss_func(preds, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0] * data.size(0)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    data = DataSet(args.datapath)
    train, val, test = data.load_splits()
    data.textfield.build_vocab(train, vectors='glove.6B.100d')
    vocab = data.textfield.vocab

    train_iter, val_iter = BucketLoader(
        train, 
        val, 
        batch_sizes=(args.batch_size, args.batch_size),
        device=device
    )

    test_iter = Iterator(
        test, 
        batch_size=args.batch_size, 
        device=device, 
        sort=False, 
        sort_within_batch=False, 
        repeat=False
    )

    trainloader = BatchWrapper(train_iter)
    valloader = BatchWrapper(val_iter)
    testloader = BatchWrapper(test_iter)

    model = BiLSTM(num_vocab=len(vocab), n_classes=9)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    for _ in range(1, args.epochs+1):
        train(model, trainloader, criterion, optimizer)
        val(model, valloader, criterion)