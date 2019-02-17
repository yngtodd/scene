import tqdm
from parser import parse_args

import torch
import torch.nn as nn
import torch.optim as optim

from scene.data import DataSet
from torchtext.data import Iterator
from scene.data.loaders import BatchWrapper, BucketLoader

from scene.models import BiLSTM
from scene.models.save import save_checkpoint


def train(model, loader, criterion, optimizer, epoch, args):
    model.train()
    total = 0
    correct = 0
    iteration = 0
    running_loss = 0.0
    for data, labels in tqdm.tqdm(loader):
        # torchtext one-indexes the labels, breaking crossentropy
        label = labels - 1
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(pred.data, 1)
        total += labels.size(0)
        correct += (predicted == label).sum().item()
        running_loss += loss.item() * data.size(0)
        iteration += 1

    epoch_loss = running_loss / total
    accuracy = 100 * correct / total
    print(f'Epoch: {epoch}, Loss: {epoch_loss:.6}, Accuracy: {accuracy:.5}')

    if epoch % 10 == 0:
        save_chekpoint(model, optimizer, epoch, args.savepath, accuracy)


def validate(model, loader, criterion, epoch):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    for data, labels in tqdm.tqdm(loader):
        # torchtext one-indexes the labels, breaking crossentropy
        label = labels - 1
        pred = model(data)
        loss = criterion(pred, label)
        _, predicted = torch.max(pred.data, 1)
        total += labels.size(0)
        correct += (predicted == label).sum().item()
        running_loss += loss.item() * data.size(0)

    epoch_loss = running_loss / total
    accuracy = 100 * correct / total
    print(f'Epoch: {epoch}, Loss: {epoch_loss:.6}, Accuracy: {accuracy:.5}')


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    data = DataSet(args.datapath)
    train_data, val_data, test_data = data.load_splits()
    vocab = data.textfield.vocab

    bucket = BucketLoader(
        train_data,
        val_data,
        batch_sizes=(args.batch_size, args.batch_size),
        device=device
    )

    train_iter, val_iter = bucket.load_iterators()

    test_iter = Iterator(
        test_data,
        batch_size=args.batch_size,
        device=device,
        sort=False,
        sort_within_batch=False,
        repeat=False
    )

    trainloader = BatchWrapper(train_iter)
    valloader = BatchWrapper(val_iter)
    testloader = BatchWrapper(test_iter)

    model = BiLSTM(num_vocab=len(vocab), n_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs+1):
        train(model, trainloader, criterion, optimizer, epoch)
        validate(model, valloader, criterion, epoch)


if __name__=='__main__':
    main()
