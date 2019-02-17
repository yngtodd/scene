import os
import tqdm 

import torch
import numpy as np
from parser import parse_args

from scene.data import DataSet
from torchtext.data import Iterator
from scene.data.loaders import BatchWrapper

from scene.models import BiLSTM


def predict(model, loader):
    model.eval()
    predictions = []
    for data in tqdm.tqdm(loader):
        pred = model(data)
        _, pred = torch.max(pred.data, 1)
        predictions.append(pred)
    return np.array(predictions)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    data = DataSet(args.datapath)
    train_data, val_data, test_data = data.load_splits()
    vocab = data.textfield.vocab

    test_iter = Iterator(
        test_data,
        batch_size=1,
        device=device,
        sort=False,
        sort_within_batch=False,
        repeat=False
    )

    testloader = BatchWrapper(test_iter)

    savepath = os.path.join(args.savepath, 'bilstm_small_val.pth')
    savepoint = torch.load(savepath)

    model = BiLSTM(num_vocab=len(vocab), n_classes=10).to(device)
    model.load_state_dict(savepoint['model_state_dict'])

    predictions = predict(model, test_iter)
    outpath = os.path.join(args.savepath, 'test_preds.npy')
    np.save(outpath, predictions) 


if __name__=='__main__':
    main()