import torch
from torchtext.data import BucketIterator


class BatchWrapper:
    """Convenience wrapper for dataloaders."""
    def __init__(self, dataloader, data="text", label="labels"):
        self.dataloader = dataloader
        self.data = data
        self.label = label
    
    def __iter__(self):
        for batch in self.dataloader:
            x = getattr(batch, self.data)
            
            if self.label is not None:
                y = getattr(batch, self.label)
            else:
                y = torch.zeros((1))

            yield (x, y)
    
    def __len__(self):
        return len(self.dataloader)


class BucketLoader:

    def __init__(self, train, val, batch_sizes=(64,64), device='cpu'):
        self.train = train
        self.val = val
        self.batch_sizes = batch_sizes
        self.device = device

    def load_iterators(self, sort_within_batch=False, repeat=False):
        train_iter, val_iter = BucketIterator.splits(
            (self.train, self.val),
            batch_sizes = self.batch_sizes,
            device = self.device,
            sort_key = lambda x: len(x.text),
            sort_within_batch = sort_within_batch,
            repeat = repeat 
        )
        return train_iter, val_iter