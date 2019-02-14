import os
import torch
from time import strftime


def save_checkpoint(model, optimizer, epoch, path, accuracy=None, filename=None):
    """Save state"""
    if filename is None:
        date = strftime("%Y_%m_%d")
        filename = "save" + date

    outpath = os.path.join(path, filename)

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'accuracy': accuracy,
        }, 
        outpath 
    )