import torch
import numpy as np
from tqdm import tqdm

from allennlp.nn import util as nn_util
from allennlp.data.iterators import DataIterator


def to_numpy(x): return x.detach().cpu().numpy()


class Predictor:

    def __init__(self, model, iterator, device):
        self.model = model
        self.iterator = iterator
        self.device = device
        
    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch)
        arry = to_numpy(out_dict["logits"])
        out = torch.sigmoid(arry)
        return out
    
    def predict(self, data):
        pred_generator = self.iterator(data, num_epochs=1, shuffle=False)
        self.model.eval()

        pred_generator_tqdm = tqdm(
            pred_generator,
            total=self.iterator.get_num_batches(data)
        )

        preds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.device)
                preds.append(self._extract_data(batch))

        return np.concatenate(preds, axis=0)