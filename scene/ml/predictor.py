import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from allennlp.nn import util as nn_util
from allennlp.data.iterators import DataIterator


def to_numpy(x): return x.detach().cpu().numpy()


class Predictor:

    def __init__(self, model, iterator, device="cpu"):
        self.model = model
        self.iterator = iterator
        self.device = device

    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch)
        arry = to_numpy(out_dict["logits"])
        out = F.softmax(arry, dim=1)
        return out

    def predict(self, data):
        pred_generator = self.iterator(data, num_epochs=1, shuffle=False)
        self.model.eval()

        pred_generator_tqdm = tqdm(
            pred_generator,
            total=self.iterator.get_num_batches(data)
        )

        ids = []
        logits = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                out = self.model(batch["tokens"], batch["id"])
                logits.append(out["logits"])
                ids.append(torch.tensor(out["id"]))

        ids = torch.cat(ids, dim=0)
        logits = torch.cat(logits, dim=0)
        output = {"id": ids, "logits": logits}

        return output
