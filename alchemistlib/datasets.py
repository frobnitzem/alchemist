""" Data sources suitable for generative molecular
    training.
"""

import torch

class MultinomialData(torch.utils.data.IterableDataset):
    def __init__(self, prob, batch_size=16):
        super().__init__()
        self.prob = prob
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            z = torch.multinomial(self.prob,
                                  num_samples=self.batch_size,
                                  replacement=True)
            yield z

