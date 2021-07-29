import torch.utils.data
import random

from typing import Iterable, Optional, Sized


class SequentialSamplerWithOneShuffle(torch.utils.data.Sampler):
    """
    Custom Sampler that defines a shuffle sequence of the data and
    continuously provides the data in the same order.
    """
    def __init__(self, data_source: Sized, splits: Optional[Iterable[int]] = None):
        """
        Init method of SequentialSamplerWithOneShuffle.
        :param data_source: the dataset to sample from.
        :param splits: a sequence of integer defining the groups within the dataset to be shuffled.
            For instance, if splits[50, 100], the shuffle considers defines a shuffle of the first
            50 elements, then a shuffle of the elements between 50 and 99, and so on.
            Please note that when a value for splits is passed, no check on the dataset size is done,
            i.e., you have to ensure that the last value in split corresponds to the dataset size
            (and not overcome it).
        """
        super(SequentialSamplerWithOneShuffle, self).__init__(data_source=data_source)
        self.data_source = data_source
        if splits is None:
            self.sequence = [x for x in range(len(self.data_source))]
            # Shuffle the sequence
            random.shuffle(self.sequence)
        else:
            self.sequence = list()
            prev_ = 0
            for next_ in splits:
                new_seq = [x for x in range(prev_, next_)]
                random.shuffle(new_seq)
                self.sequence += new_seq
                prev_ = next_

    def __iter__(self):
        return iter(self.sequence)

    def __len__(self) -> int:
        return len(self.data_source)
