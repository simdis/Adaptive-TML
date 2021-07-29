import numpy as np
import torch

from typing import List, Union


class FilterSelectionLayer(torch.nn.Module):
    """
    This layer implements a fixed filter selection.
    Given the usual (batch_size, num_filters, h, w) batch of data,
    it keeps only the f filters specified in the __init__, i.e.,
    providing an output of shape (batch_size, f, h, w).

    :filters_to_keep: an array containing the filters to keep.
    There is no check on the validity of filters, i.e., the forward
    will fail if out-of-the-boundary indices are specified.
    """
    def __init__(self, filters_to_keep: Union[np.ndarray, List[int]]) -> None:
        super(FilterSelectionLayer, self).__init__()
        self.filters_to_keep = torch.Tensor(filters_to_keep)
        self.filters_to_keep = self.filters_to_keep.type(torch.IntTensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.filters_to_keep]


class IdentityLayer(torch.nn.Module):
    """
    Class IdentityLayer.
    This custom layer implements a simple identity.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
