import torch
import torchvision.datasets.folder

from utils import utils

from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class ImageFolderWithClassSelection(torchvision.datasets.folder.ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            get_filenames: bool = False,
            class_names: Optional[List[str]] = None
    ) -> None:
        """
        Init method of ImageFolderWithClassSelection.
        :param root: the path to the dataset.
        :param transform: a function containing the transformation of the input.
        :param target_transform: a function containing the transformation of the output label.
        :param is_valid_file: a function that checks the validity of a file given its path.
            It should not be used if extensions is provided.
        :param get_filenames: a boolean flag. If true, the filename is provided along with the input and its label.
        :param class_names: the list of classes to be kept within the dataset folder.
        """
        self.class_names = class_names
        self.get_filenames = get_filenames
        super(ImageFolderWithClassSelection, self).__init__(
            root=root, loader=torchvision.datasets.folder.default_loader,
            transform=transform, target_transform=target_transform,
            is_valid_file=is_valid_file
        )

    def __getitem__(self, index: int) -> Union[Tuple[Any, Any], Tuple[Any, Any, str]]:
        """
        Overrides get_item method.
        :param index: Index of sample.
        :return: a Tuple containing the sample, its label, and, if any, the filename.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.get_filenames:
            return sample, target, path
        return sample, target

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Overrides the find_classes method to keep into account the class_names parameter.
        :param directory: the path to the dataset directory.
        :return: see find_classes of DatasetFolder class in PyTorch.
        """
        return utils.find_classes(directory=directory, class_names=self.class_names)
