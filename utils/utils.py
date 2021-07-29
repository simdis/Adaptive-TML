import os
import numpy as np

from typing import Dict, List, Optional, Tuple


def find_classes(directory: str, class_names: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, int]]:
    """
    Find the classes within the given directory.
    Each class is assumed to be a folder within the directory parameter.
    Optionally a subset of directories can be considered by passing the desired
    list of classes.
    :param directory: a string containing the path to the directory containing the dataset.
    :param class_names: an optional list of classes to be selected within the directory.
    :return: a tuple of two elements. The first is the list of classes, the second a dict
        mapping each class to the corresponding index (or label).
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    if class_names is None:
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    else:
        class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names) if cls_name in classes}
        # Check the validity of provided classes
        if len(class_to_idx) < len(class_names):
            raise RuntimeError(f"The directory {directory} does not contain all the classes ({class_names})")
    return classes, class_to_idx


def random_subsample_classes(
        directory: str, num_classes: int = 2, min_samples: int = 1,
        classes_to_skip: Optional[List[str]] = None,
        random_generator: Optional[np.random.Generator] = None
) -> List[str]:
    """
    Subsample the given number of classes within a dataset organized into folders, each of them
    containing the samples of a single class.
    :param directory: the directory containing the dataset.
    :param num_classes: the number of classes to sample.
    :param min_samples: the minimum number of samples to consider a class.
    :param classes_to_skip: an optional list of classes to skip.
    :param random_generator: the numpy generator, if one wants to pass it.
    :return: a list containing up to num_classes classes.
    """
    assert num_classes > 0, "The number of classes must be strictly positive."
    assert min_samples > 0, "The minimum number of samples must be strictly positive."
    if classes_to_skip is None:
        classes_to_skip = list()
    # Define the classes as the directories satisfying all the conditions.
    classes = [d.name for d in os.scandir(directory)
               if d.is_dir() and d.name not in classes_to_skip
               and len([f for f in os.scandir(d.path) if f.is_file()]) >= min_samples]
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    if random_generator:
        idxs = random_generator.choice(len(classes), size=num_classes, replace=False)
    else:
        idxs = np.random.choice(len(classes), size=num_classes, replace=False)
    # Filter the output
    classes = np.array(classes)[idxs]
    if num_classes == 1:
        return list(classes)
    return classes
