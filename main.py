import copy
import random
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import sklearn.neighbors as neighbors
import sklearn.svm as svm
import torch
import torch.utils.data
import torchvision

import argparse
import os
import time

from tqdm import tqdm

import synthetic_dataset.synthetic_dataset
import synthetic_dataset.rotating_hyperplane_dataset
from audio_utils import transforms as audio_transforms
from audio_utils import spectrogram_dataloader_pytorch as audio_datasets
from audio_utils import loaders as audio_loaders
from pytorch_extensions import dataset as image_datasets
from pytorch_extensions import layers as custom_pytorch_layers
from pytorch_extensions import sampler
from pytorch_extensions import torch_utils
from utils import utils

from tinyknn import active_tiny_knn, hybrid_tiny_knn, incremental_knn, condensed_nearest_neighbor, condensing_in_time, \
    active_cdt_functions

import river

from typing import Any, Callable, Dict, Iterator, List, Optional, Union, Tuple


def define_and_parse_flags(parse: bool = True) -> Union[argparse.ArgumentParser, argparse.Namespace]:
    """
    Define the FLAGS of this script.
    :param parse: whether to parse or not the defined flags.
    :return: the parser or the parsed namespace.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=int(np.random.rand() * 100))
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--dr_policy', default="none",
                        choices=["none", "filter_selection", "filter_selection_class_distance",
                                 "class_distance", "filter_selection_plus",
                                 "filter_selection_class_distance_plus"],
                        help="Dimensionality reduction policy. The possible values are"
                             "'none' (default when a non admissible value is provided), "
                             "'filter_selection', 'filter_selection_class_distance', and "
                             "'class_distance'. 'filter_selection_plus', and "
                             "'filter_selection_class_distance_plus' are equivalent "
                             "to their base version, but with the base_cnn modified "
                             "accordingly to their action. Filter selection requires a "
                             "further array specifying the filter indices to keep named filters_to_keep."
                             " Class distance is automatically learned by the model based on "
                             " the training features. It keeps the class_distance_filters that maximized "
                             "the mean distance among the classes."
                        )
    parser.add_argument('--filters_to_keep', type=str, default="1,2,3,4,5",
                        help="Comma separated number of filters to keep in filter selection mode(s).")
    parser.add_argument('--class_distance_filters', type=int, default=2,
                        help="Number of filters to keep in class_distance mode(s).")
    parser.add_argument('--cnn_fe', default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                        help="The CNN used as feature extractor.")
    parser.add_argument('--cnn_fe_weights_path', type=str, default="",
                        help="The path to the pth file containing the weights of the CNN FE. **deprecated**")
    parser.add_argument('--feature_layer', type=str, default='maxpool',
                        help="Convolutional Layer at which the features are extracted.")

    parser.add_argument('--data_dir', type=str, required=True,
                        help="The path to the dataset")
    parser.add_argument('--second_data_dir', type=str, default=None,
                        help="The (optional) path to a second dataset. If provided, is used after the change."
                             " It is assumed that the classes are the same in both the datasets.")

    parser.add_argument('--output_dir', type=str, default='./output/')

    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--is_audio', action='store_true', help="Boolean flag that switches among image and audio.")
    parser.add_argument('--audio_seconds', type=int, default=1, help="Length of audio waves.")
    parser.add_argument('--sample_rate', type=int, default=22050)
    parser.add_argument('--n_fft', type=int, default=512)
    parser.add_argument('--hop_length', type=int, default=-1)
    parser.add_argument('--top_db', type=int, default=80,
                        help="Cut-off of decibels (default and suggested value is 80db).")
    parser.add_argument('--is_synthetic', action='store_true',
                        help="Boolean flag that forces to use a synthetic dataset "
                             "(the path to dataset in this case is ignored).")
    parser.add_argument('--do_rotating', action='store_true',
                        help="When is_synthetic is provided, use the rotating hyperplane dataset.")
    parser.add_argument('--grid_size', type=int, default=7,
                        help="The size of the synthetic dataset squares.")
    parser.add_argument('--concept_drift_magnitude_mean', type=str, default="0",
                        help="The comma separated magnitude of change per each class mean or a single value for all.")
    parser.add_argument('--concept_drift_magnitude_cov', type=str, default="0",
                        help="The comma separated magnitude of change per each class cov or a single value for all.")
    parser.add_argument('--concept_drift_time', type=int, default=1,
                        help="The number of steps in which the concept drift occurs (1=abrupt, >1=gradual).")
    parser.add_argument('--synthetic_classes_mean_scale', type=str, default="1.0",
                        help="The comma separated width of possible mean values for each class.")
    parser.add_argument('--synthetic_classes_mean_min', type=str, default="0.0",
                        help="The comma separated minimum possible mean values for each class.")
    parser.add_argument('--synthetic_classes_cov_scale', type=str, default="0.5",
                        help="The comma separated width of possible cov values for each class.")
    parser.add_argument('--synthetic_classes_cov_min', type=str, default="0.1",
                        help="The comma separated minimum possible cov values for each class.")

    parser.add_argument('--do_incremental', action='store_true', help="Activate the incremental experiments.")
    parser.add_argument('--do_passive', action='store_true', help="Activate passive experiments.")
    parser.add_argument('--do_active', action='store_true', help="Activate active experiments.")
    parser.add_argument('--skip_base_exps', action='store_true', help="Skip the experiments with SVM and NN.")
    parser.add_argument('--skip_nn_exps', action='store_true', help="Skip the experiments with NN.")

    parser.add_argument('--do_knn_adwin', action='store_true', help="Test against the kNN+ADWIN method.")
    parser.add_argument('--do_knn_adwin_paw', action='store_true',
                        help="Test against the kNN+ADWIN method with the PAW.")
    parser.add_argument('--do_sam_knn', action='store_true', help="Test against the SAM-kNN method.")
    parser.add_argument('--do_soa_without_dl', action='store_true',
                        help="Apply the State of the Art algorithms without the feature extractor and the "
                             "dimensionality reduction operator.")

    parser.add_argument('--incremental_step', type=int, default=1,
                        help="The number of samples to be added at each incremental step.")
    parser.add_argument('--cit_max_samples', type=int, default=10000,
                        help="The memory bound of CIT algorithm (measured as number of samples).")
    parser.add_argument('--window_length', type=int,
                        help="The size of the history window in Active Tiny kNN or the training window "
                             "in Hybrid Tiny kNN (default: --cit_max_samples).")
    parser.add_argument('--samples_per_class_to_test', type=str, default="1,2,3,4,5,10,20,30,40,50,60,70,80,90,100",
                        help="The comma separated list of initial training set sizes to be tested. Each value "
                             "corresponds to the number of samples per each class.")
    parser.add_argument('--base_test_samples', type=int, default=500, help="Number of samples per class in testing.")
    parser.add_argument('--n_binomial', type=int, default=50,
                        help="The value of N when using binomial distribution in CUSUM CDT.")

    parser.add_argument('--nn_lr_base', type=str, default="1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5",
                        help="The comma separated learning rates to test in NN-base classifier.")
    parser.add_argument('--nn_lr_incremental', type=str, default="1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6",
                        help="The comma separated learning rates to test in NN-base classifier "
                             "during incremental updates.")
    parser.add_argument('--nn_weights_alpha', type=float, default=1e-5,
                        help="The weights L2 regularization coefficient in Adam Optimizer.")
    parser.add_argument('--nn_training_epochs', type=int, default=3,
                        help="The number of epochs the NN classifiers are trained for.")
    parser.add_argument('--nn_batch_size', type=int, default=5,
                        help="The batch size in NN classifiers training.")

    parser.add_argument('--num_readers', type=int, default=8)

    parser.add_argument('--classes_to_change', type=int, default=0,
                        help="Number of classes to change. Must be <= the number of classes.")
    parser.add_argument('--classes_to_add', type=int, default=0,
                        help="Number of classes to add after a change.")

    # Return the parser or the parsed values according to the parameter 'parse'.
    if parse:
        args = parser.parse_args()
        # Fix default values based on other arguments.
        args.window_length = args.window_length if args.window_length else args.cit_max_samples
        return args
    return parser


def generate_classes(
        flags: argparse.Namespace
) -> Tuple[List[str], List[str]]:
    """
    Part of code that generates the classes.
    :param flags: the namespace of argparse with the parameters.
    :return: a tuple containing the two lists of classes before and after the change.
    If there is no change, the second list corresponds to the first one.
    """
    # Get num_classes classes.
    all_classes = utils.random_subsample_classes(
        directory=flags.data_dir,
        num_classes=flags.num_classes + flags.classes_to_change + flags.classes_to_add
    )
    if flags.classes_to_change or flags.classes_to_add:
        classes_before_change = [x for x in all_classes[:flags.num_classes]]
        classes_after_change = [x for x in all_classes[:flags.num_classes]]
        for ii in range(flags.classes_to_change):
            classes_after_change[ii] = all_classes[flags.num_classes + ii]
        for ii in range(flags.classes_to_add):
            classes_after_change.append(all_classes[flags.num_classes + flags.classes_to_change + ii])
    else:
        classes_before_change = all_classes
        classes_after_change = all_classes

    print('The class(es) before the change is/are {}, and {}.'.format(
        ', '.join(classes_before_change[:-1]), classes_before_change[-1])
    )
    print('The class(es) after the change is/are {}, and {}.'.format(
        ', '.join(classes_after_change[:-1]), classes_after_change[-1])
    )
    return classes_before_change, classes_before_change


def generate_dataloader(
        flags: argparse.Namespace,
        classes_before_change: List[str],
        classes_after_change: List[str]
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.Sampler, int, List[int]]:
    """
    The part of code that deals with the creation of the dataloader
    :param flags: the namespace of argparse with the parameters.
    :param classes_before_change:
    :param classes_after_change:
    :return: a Tuple with the dataloader, the sampler, the number of images, and the size of the datasets.
    """
    # Define the dataloader.
    # dataset = None
    splits_length = None
    synthetic_dataset_size = \
        (flags.base_test_samples + int(max(flags.samples_per_class_to_test.split(',')))) * flags.num_classes * 2
    if flags.is_synthetic and flags.do_rotating:
        print("Creating Rotating Hyperplane Dataset Instance.")
        dataset = synthetic_dataset.rotating_hyperplane_dataset.RotatingHyperplaneGridDataset(
            grid_size=flags.grid_size,
            mag_change=0.001,  # Fixed according to related literature
            noise_percentage=0.01,  # No reference in related literature
            sigma_percentage=0.01,  # No reference in related literature
            dataset_size=synthetic_dataset_size,
            seed=flags.seed * 20,
            transform=None
        )
    elif flags.is_synthetic:
        print("Creating Synthetic Dataset Instance.")
        class_mean_scale = np.array(flags.synthetic_classes_mean_scale.split(','), dtype=np.float32)
        class_mean_min = np.array(flags.synthetic_classes_mean_min.split(','), dtype=np.float32)
        class_cov_scale = np.array(flags.synthetic_classes_cov_scale.split(','), dtype=np.float32)
        class_cov_min = np.array(flags.synthetic_classes_cov_min.split(','), dtype=np.float32)
        if len(class_mean_scale) == 1:
            class_mean_scale = float(class_mean_scale)
        if len(class_mean_min) == 1:
            class_mean_min = float(class_mean_min)
        if len(class_cov_scale) == 1:
            class_cov_scale = float(class_cov_scale)
        if len(class_cov_min) == 1:
            class_cov_min = float(class_cov_min)
        mean_change_magnitude = np.array(flags.concept_drift_magnitude_mean.split(','), dtype=np.float32)
        cov_change_magnitude = np.array(flags.concept_drift_magnitude_cov.split(','), dtype=np.float32)
        if len(mean_change_magnitude) == 1:
            mean_change_magnitude = float(mean_change_magnitude[0])
        elif len(mean_change_magnitude) == flags.num_classes:
            mean_change_magnitude = np.array(mean_change_magnitude).reshape((flags.num_classes,))
        else:
            mean_change_magnitude = np.array(mean_change_magnitude).reshape(
                (flags.grid_size, flags.grid_size, flags.num_classes)
            )
        if len(cov_change_magnitude) == 1:
            cov_change_magnitude = float(cov_change_magnitude[0])
        elif len(cov_change_magnitude) == flags.num_classes:
            cov_change_magnitude = np.array(cov_change_magnitude).reshape((flags.num_classes,))
        else:
            cov_change_magnitude = np.array(mean_change_magnitude).reshape(
                (flags.grid_size * flags.grid_size, flags.grid_size * flags.grid_size, flags.num_classes)
            )
        dataset = synthetic_dataset.synthetic_dataset.SyntheticMultivariateNormalGridDataset(
            grid_size=flags.grid_size,
            num_classes=flags.num_classes,
            dataset_size=synthetic_dataset_size,
            mean_change_magnitude=mean_change_magnitude,
            mean_change_duration=flags.concept_drift_time,
            cov_change_magnitude=cov_change_magnitude,
            cov_change_duration=flags.concept_drift_time,
            change_beginning=synthetic_dataset_size // 2,
            mean_scale=class_mean_scale,
            mean_min=class_mean_min,
            cov_scale=class_cov_scale,
            cov_min=class_cov_min,
            seed=flags.seed * 20,
            transform=None
        )
    elif flags.is_audio:
        print("Creating Speech Command Dataset Instance.")
        # Create the audio transform
        audio_transform = audio_transforms.spectrogram_transforms(
            n_fft=flags.n_fft,
            hop_length=flags.hop_length if flags.hop_length > 0 else flags.n_fft,
            top_db=flags.top_db
        )
        dataset = audio_datasets.SpectrogramFolder(
            root=flags.data_dir,
            loader=audio_loaders.spectrogram_loader_librosa,
            loader_kwargs={
                "sample_rate": flags.sample_rate,
                "max_seconds": flags.audio_seconds,
            },
            extensions=(".wav", ".mp3"),
            transform=audio_transform,
            class_names=classes_before_change
        )
        if flags.second_data_dir:
            # Create the second dataset and join both the datasets.
            second_dataset = audio_datasets.SpectrogramFolder(
                root=flags.second_data_dir,
                loader=audio_loaders.spectrogram_loader_librosa,
                loader_kwargs={
                    "sample_rate": flags.sample_rate,
                    "max_seconds": flags.audio_seconds,
                },
                extensions=(".wav", ".mp3"),
                transform=audio_transform,
                class_names=classes_after_change
            )
            splits_length = [len(dataset), len(dataset) + len(second_dataset)]
            dataset = torch.utils.data.ConcatDataset(
                [dataset, second_dataset]
            )
            # dataset.__add__(second_dataset)
    else:
        print("Creating ImageNet Dataset Instance.")
        # Image case
        image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(flags.image_size),
            torchvision.transforms.RandomCrop(flags.image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])
        dataset = image_datasets.ImageFolderWithClassSelection(
            root=flags.data_dir,
            transform=image_transform,
            class_names=classes_before_change
        )
        if flags.second_data_dir:
            # Create the second dataset and join both the datasets.
            second_dataset = image_datasets.ImageFolderWithClassSelection(
                root=flags.data_dir,
                transform=image_transform,
                class_names=classes_after_change
            )
            splits_length = [len(dataset), len(dataset) + len(second_dataset)]
            dataset = torch.utils.data.ConcatDataset(
                [dataset, second_dataset]
            )
            # dataset.__add__(second_dataset)

    # Create the batch sampler
    batch_sampler = sampler.SequentialSamplerWithOneShuffle(
        data_source=dataset,
        splits=splits_length
    )
    # Create the data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        # batch_size=1,  # The test dataloader has a fixed batch size of 1
        # shuffle=False,  # To avoid re-shuffling at every "epoch"
        sampler=batch_sampler,
        num_workers=flags.num_readers
    )
    print(f"The number of samples within the dataset is {len(dataset)}")
    return dataloader, batch_sampler, len(dataset), splits_length if splits_length else [len(dataset)]


def identity_dimred_(x: np.ndarray):
    """
    Default dimensionality reduction operator.
    It simply forwards the input.
    :param x: the input numpy array.
    :return: the input without any change.
    """
    return x


def train_class_distance_dimred(
        train_features: np.ndarray, train_labels: np.ndarray,
        num_classes: int, features_shape: Union[np.ndarray, torch.Tensor], num_filters: int
) -> Tuple[Callable, np.ndarray, np.ndarray, np.ndarray]:
    """
    Function that trains the class distance dimensionality reduction operator.
    :param train_features: a numpy array containing the training features.
        The shape is (num_training_samples, *features_shape).
    :param train_labels: a numpy array containing the training labels.
        The shape is (num_training_samples, ).
    :param num_classes: the number of classes in the considered problem.
    :param features_shape: the shape of the training features.
        The feature shape should be (num_filters, width, height).
    :param num_filters: the number of filters to keep.
    :return: a tuple containing: the callable dimensionality reduction operator, the reduced
        training features and labels, the indices of the kept filters.
    """
    mean_image = np.zeros((num_classes, *features_shape))
    for cc in range(num_classes):
        mean_image[cc] = np.mean(
            train_features[train_labels == cc], axis=0
        )
    # Get the filter characterized the by the highest distance among classes
    distances = metrics.pairwise_distances(mean_image.reshape(num_classes, -1))
    sum_of_distances = np.sum(distances, axis=0)
    high_class_distance_filter = np.argsort(-1 * sum_of_distances)[:num_filters]
    high_class_distance_filter = high_class_distance_filter.astype(np.int32)

    # Define the dimred_ function
    def dimred_(x):
        # Reshape the flattened features
        x = x.reshape(features_shape)
        # Extract the filters
        x = x[high_class_distance_filter]
        # Flatten again and return
        return x.reshape((1, -1))

    # Reduce the training features
    bs = train_features.shape[0]
    train_features = train_features.reshape((-1, *features_shape))[:, high_class_distance_filter]
    train_features = train_features.reshape((bs, -1))

    return dimred_, train_features, train_labels, high_class_distance_filter


def learn_dimred(
        dr_args: Dict[str, Any],
        train_features: np.ndarray, train_labels: np.ndarray,
        num_classes: int, features_shape: Union[np.ndarray, torch.Tensor]
) -> Tuple[Callable, np.ndarray, np.ndarray, np.ndarray]:
    """
    Function that defines the class distance dimensionality reduction operator learning.
    It checks the validity of the parameters and starts the learning.
    :param dr_args: the dimensionality reduction arguments. This is a dictionary containing
        the type of dimensionality reduction and its parameters.
    :param train_features: a numpy array containing the training features.
        The shape is (num_training_samples, *features_shape).
    :param train_labels: a numpy array containing the training labels.
        The shape is (num_training_samples, ).
    :param num_classes: the number of classes in the considered problem.
    :param features_shape: the shape of the training features.
        The feature shape should be (num_filters, width, height).
    :return: a tuple containing: the callable dimensionality reduction operator, the reduced
        training features and labels, the indices of the kept filters.
    """
    if dr_args['type'] == "class_distance":
        return train_class_distance_dimred(
            train_features=train_features, train_labels=train_labels,
            num_classes=num_classes, features_shape=features_shape,
            num_filters=dr_args['filters']
        )
    else:
        raise ValueError(f"Invalid dr_args fields. Accepted dr_args['type'] are [class_distance].")


def compute_training_features(
        num_training_samples: int, num_classes: int, features_size: int,
        iterator: Iterator, cnn_forward_fn: Callable, cnn_forward_args: Optional[Tuple[Any, ...]] = None,
        _verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Function that extracts the features of the initial training set.
    :param num_training_samples: the number of samples per class to be computed.
    :param num_classes: the number of classes.
    :param features_size: the size of the features.
    :param iterator: the dataloader iterator. It should provides at each iteration a tuple
        containing the sample and its label.
    :param cnn_forward_fn: the feature extractor forward function.
    :param cnn_forward_args: the feature extractor forward function arguments.
    :param _verbose: whether print verbose messages or not.
    :return: a tuple containing the arrays of training features and labels, plus
        an integer that contains the number of iterations done.
    """
    training_features = np.zeros((num_training_samples * num_classes, features_size))
    training_labels = np.zeros(num_training_samples * num_classes)

    if cnn_forward_args is None:
        cnn_forward_args = tuple()  # Empty tuple

    # Get the training samples from test dataloader, respecting the proportions among classes.
    effective_num_of_samples = 0
    ii = 0
    class_counter = np.ones(num_classes) * num_training_samples
    tot_images = np.sum(class_counter)
    while tot_images > 0:
        # Get the image and its features
        im_, lb_ = next(iterator)
        if _verbose:
            print('Iteration {} -- Label {} -- Class Counter {}'.format(
                effective_num_of_samples, lb_, class_counter))
        # Save features and label
        if class_counter[lb_]:
            ft_ = torch.flatten(cnn_forward_fn(im_, *cnn_forward_args))
            training_features[ii] = ft_.data.numpy()
            training_labels[ii] = lb_.data.numpy()
            ii += 1
            class_counter[lb_] -= 1
            tot_images -= 1
            if _verbose:
                print('Saving! New class counter {}'.format(class_counter))
        # Count the effective number of samples
        effective_num_of_samples += 1

    return training_features, training_labels, effective_num_of_samples


def compute_training_features_nodl(
        num_training_samples: int, num_classes: int, features_size: int,
        iterator: Iterator, _verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Function that extracts the features of the initial training set without the deep learning part.
    :param num_training_samples: the number of samples per class to be computed.
    :param num_classes: the number of classes.
    :param features_size: the size of the features.
    :param iterator: the dataloader iterator. It should provides at each iteration a tuple
        containing the sample and its label.
    :param _verbose: whether print verbose messages or not.
    :return: a tuple containing the arrays of training features and labels, plus
        an integer that contains the number of iterations done.
    """
    training_features = np.zeros((num_training_samples * num_classes, features_size))
    training_labels = np.zeros(num_training_samples * num_classes)

    # Get the training samples from test dataloader, respecting the proportions among classes.
    effective_num_of_samples = 0
    ii = 0
    class_counter = np.ones(num_classes) * num_training_samples
    tot_images = np.sum(class_counter)
    while tot_images > 0:
        # Get the image and its features
        im_, lb_ = next(iterator)
        if _verbose:
            print('Iteration {} -- Label {} -- Class Counter {}'.format(
                effective_num_of_samples, lb_, class_counter))
        # Save features and label
        if class_counter[lb_]:
            ft_ = torch.flatten(im_)
            training_features[ii] = ft_.data.numpy()
            training_labels[ii] = lb_.data.numpy()
            ii += 1
            class_counter[lb_] -= 1
            tot_images -= 1
            if _verbose:
                print('Saving! New class counter {}'.format(class_counter))
        # Count the effective number of samples
        effective_num_of_samples += 1

    return training_features, training_labels, effective_num_of_samples


def extract_training_features(
        training_features: np.ndarray, training_labels: np.ndarray, num_samples_per_class: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function that provides among already extracted features the given number of samples per class.
    :param training_features: the training features of shape (num_samples, features_size).
    :param training_labels: the training labels of shape (num_samples, ).
    :param num_samples_per_class: the number of samples
    :return: the first samples of each class (and their labels) up to the desired number.
    """
    assert num_samples_per_class * np.size(np.unique(training_labels)) <= np.size(training_labels), \
        "The number of samples per class multiplied by the number of classes should be " \
        "smaller or equal the number of samples"
    # Compute the indices to keep
    idx_to_keep = [np.where(training_labels == c)[0] for c in np.unique(training_labels)]
    # Keep the first num_samples_per_class in each array
    idx_to_keep = [a[:num_samples_per_class] for a in idx_to_keep]
    # Join all the arrays together
    idx_to_keep = np.sort(np.concatenate(idx_to_keep))
    return training_features[idx_to_keep], training_labels[idx_to_keep]


def extract_next_data(
        test_iterator: Iterator, time_results_dict: dict, base_cnn: Callable, dimred_: Callable, current_idx: int,
        apply_deep_learning: bool = True
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    An utility function that extracts the features in input to the kNN-based classifier and the corresponding label
    for the subsequent and optional testing/adaptation.
    :param test_iterator: the dataset iterator.
    :param time_results_dict: the dictionary where to store the extraction time statistics at the keys "dl" and "fe_dr".
        It is assumed that at those keys there is an array of shape (num_tested_cases, ).
    :param base_cnn: the feature extractor PyTorch module.
    :param dimred_: the dimensionality reduction callable.
    :param current_idx: the index at where store the time statistics within the time_results_dict.
    :param apply_deep_learning: whether to apply the feature extractor and the dimensionality reduction on new data.
    :return: a Tuple containing the features and the label as properly shaped numpy arrays.
    """
    # Get the sample
    st_time = time.time()
    im_, lb_ = next(test_iterator)
    time_results_dict["dl"][current_idx] += time.time() - st_time
    # Compute feature extractor + dimensionality reduction
    if apply_deep_learning:
        st_time = time.time()
        ft_ = torch.flatten(base_cnn(im_))  # FE with flattening.
        ft_ = dimred_(ft_.data.numpy().reshape(1, -1))  # DR
        time_results_dict["fe_dr"][current_idx] += time.time() - st_time
    else:
        # The output is simply the input.
        ft_ = im_.data.numpy()

    return ft_, lb_


def generate_cnn(
        flags: argparse.Namespace,
        require_fix: bool = False
) -> torch.nn.Module:
    """
    Generate the Feature Extractor along with its dimensionality reduction operator.
    :param flags: the namespace of argparse with the parameters.
    :param require_fix: whether to fix the names of the weights within the downloaded pth.
    :return: the pytorch based feature extractor.
    """
    if require_fix:
        # resnet_mg2.convert_weights_mapping(
        #     flags.cnn_fe, flags.cnn_fe_weights_path, flags.cnn_fe_weights_path
        # )
        pass
    if "resnet" in flags.cnn_fe:
        # base_cnn = resnet_mg2.load_resnet(
        #     name=flags.cnn_fe,
        #     pretrained=True,
        #     requires_grad=False,
        #     weights_path=flags.cnn_fe_weights_path,
        #     num_classes=flags.num_classes
        # )
        # Asks directly for the torchvision model.
        base_cnn = torchvision.models.resnet18(pretrained=True)
    else:
        raise ValueError(f"Unsupported CNN: {flags.cnn_fe}\n")

    return base_cnn


def generate_dimensionality_reduction_operator(
        flags: argparse.Namespace,
        feature_extractor: torch.nn.Module
) -> Tuple[torch.nn.Sequential, bool, Optional[Dict[str, Any]]]:
    """
    Generate the dimensionality reduction operator.
    :param flags: the namespace of argparse with the parameters.
    :param feature_extractor: the feature extractor pytorch module.
    :return:
    """
    output_ = torch.nn.Sequential()
    dr_to_train = False
    dr_args = None
    if "plus" in flags.dr_policy:
        # Define the modified CNN.
        torch_utils.append_layers_with_reduced_filters(
            sequential=feature_extractor, feature_layer=flags.feature_layer,
            copy_cnn=output_, filters_to_keep=flags.filters_to_keep.split(','))
    else:
        output_.add_module("fe", feature_extractor)
        if "filter_selection" in flags.dr_policy:
            dr_ = custom_pytorch_layers.FilterSelectionLayer(
                filters_to_keep=flags.filters_to_keep
            )
            output_.add_module("dr", dr_)
        if "class_selection" in flags.dr_policy:
            # Set to true the need for training class specific DR operator.
            dr_to_train = True
            dr_args = {
                "type": "class_distance",
                "filters": flags.class_distance_filters
            }
    return output_, dr_to_train, dr_args


def main(flags: argparse.Namespace) -> None:
    """
    Main function.
    :type flags: argparse.Namespace
    :param flags: the namespace of argparse with the parameters.
    :return: Nothing.
    """
    print('FLAGS: ')
    for k in vars(flags):
        print('{} : {}'.format(k, vars(flags)[k]))

    # Create output directory, if it does not exist
    if not os.path.exists(flags.output_dir):
        os.makedirs(flags.output_dir)

    # Fix the seed.
    # todo: replace np.random.seed with a random_generator
    random.seed(flags.seed)
    np.random.seed(flags.seed)
    torch.manual_seed(flags.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Define the classes.
    classes_before_change, classes_after_change = generate_classes(flags)

    # Define the dataloader
    dataloader, batch_sampler, test_images, datasets_splits = generate_dataloader(
        flags, classes_before_change, classes_after_change
    )

    # Define the CNN's feature extractor.
    base_cnn = generate_cnn(flags)

    # Define Dimensionality Reduction operator.
    base_cnn, dr_to_train, dr_args = generate_dimensionality_reduction_operator(
        flags, feature_extractor=base_cnn
    )

    # Compute features size
    if flags.is_synthetic:
        input_shape = (3, flags.grid_size, flags.grid_size)
    elif flags.is_audio:
        input_shape, _ = next(iter(dataloader))
        input_shape = input_shape.shape[1:]
    else:
        # Image
        input_shape = (3, flags.image_size, flags.image_size)

    features_size = torch_utils.infer_end_of_cnn_shape(
        shape=input_shape,
        activation=base_cnn.forward
    )
    features_shape = torch_utils.infer_end_of_cnn_shape(
        shape=input_shape,
        activation=base_cnn.forward,
        flatten=False
    )
    print(f"The extracted features have a shape {features_shape} --> (size {features_size})")

    # Compute some constants
    num_datasets = 2 if flags.is_synthetic else len(datasets_splits)
    num_test_samples = flags.base_test_samples * flags.num_classes * num_datasets
    num_test_dataloader_samples_to_skip = \
        datasets_splits[0] - (num_test_samples // num_datasets)  # Skip only on the first dataset!
    if flags.is_synthetic:
        num_test_dataloader_samples_to_skip = datasets_splits[0] - num_test_samples

    samples_per_class_to_test = np.array(flags.samples_per_class_to_test.split(','), dtype=int)
    nn_lr0_to_test = np.array(flags.nn_lr_base.split(','), dtype=float)
    nn_lr0_incremental_to_test = np.array(flags.nn_lr_incremental.split(','), dtype=float)
    nn_base_names = [f"c_{lr_:.0e}" for lr_ in nn_lr0_to_test]
    nn_inc_names = [f"i_{lr_c:.0e}_{lr_i:.0e}" for lr_c in nn_lr0_to_test for lr_i in nn_lr0_incremental_to_test]
    num_nn_grid = len(nn_base_names)
    num_nn_inc_grid = len(nn_inc_names)

    num_comparisons = np.size(samples_per_class_to_test)
    num_incremental_comparisons = np.max(samples_per_class_to_test) * flags.num_classes // flags.incremental_step

    # Define the features to be used in the following
    test_iterator = iter(dataloader)
    start_time = time.time()
    training_features_all, training_labels_all, _ = \
        compute_training_features(
            num_training_samples=np.max(samples_per_class_to_test),
            num_classes=flags.num_classes,
            features_size=int(features_size),
            iterator=test_iterator,
            cnn_forward_fn=base_cnn.forward
        )
    print(f"Features extracted in {time.time() - start_time:.3f} seconds.")

    ####################################################################################################################
    # Base exps (they can be considered as incremental exps).
    # In this experiment, the proposed solution along with other baseline classifiers is tested.
    ####################################################################################################################
    # Define base samples data-structures
    accuracy = {
        "knn": np.zeros(num_comparisons),
        "svm": np.zeros(num_comparisons),
        "nn": np.zeros((num_nn_grid, num_comparisons)),
        "nni": np.zeros((num_nn_inc_grid, num_comparisons))
    }
    predictions = {
        "labels": np.zeros(num_test_samples, dtype=int),
        "knn": np.zeros((num_comparisons, num_test_samples, flags.num_classes)),
        "svm": np.zeros((num_comparisons, num_test_samples, flags.num_classes)),
        "nn": np.zeros((num_nn_grid, num_comparisons, num_test_samples, flags.num_classes)),
        "nni": np.zeros((num_nn_inc_grid, num_comparisons, num_test_samples, flags.num_classes))
    }
    train_time = {
        "knn": np.zeros(num_comparisons),
        "svm": np.zeros(num_comparisons),
        "nn": np.zeros(num_comparisons),
        "nni": np.zeros(num_comparisons)
    }
    test_time = {
        "dl": np.zeros(num_comparisons),
        "fe_dr": np.zeros(num_comparisons),
        "knn": np.zeros(num_comparisons),
        "svm": np.zeros(num_comparisons),
        "nn": np.zeros(num_comparisons),
        "nni": np.zeros(num_comparisons)
    }
    class_distance_filter_stats = np.zeros((num_comparisons, flags.class_distance_filters))
    # Start experiments
    if not flags.skip_base_exps:
        for ii, samples_per_class in enumerate(tqdm(samples_per_class_to_test)):
            # Extract the features with the current number of classes.
            training_features, training_labels = \
                extract_training_features(
                    training_features=training_features_all,
                    training_labels=training_labels_all,
                    num_samples_per_class=samples_per_class
                )

            # Train dimensionality reduction operator, if any.
            if dr_to_train:
                dimred_, training_features, training_labels, fs_ = \
                    learn_dimred(
                        dr_args=dr_args,
                        train_features=training_features,
                        train_labels=training_labels,
                        num_classes=flags.num_classes,
                        features_shape=features_shape
                    )
                class_distance_filter_stats[ii] = fs_
                # Define reduced features size
                realtime_features_size = training_features.shape[1]
            else:
                # Define dimred as the identity.
                dimred_ = identity_dimred_
                # Set the value of realtime features size
                realtime_features_size = features_size

            # Train
            # 1) kNN
            start_time = time.time()
            n_neighbors = int(np.ceil(np.sqrt(samples_per_class * flags.num_classes)))
            knn_ = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
            knn_.fit(training_features, training_labels)
            train_time["knn"][ii] = time.time() - start_time

            # 2) SVM
            start_time = time.time()
            svm_ = svm.SVC()
            svm_.fit(training_features, training_labels)
            train_time["svm"][ii] = time.time() - start_time

            # 3) NN/NNI
            if not flags.skip_nn_exps:
                start_time = time.time()
                nn_to_train = list()
                opt_list = list()
                loss_criterion = torch.nn.CrossEntropyLoss()
                # Create the NN-FC1 classifiers and their optimizers.
                for lr_ in nn_lr0_to_test:
                    nn_fc1 = torch.nn.Linear(
                        in_features=realtime_features_size,
                        out_features=flags.num_classes,
                    )
                    nn_to_train.append(nn_fc1)
                    opt_ = torch.optim.Adam(
                        nn_fc1.parameters(),
                        lr=lr_,
                        weight_decay=flags.nn_weights_alpha
                    )
                    opt_list.append(opt_)
                    # Define a data loader from numpy arrays
                training_features_dataset = \
                    torch.utils.data.TensorDataset(
                        torch.Tensor(training_features), torch.Tensor(training_labels).long()
                    )
                training_features_dataloader = torch.utils.data.DataLoader(
                    training_features_dataset, batch_size=flags.nn_batch_size, shuffle=True
                )
                # Train.
                for ee in range(flags.nn_training_epochs):
                    for kk, (inputs_, lbls_) in enumerate(training_features_dataloader):
                        for (nn_clf_, opt_) in zip(nn_to_train, opt_list):
                            # Zeroes gradients.
                            opt_.zero_grad()
                            # Forward
                            outputs = nn_clf_.forward(inputs_)
                            # Compute loss
                            loss = loss_criterion(outputs, lbls_)
                            # Train step
                            loss.backward()
                            opt_.step()
                # Save train time. It is the mean of all the NN classifiers.
                train_time["nn"][ii] = (time.time() - start_time) / num_nn_grid
                train_time["nni"][ii] = train_time["nn"][ii]
                # Delete useless data structures
                del training_features_dataset
                del training_features_dataloader

                # Create the NN-FC1s and the optimizers for following steps.
                nn_base = list()
                nn_incr = list()
                opt_list.clear()  # Empty the optimizer list.
                for nn_fc1 in nn_to_train:
                    # Copy "base" version
                    nn_base.append(copy.deepcopy(nn_fc1))
                    # Copy an incremental version along with its optimizer for each possible learning rate
                    for lr_i in nn_lr0_incremental_to_test:
                        nn_fc1_copy = copy.deepcopy(nn_fc1)
                        nn_incr.append(nn_fc1_copy)
                        opt_ = torch.optim.Adam(
                            nn_fc1_copy.parameters(),
                            lr=lr_i,
                            weight_decay=flags.nn_weights_alpha
                        )
                        opt_list.append(opt_)
                del nn_to_train  # Remove "originals"

            # Test
            errors_knn_ = np.zeros(num_test_samples)
            errors_svm_ = np.zeros(num_test_samples)
            errors_nn_ = np.zeros((num_nn_grid, num_test_samples))
            errors_nni_ = np.zeros((num_nn_inc_grid, num_test_samples))

            # Skip training samples by updating the sampler
            batch_sampler.update_start(start=num_test_dataloader_samples_to_skip)
            test_iterator = iter(dataloader)
            # for _ in range(num_test_dataloader_samples_to_skip):
            #     next(test_iterator)

            for jj in tqdm(range(num_test_samples)):
                # Get the sample
                ft_, lb_ = extract_next_data(
                    test_iterator=test_iterator,
                    time_results_dict=test_time,
                    base_cnn=base_cnn,
                    dimred_=dimred_,
                    current_idx=ii
                )
                # KNN Classification
                st_time = time.time()
                pred_ = knn_.predict_proba(ft_)
                predicted_label = np.argmax(pred_)
                errors_knn_[jj] = (not predicted_label == lb_.data.numpy())
                predictions["knn"][ii, jj] = pred_
                test_time["knn"][ii] += time.time() - st_time
                # SVM Classification
                st_time = time.time()
                pred_ = svm_.decision_function(ft_)
                predicted_label = svm_.predict(ft_)
                errors_svm_[jj] = (not predicted_label == lb_.data.numpy())
                if flags.num_classes > 2:
                    predictions["svm"][ii, jj] = pred_
                else:
                    predictions["svm"][ii, jj, 0] = pred_  # Save a value only for the first class.
                test_time["svm"][ii] += time.time() - st_time
                # NN Classification
                if not flags.skip_nn_exps:
                    # NN Classification
                    with torch.no_grad():
                        st_time = time.time()
                        for nn_i, nn_fc1 in enumerate(nn_base):
                            pred_ = nn_fc1.forward(torch.Tensor(ft_))
                            _, predicted_label = torch.max(pred_.data, 1)
                            errors_nn_[nn_i, jj] = (predicted_label != lb_)
                            predictions["nn"][nn_i, ii, jj] = pred_.data.numpy()
                        test_time["nn"][ii] += (time.time() - st_time) / num_nn_grid
                    # NNI Classification
                    st_time = time.time()
                    for nn_i, (nn_fc1, opt_) in enumerate(zip(nn_incr, opt_list)):
                        opt_.zero_grad()
                        pred_ = nn_fc1.forward(torch.Tensor(ft_))
                        _, predicted_label = torch.max(pred_.data, 1)
                        errors_nni_[nn_i, jj] = (predicted_label != lb_)
                        predictions["nni"][nn_i, ii, jj] = pred_.data.numpy()
                        loss = loss_criterion(pred_, lb_)
                        loss.backward()
                        opt_.step()
                    test_time["nni"][ii] += (time.time() - st_time) / num_nn_inc_grid

                # Save prediction label (only the first time)
                if not ii:
                    predictions["labels"][jj] = lb_

            # Save results
            accuracy["knn"][ii] = 1 - np.sum(errors_knn_) / num_test_samples
            accuracy["svm"][ii] = 1 - np.sum(errors_svm_) / num_test_samples
            accuracy["nn"][:, ii] = 1 - np.sum(errors_nn_, axis=1) / num_test_samples
            accuracy["nni"][:, ii] = 1 - np.sum(errors_nni_, axis=1) / num_test_samples

            # print(f"Samples per class {samples_per_class} -- "
            #       f"Accuracy: kNN {accuracy['knn'][ii]:.03f} - SVM {accuracy['svm'][ii]:.03f}")

            # Remove useless data structures.
            del training_features
            del training_labels
            del knn_
            del svm_
            if not flags.skip_nn_exps:
                del nn_base
                del nn_incr
                del opt_list

        # After the evaluation, the results are saved to file.
        o_dict = {
            "samples_per_class": samples_per_class_to_test,
            "a_knn": accuracy["knn"],
            "a_svm": accuracy["svm"]
        }
        for alg_name in train_time:
            o_dict[f"tr_{alg_name}"] = train_time[alg_name]
        for alg_name in test_time:
            o_dict[f"te_{alg_name}"] = test_time[alg_name]
        for ii, alg_name in enumerate(nn_base_names):
            o_dict[f"a_nn_{alg_name}"] = accuracy["nn"][ii]
        for ii, alg_name in enumerate(nn_inc_names):
            o_dict[f"a_nni_{alg_name}"] = accuracy["nni"][ii]
        df_ = pd.DataFrame(o_dict)
        print("Accuracy stats: ")
        print(df_.tail())

        df_.to_csv(
            os.path.join(flags.output_dir, f"results_base_{flags.seed}.csv"),
            float_format="%.3f"
        )
        for alg_name in predictions:
            np.save(
                os.path.join(flags.output_dir, f"predictions_{alg_name}_seed_{flags.seed}"),
                predictions[alg_name]
            )
        if dr_to_train:
            np.save(
                os.path.join(flags.output_dir, f"class_distance_filter_stats_seed_{flags.seed}"),
                class_distance_filter_stats
            )

    ####################################################################################################################
    # Incremental exps:
    # Given two data-loaders, the incremental-TML solution learns from the training dataloader, then its accuracy
    # capability is estimated with the testing dataloader.
    # Needless to say, no sample belonging to the test dataloader is added to incremental-TML knowledge base.
    #
    # Observation: the training dataloader is not created, since the features after the feature extractor and the
    # dimensionality reduction operator are already available.
    ####################################################################################################################
    # Define base samples data-structures
    accuracy_incremental = {
        "knn": np.zeros(num_incremental_comparisons)
    }
    predictions_incremental = {
        "knn": np.zeros((num_incremental_comparisons, num_test_samples, flags.num_classes)),
    }
    if flags.do_incremental:
        training_features, training_labels = \
            extract_training_features(
                training_features=training_features_all,
                training_labels=training_labels_all,
                num_samples_per_class=1
            )
        if dr_to_train:
            dimred_, training_features, training_labels, fs_ = \
                learn_dimred(
                    dr_args=dr_args,
                    train_features=training_features,
                    train_labels=training_labels,
                    num_classes=flags.num_classes,
                    features_shape=features_shape
                )
            class_distance_filter_stats[ii] = fs_
            # Define reduced features size
            # realtime_features_size = training_features.shape[1]
        else:
            # Define dimred as the identity.
            dimred_ = identity_dimred_
            # Set the value of realtime features size
            # realtime_features_size = features_size

        # Create the incremental kNN object. The number of neighbors is automatically computed.
        knn_ = incremental_knn.IncrementalTinyNearestNeighbor()
        knn_.fit(x=training_features, y=training_labels)

        # Compute the mask of all the training features to skip the already given samples
        train_mask = np.all(np.isin(training_features_all, training_features), axis=1)
        assert np.sum(train_mask) == flags.num_classes, "The train mask does not contain the correct number of True(s)."
        train_mask = np.logical_not(train_mask)

        # Loop over the other training samples.
        # Warning: the last batch of training samples is not tested.
        for ii, (ft_, lb_) in enumerate(zip(training_features_all[train_mask], training_labels_all[train_mask])):
            # Test current kNN
            # At the first iteration it contains one sample per class.
            if ii % flags.incremental_step == 0:
                errors_knn_ = np.zeros(num_test_samples)
                # Init test iterator
                batch_sampler.update_start(start=num_test_dataloader_samples_to_skip)
                test_iterator = iter(dataloader)
                # Loop over test samples.
                for jj in tqdm(range(num_test_samples)):
                    # Get the sample
                    im_, lb_ = next(test_iterator)
                    # Compute feature extractor + dimensionality reduction
                    ft_ = torch.flatten(base_cnn(im_))  # FE with flattening.
                    ft_ = dimred_(ft_.data.numpy().reshape(1, -1))  # DR
                    # KNN Classification (the incremental update is "disabled" by not providing the true label)
                    pred_ = knn_.predict_proba(ft_)
                    predicted_label = np.argmax(pred_)
                    errors_knn_[jj] = (not predicted_label == lb_.data.numpy())
                    predictions_incremental["knn"][ii // flags.incremental_step, jj] = pred_

                accuracy_incremental["knn"][ii // flags.incremental_step] = 1 - np.sum(errors_knn_) / num_test_samples
            # Incremental prediction
            knn_.predict(ft_.reshape(1, -1), y_true=lb_)

        # After the evaluation, the results are saved to file.
        incremental_samples = np.arange(flags.num_classes, flags.num_classes * np.max(samples_per_class_to_test),
                                        flags.incremental_step)
        if np.size(incremental_samples) < num_incremental_comparisons:
            incremental_samples.resize((num_incremental_comparisons,))
        df_ = pd.DataFrame({
            "samples": incremental_samples,
            "a_knn": accuracy_incremental["knn"]
        })
        print("Incremental stats: ")
        print(df_.tail())

        df_.to_csv(
            os.path.join(flags.output_dir, f"results_incremental_{flags.seed}.csv"),
            float_format="%.3f"
        )
        np.save(
            os.path.join(flags.output_dir, f"predictions_knn_incremental_seed_{flags.seed}"),
            predictions_incremental["knn"]
        )

    ####################################################################################################################
    # Passive (CIT), Active, and Hybrid exps:
    # The CIT algorithm passively updates over time, by adding the misclassified samples to its knowledge base.
    # The Active algorithm relies on a CDT to detect changes and then adapt.
    # The Hybrid mixes both the CIT and the Active.
    ####################################################################################################################
    # Define Active Tiny kNN and Hybrid Tiny kNN configurations to test
    # todo: make this choice available in FLAGS!
    # active_cases = ['accuracy_fast_condensing', 'accuracy_fast', 'confidence_fast_condensing', 'confidence_fast']
    active_cases = ['accuracy_fast_condensing', 'accuracy_fast']
    num_active_cases = len(active_cases)

    thresholds_to_test = np.array([10, 20, 25, 50, 100])
    # hybrid_names = ['accuracy_fast', 'confidence_fast']
    hybrid_names = ['accuracy_fast']
    condensing_suffix = ['_condensing', '']
    hybrid_cases = ['{}_{}_{}'.format(nn, cc, tr) for nn in hybrid_names for cc in condensing_suffix for tr in
                    thresholds_to_test if not (nn == 'confidence_fast' and tr < 50)]
    num_hybrid_cases = len(hybrid_cases)

    # Fix window length: the maximum value is number of samples per class times the number of classes.
    w_len = np.max(samples_per_class_to_test) * flags.num_classes * 10
    if flags.window_length < w_len:
        w_len = flags.window_length

    accuracy_tiny = {
        "cit": np.zeros(num_comparisons),
        "c_knn": np.zeros(num_comparisons),
    }
    predictions_tiny = {
        "cit": np.zeros((num_comparisons, num_test_samples, flags.num_classes)),
        "c_knn": np.zeros((num_comparisons, num_test_samples, flags.num_classes)),
    }
    errors_tiny = {
        "cit": np.zeros((num_comparisons, num_test_samples)),
        "c_knn": np.zeros((num_comparisons, num_test_samples)),
    }
    train_time_tiny = {
        "cit": np.zeros(num_comparisons),
        "c_knn": np.zeros(num_comparisons),
    }
    test_time_tiny = {
        "dl": np.zeros(num_comparisons),
        "fe_dr": np.zeros(num_comparisons),
        "cit": np.zeros(num_comparisons),
        "c_knn": np.zeros(num_comparisons),
    }
    samples_tiny = {
        "c_knn": np.zeros((num_comparisons, num_test_samples + 1)),
        "cit": np.zeros((num_comparisons, num_test_samples + 1)),
    }
    if flags.do_active:
        accuracy_tiny["active"] = np.zeros((num_active_cases, num_comparisons))
        accuracy_tiny["hybrid"] = np.zeros((num_hybrid_cases, num_comparisons))
        predictions_tiny["active"] = np.zeros((num_active_cases, num_comparisons, num_test_samples, flags.num_classes))
        predictions_tiny["hybrid"] = np.zeros((num_hybrid_cases, num_comparisons, num_test_samples, flags.num_classes))
        errors_tiny["active"] = np.zeros((num_active_cases, num_comparisons, num_test_samples))
        errors_tiny["hybrid"] = np.zeros((num_hybrid_cases, num_comparisons, num_test_samples))
        train_time_tiny["active"] = np.zeros((num_active_cases, num_comparisons))
        train_time_tiny["hybrid"] = np.zeros((num_hybrid_cases, num_comparisons))
        test_time_tiny["active"] = np.zeros((num_active_cases, num_comparisons))
        test_time_tiny["hybrid"] = np.zeros((num_hybrid_cases, num_comparisons))
        samples_tiny["active"] = np.zeros((num_active_cases, num_comparisons, num_test_samples + 1))
        samples_tiny["hybrid"] = np.zeros((num_hybrid_cases, num_comparisons, num_test_samples + 1))

        cdt_metric_tiny = {
            "active": np.zeros((num_active_cases, num_comparisons, num_test_samples)),
            "hybrid": np.zeros((num_hybrid_cases, num_comparisons, num_test_samples)),
        }
        cdt_detection_tiny = {
            "active": np.zeros((num_active_cases, num_comparisons, num_test_samples)),
            "hybrid": np.zeros((num_hybrid_cases, num_comparisons, num_test_samples)),
        }
        cdt_refined_tiny = {
            "active": np.zeros((num_active_cases, num_comparisons, num_test_samples)),
            "hybrid": np.zeros((num_hybrid_cases, num_comparisons, num_test_samples)),
        }

    # Store the permutation sequences for further usage.
    perm_idxs_dict = dict()

    # Start experiments
    if flags.do_passive or flags.do_active:
        for ii, samples_per_class in enumerate(tqdm(samples_per_class_to_test)):
            # Extract the features with the current number of classes.
            training_features, training_labels = \
                extract_training_features(
                    training_features=training_features_all,
                    training_labels=training_labels_all,
                    num_samples_per_class=samples_per_class
                )

            # Train dimensionality reduction operator, if any.
            if dr_to_train:
                dimred_, training_features, training_labels, _ = \
                    learn_dimred(
                        dr_args=dr_args,
                        train_features=training_features,
                        train_labels=training_labels,
                        num_classes=flags.num_classes,
                        features_shape=features_shape
                    )
            else:
                # Define dimred as the identity.
                dimred_ = identity_dimred_

            # Create the permutation indices (to have all equals in all the objects)
            perm_idxs = np.random.permutation(training_features.shape[0])
            perm_idxs_dict[ii] = perm_idxs

            if flags.do_passive:
                # Create the condensed kNN object. The number of neighbors is automatically computed.
                start_time = time.time()
                c_knn_ = condensed_nearest_neighbor.CondensedNearestNeighbor(
                    shuffle=True,
                    perm_idxs=perm_idxs
                )
                c_knn_.fit(training_features, training_labels)
                train_time_tiny["c_knn"] = time.time() - start_time
                samples_tiny["c_knn"][ii, 0] = c_knn_.get_knn_samples()
                # Create the cit object
                start_time = time.time()
                cit_ = condensing_in_time.CondensingInTimeNearestNeighbor(
                    shuffle=True,
                    perm_idxs=perm_idxs,
                    max_samples=flags.cit_max_samples
                )
                cit_.fit(training_features, training_labels)
                train_time_tiny["cit"] = time.time() - start_time
                samples_tiny["cit"][ii, 0] = cit_.get_knn_samples()

            if flags.do_active:
                # Create the various active Tiny kNN objects.
                active_tiny_knn_list = list()
                active_tiny_knn_idx = 0
                # cdt_metric_list = ['accuracy', 'confidence']
                cdt_metric_list = ['accuracy']
                cdt_metric_init_fn_list = [
                    active_cdt_functions.initialize_cusum_cdt_accuracy,
                    # active_cdt_functions.initialize_cusum_cdt_change_normal_mean
                ]
                _adaptation_mode = 'fast'
                for _cdt_metric, _cdt_init_fn in zip(cdt_metric_list, cdt_metric_init_fn_list):
                    for _condensing in [True, False]:
                        # Create and fit the object
                        start_time = time.time()
                        kac_ = active_tiny_knn.AdaptiveNearestNeighbor(
                            history_window_length=w_len,
                            step_size=flags.n_binomial,
                            use_condensing=_condensing,
                            perm_idxs=perm_idxs,
                            cdt_metric=_cdt_metric,
                            adaptation_mode=_adaptation_mode,
                            cdt_init_fn=_cdt_init_fn
                        )
                        kac_.fit(training_features, training_labels)
                        train_time_tiny["active"][active_tiny_knn_idx] = time.time() - start_time
                        samples_tiny["active"][active_tiny_knn_idx] = kac_.get_knn_samples()
                        # Add the object to the list
                        active_tiny_knn_list.append(kac_)
                        active_tiny_knn_idx += 1

                # Create the various hybrid Tiny kNN objects.
                hybrid_tiny_knn_list = list()
                hybrid_tiny_knn_idx = 0
                # Note that cdt_metric_list, etc. are the same of active case
                cdt_arguments_dict_list = [{"allow_above_p0": False}, {"allow_above_mu0": False}]
                for _cdt_metric, _cdt_init_fn, _cdt_args in zip(
                        cdt_metric_list, cdt_metric_init_fn_list, cdt_arguments_dict_list
                ):
                    for _condensing in [True, False]:
                        for _threshold in thresholds_to_test:
                            if not (_cdt_metric == 'confidence' and _threshold < 50):
                                # Create and fit the object
                                start_time = time.time()
                                kac_ = hybrid_tiny_knn.AdaptiveHybridNearestNeighbor(
                                    window_length=w_len,
                                    cdt_threshold=_threshold,
                                    step_size=flags.n_binomial,
                                    use_condensing=_condensing,
                                    perm_idxs=perm_idxs,
                                    cdt_metric=_cdt_metric,
                                    adaptation_mode=_adaptation_mode,
                                    cdt_init_fn=_cdt_init_fn,
                                    cdt_init_kwargs=_cdt_args
                                )
                                kac_.fit(training_features, training_labels)
                                train_time_tiny["hybrid"][hybrid_tiny_knn_idx] = time.time() - start_time
                                samples_tiny["hybrid"][hybrid_tiny_knn_idx] = kac_.get_knn_samples()
                                # Add the object to the list
                                hybrid_tiny_knn_list.append(kac_)
                                hybrid_tiny_knn_idx += 1

            # Skip training samples by updating the sampler
            batch_sampler.update_start(start=num_test_dataloader_samples_to_skip)
            test_iterator = iter(dataloader)

            for jj in tqdm(range(num_test_samples)):
                # Get the sample
                ft_, lb_ = extract_next_data(
                    test_iterator=test_iterator,
                    time_results_dict=test_time,
                    base_cnn=base_cnn,
                    dimred_=dimred_,
                    current_idx=ii
                )

                if flags.do_passive:
                    # Predict condensed knn
                    st_time = time.time()
                    predicted_label = c_knn_.predict(ft_)
                    test_time_tiny["c_knn"][ii] += time.time() - st_time
                    errors_tiny["c_knn"][ii, jj] = (not predicted_label == lb_.data.numpy())
                    predictions_tiny["c_knn"][ii, jj] = c_knn_.predict_proba(ft_)
                    samples_tiny["c_knn"][ii, jj] = c_knn_.get_knn_samples()
                    # Predict passive
                    predictions_tiny["cit"][ii, jj] = cit_.predict_proba(ft_)  # Proba before possible adaptations!
                    st_time = time.time()
                    predicted_label = cit_.predict(ft_, y_true=lb_.data.numpy())
                    test_time_tiny["cit"][ii] += time.time() - st_time
                    errors_tiny["cit"][ii, jj] = (not predicted_label == lb_.data.numpy())
                    samples_tiny["cit"][ii, jj] = cit_.get_knn_samples()
                if flags.do_active:
                    # Active and hybrid inner loop:
                    for _key, _list in zip(["active", "hybrid"], [active_tiny_knn_list, hybrid_tiny_knn_list]):
                        for kk, kac_ in enumerate(_list):
                            # Predict the probabilities always before the "classic" predict because the latter handles
                            # the adaptation of the model.
                            pred_proba = kac_.predict_proba(ft_)
                            st_time = time.time()
                            # Check if there is a prediction for each class: fill with -1 otherwise
                            if np.size(pred_proba) < flags.num_classes:
                                pred_proba = \
                                    np.append(pred_proba, -1 * np.ones(flags.num_classes - np.size(pred_proba)))
                            pred_lbs_ = kac_.predict(ft_, y_true=lb_.data.numpy())
                            test_time_tiny[_key][kk, ii] += time.time() - st_time
                            # errors_active[kk, jj] = (not pred_lbs_ == lb_.data.numpy())
                            errors_tiny[_key][kk, ii, jj] = (not pred_lbs_ == lb_.data.numpy())
                            predictions_tiny[_key][kk, ii, jj] = pred_proba
                            samples_tiny[_key][kk, ii, jj] = kac_.get_knn_samples()

            # Save results
            if flags.do_passive:
                accuracy_tiny["c_knn"][ii] = 1 - np.sum(errors_tiny["c_knn"][ii]) / num_test_samples
                accuracy_tiny["cit"][ii] = 1 - np.sum(errors_tiny["cit"][ii]) / num_test_samples

            if flags.do_active:
                accuracy_tiny["active"][:, ii] = 1 - np.sum(errors_tiny["active"][:, ii], axis=1) / num_test_samples
                accuracy_tiny["hybrid"][:, ii] = 1 - np.sum(errors_tiny["hybrid"][:, ii], axis=1) / num_test_samples

                for _key, _list in zip(["active", "hybrid"], [active_tiny_knn_list, hybrid_tiny_knn_list]):
                    for kk, kac_ in enumerate(_list):
                        cdt_metric = kac_.get_cdt_metric_history()
                        cdt_metric_tiny[_key][kk, ii, :np.size(cdt_metric)] = cdt_metric
                        # Save detections with -1 since the indices start from zero and not 1 :)
                        cdt_detection_tiny[_key][kk, ii, kac_.get_detections() - 1] = 1
                        cdt_refined_tiny[_key][kk, ii, kac_.get_estimated_change_times() - 1] = 1

            # Remove useless data structures.
            del training_features
            del training_labels
            if flags.do_passive:
                del c_knn_
                del cit_
            if flags.do_active:
                active_tiny_knn_list.clear()
                hybrid_tiny_knn_list.clear()

        # After the evaluation, the results are saved to file.
        o_dict_tiny = {
            "samples_per_class": samples_per_class_to_test,
            "a_c_knn": accuracy_tiny["c_knn"],
            "a_cit": accuracy_tiny["cit"]
        }
        for alg_name in train_time_tiny:
            if alg_name not in ["active", "hybrid"]:
                o_dict_tiny[f"tr_{alg_name}"] = train_time_tiny[alg_name]
        for alg_name in test_time_tiny:
            if alg_name not in ["active", "hybrid"]:
                o_dict_tiny[f"te_{alg_name}"] = test_time_tiny[alg_name]
        # Deal with multiple options of active and hybrid algorithms
        if flags.do_active:
            for ii, alg_name in enumerate(active_cases):
                o_dict_tiny[f"a_active_{alg_name}"] = accuracy_tiny["active"][ii]
                o_dict_tiny[f"tr_active_{alg_name}"] = train_time_tiny["active"][ii]
                o_dict_tiny[f"te_active_{alg_name}"] = test_time_tiny["active"][ii]
            for ii, alg_name in enumerate(hybrid_cases):
                o_dict_tiny[f"a_hybrid_{alg_name}"] = accuracy_tiny["hybrid"][ii]
                o_dict_tiny[f"tr_hybrid_{alg_name}"] = train_time_tiny["hybrid"][ii]
                o_dict_tiny[f"te_hybrid_{alg_name}"] = test_time_tiny["hybrid"][ii]

        df_tiny = pd.DataFrame(o_dict_tiny)
        print("Tiny Accuracy stats: ")
        print(df_tiny.tail())

        df_tiny.to_csv(
            os.path.join(flags.output_dir, f"results_tiny_{flags.seed}.csv"),
            float_format="%.3f"
        )
        for alg_name in predictions_tiny:
            np.save(
                os.path.join(flags.output_dir, f"predictions_{alg_name}_seed_{flags.seed}"),
                predictions_tiny[alg_name]
            )
            np.save(
                os.path.join(flags.output_dir, f"samples_{alg_name}_seed_{flags.seed}"),
                samples_tiny[alg_name]
            )
            if alg_name in cdt_metric_tiny:
                np.save(
                    os.path.join(flags.output_dir, f"cdt_metric_{alg_name}_seed_{flags.seed}"),
                    cdt_metric_tiny[alg_name]
                )
                np.save(
                    os.path.join(flags.output_dir, f"cdt_detection_{alg_name}_seed_{flags.seed}"),
                    cdt_detection_tiny[alg_name]
                )
                np.save(
                    os.path.join(flags.output_dir, f"cdt_refined_{alg_name}_seed_{flags.seed}"),
                    cdt_refined_tiny[alg_name]
                )

    ####################################################################################################################
    # Other methods from related literature experiments.
    # todo: improve this description by adding the reference to the papers and more insightful details.
    # The kNN+ADWIN updates the knowledge set of the kNN when the ADWIN CDT detects a change.
    # The kNN+ADWIN with PAW updates the knowledge set of the KNN as kNN+ADWIN but introduces the Probabilist Adaptive
    # Window (PAW) to removes redundant knowledge.
    # The SAM-kNN imitates the short term and long term memory in human brain.
    ####################################################################################################################
    accuracy_soa = {
        "knn_adwin": np.zeros(num_comparisons),
        "knn_adwin_paw": np.zeros(num_comparisons),
        "knn_sam": np.zeros(num_comparisons),
    }
    predictions_soa = {
        "knn_adwin": np.zeros((num_comparisons, num_test_samples, flags.num_classes)),
        "knn_adwin_paw": np.zeros((num_comparisons, num_test_samples, flags.num_classes)),
        "knn_sam": np.zeros((num_comparisons, num_test_samples, flags.num_classes)),
    }
    errors_soa = {
        "knn_adwin": np.zeros((num_comparisons, num_test_samples)),
        "knn_adwin_paw": np.zeros((num_comparisons, num_test_samples)),
        "knn_sam": np.zeros((num_comparisons, num_test_samples)),
    }
    train_time_soa = {
        "knn_adwin": np.zeros(num_comparisons),
        "knn_adwin_paw": np.zeros(num_comparisons),
        "knn_sam": np.zeros(num_comparisons),
    }
    test_time_soa = {
        "dl": np.zeros(num_comparisons),
        "fe_dr": np.zeros(num_comparisons),
        "knn_adwin": np.zeros(num_comparisons),
        "knn_adwin_paw": np.zeros(num_comparisons),
        "knn_sam": np.zeros(num_comparisons),
    }
    samples_soa = {
        "knn_adwin": np.zeros((num_comparisons, num_test_samples + 1)),
        "knn_adwin_paw": np.zeros((num_comparisons, num_test_samples + 1)),
        "knn_sam": np.zeros((num_comparisons, num_test_samples + 1)),
    }
    # cdt_metric_soa = {
    #     "knn_adwin": np.zeros((num_comparisons, num_test_samples)),
    #     "knn_adwin_paw": np.zeros((num_comparisons, num_test_samples)),
    #     "knn_sam": np.zeros((num_comparisons, num_test_samples)),
    # }
    # cdt_detection_soa = {
    #     "knn_adwin": np.zeros((num_comparisons, num_test_samples)),
    #     "knn_adwin_paw": np.zeros((num_comparisons, num_test_samples)),
    #     "knn_sam": np.zeros((num_comparisons, num_test_samples)),
    # }
    # cdt_refined_soa = {
    #     "knn_adwin": np.zeros((num_comparisons, num_test_samples)),
    #     "knn_adwin_paw": np.zeros((num_comparisons, num_test_samples)),
    #     "knn_sam": np.zeros((num_comparisons, num_test_samples)),
    # }

    # Start experiments
    if flags.do_knn_adwin or flags.do_knn_adwin_paw or flags.do_sam_knn:
        # Recompute the training features only if the dataset is synthetic.
        if flags.is_synthetic and flags.do_soa_without_dl:
            # Define the features to be used in the following
            print(f"Synthetic dataset. Recomputing the features without deep learning for SOA algorithms.")
            test_iterator = iter(dataloader)
            start_time = time.time()
            training_features_all, training_labels_all, _ = \
                compute_training_features_nodl(
                    num_training_samples=np.max(samples_per_class_to_test),
                    num_classes=flags.num_classes,
                    features_size=int(np.prod(input_shape)),
                    iterator=test_iterator
                )
            print(f"Features extracted in {time.time() - start_time:.3f} seconds.")

        # Define get_samples functions
        def get_knn_adwin_samples(knn: river.neighbors.knn_adwin.KNNADWINClassifier):
            return knn.data_window.size

        def get_knn_sam_samples(knn: river.neighbors.sam_knn.SAMKNNClassifier):
            return len(knn.STMLabels) + len(knn.LTMLabels)

        # Experiments
        for ii, samples_per_class in enumerate(tqdm(samples_per_class_to_test)):
            # Extract the features with the current number of classes.
            training_features, training_labels = \
                extract_training_features(
                    training_features=training_features_all,
                    training_labels=training_labels_all,
                    num_samples_per_class=samples_per_class
                )

            # Train dimensionality reduction operator, if any.
            if dr_to_train:
                dimred_, training_features, training_labels, _ = \
                    learn_dimred(
                        dr_args=dr_args,
                        train_features=training_features,
                        train_labels=training_labels,
                        num_classes=flags.num_classes,
                        features_shape=features_shape
                    )
            else:
                # Define dimred as the identity.
                dimred_ = identity_dimred_

            # DISCLAIMER: The number of neighbors is here fixed in a similar way to the proposed algorithms, i.e.,
            # to the ceiling of the square root of the number of sampling.
            # Similarly, the window size (when available) is sized as per active tiny knn. Note that w_len has been
            # already computed at this stage, independently of the flag do_active.
            num_neighbors = int(np.ceil(np.sqrt(samples_per_class)))

            # Create the permutation indices if not already created.
            perm_idxs = \
                perm_idxs_dict[ii] if ii in perm_idxs_dict else np.random.permutation(training_features.shape[0])
            training_features = training_features[perm_idxs]
            training_labels = training_labels[perm_idxs]

            knn_soa_to_test = list()
            knn_soa_keys = list()
            knn_soa_get_samples_fn = list()

            # Fit the algorithm(s)
            if flags.do_knn_adwin:
                start_time = time.time()
                knn_adwin = river.neighbors.knn_adwin.KNNADWINClassifier(
                    n_neighbors=num_neighbors,
                    window_size=w_len,
                    p=2  # Euclidean distance as distance metric
                )
                # Call partial_fit with initial training data. There is no other way than call sample by sample!
                for _x, _y in zip(training_features, training_labels):
                    knn_adwin.learn_one({ii: v for ii, v in enumerate(_x)}, _y)
                train_time_soa["knn_adwin"] = time.time() - start_time
                # todo: check if there is another way than accessing the single attributes
                samples_soa["knn_adwin"][ii, 0] = get_knn_adwin_samples(knn_adwin)
                knn_soa_to_test.append(knn_adwin)
                knn_soa_keys.append("knn_adwin")
                knn_soa_get_samples_fn.append(get_knn_adwin_samples)
            if flags.do_knn_adwin_paw:
                print("KNN ADWIN PAW not supported yet.")
            if flags.do_sam_knn:
                start_time = time.time()
                knn_sam = river.neighbors.sam_knn.SAMKNNClassifier(
                    n_neighbors=num_neighbors,
                    window_size=w_len
                    # Default for other params.
                    # todo: check distance_weighting param meaning!
                )
                # Call partial_fit with initial training data. There is no other way than call sample by sample!
                for _x, _y in zip(training_features, training_labels):
                    knn_sam.learn_one({ii: v for ii, v in enumerate(_x)}, _y)
                train_time_soa["knn_sam"] = time.time() - start_time
                # todo: check if there is another way than accessing the single attributes
                samples_soa["knn_sam"][ii, 0] = get_knn_sam_samples(knn_sam)
                knn_soa_to_test.append(knn_sam)
                knn_soa_keys.append("knn_sam")
                knn_soa_get_samples_fn.append(get_knn_sam_samples)

            # Skip training samples by updating the sampler
            batch_sampler.update_start(start=num_test_dataloader_samples_to_skip)
            test_iterator = iter(dataloader)

            for jj in tqdm(range(num_test_samples)):
                # Get the sample
                ft_, lb_ = extract_next_data(
                    test_iterator=test_iterator,
                    time_results_dict=test_time,
                    base_cnn=base_cnn,
                    dimred_=dimred_,
                    current_idx=ii,
                    apply_deep_learning=not (flags.is_synthetic and flags.do_soa_without_dl)
                )

                # Create a dict from ft_
                ft_dict = {ii: v for ii, v in enumerate(ft_.flatten())}

                # if flags.do_knn_adwin or flags.do_knn_sam:
                for kac_, _key, _get_samples in zip(knn_soa_to_test, knn_soa_keys, knn_soa_get_samples_fn):
                    try:
                        pred_proba = kac_.predict_proba_one(ft_dict)
                        #  The predict_proba_one method returns a dict {label: prob} where the probability is computed
                        # and weighted by the distance of each neighbor.
                        # Convert the dict to a numpy array assuming that the labels are ordered indices.
                        pred_proba = np.array([pred_proba[x] for x in sorted(pred_proba.keys())])
                    except NotImplementedError:
                        pred_proba = np.zeros(flags.num_classes)
                        pred_proba[kac_.predict_one(ft_dict)] = 1
                    st_time = time.time()
                    # Check the case where there is no probability
                    if np.sum(pred_proba) == 0:
                        pred_proba -= 1
                        pred_lbs_ = -1
                    else:
                        # Check if there is a prediction for each class: fill with -1 otherwise
                        if np.size(pred_proba) < flags.num_classes:
                            pred_proba = \
                                np.append(pred_proba, -1 * np.ones(flags.num_classes - np.size(pred_proba)))
                        pred_lbs_ = np.argmax(pred_proba)
                    # Learning step
                    kac_.learn_one(ft_dict, int(lb_.data.numpy()))
                    test_time_soa[_key][ii] += time.time() - st_time
                    errors_soa[_key][ii, jj] = (not pred_lbs_ == lb_.data.numpy())
                    predictions_soa[_key][ii, jj] = pred_proba
                    samples_soa[_key][ii, jj] = _get_samples(kac_)

            # Save results
            for alg_name in accuracy_soa:
                accuracy_soa[alg_name][ii] = 1 - np.sum(errors_soa[alg_name][ii]) / num_test_samples
            # accuracy_soa["knn_adwin"][ii] = 1 - np.sum(errors_soa["knn_adwin"][ii]) / num_test_samples
            # accuracy_soa["knn_adwin_paw"][ii] = 1 - np.sum(errors_soa["knn_adwin_paw"][ii]) / num_test_samples
            # accuracy_soa["knn_sam"][ii] = 1 - np.sum(errors_soa["knn_sam"][ii]) / num_test_samples

            # Remove useless data structures.
            del training_features
            del training_labels
            knn_soa_to_test.clear()
            knn_soa_keys.clear()
            knn_soa_get_samples_fn.clear()

        # After the evaluation, the results are saved to file.
        o_dict_soa = {
            "samples_per_class": samples_per_class_to_test,
        }
        for alg_name in accuracy_soa:
            o_dict_soa[f"a_{alg_name}"] = accuracy_soa[alg_name]
            o_dict_soa[f"tr_{alg_name}"] = train_time_soa[alg_name]
        for alg_name in test_time_soa:
            o_dict_soa[f"te_{alg_name}"] = test_time_soa[alg_name]

        df_soa = pd.DataFrame(o_dict_soa)
        print("State of the Art Accuracy stats: ")
        print(df_soa.tail())

        df_soa.to_csv(
            os.path.join(flags.output_dir, f"results_soa_{flags.seed}.csv"),
            float_format="%.3f"
        )
        for alg_name in predictions_soa:
            np.save(
                os.path.join(flags.output_dir, f"predictions_{alg_name}_seed_{flags.seed}"),
                predictions_soa[alg_name]
            )
            np.save(
                os.path.join(flags.output_dir, f"samples_{alg_name}_seed_{flags.seed}"),
                samples_soa[alg_name]
            )


if __name__ == "__main__":
    # Define and parse the script flags.
    flags_ = define_and_parse_flags(parse=True)
    # Execute the main function with the given flags.
    main(flags=flags_)
