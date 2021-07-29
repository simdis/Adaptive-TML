import numpy as np
import torch

from typing import Any, Callable, List, Optional, Tuple, Union


def infer_end_of_cnn_shape(
        shape: Union[np.ndarray, torch.Tensor],
        activation: Callable,
        act_args: Optional[Tuple[Any, ...]] = None,
        flatten: bool = True
) -> torch.Tensor:
    """
    Infer the shape of features at the end of the given CNN, with an input of the give shape.
    :param shape: the input shape.
    :param activation: the function that computes the activations.
    :param act_args: the tuple of arguments of the activation function.
    :param flatten: a boolean flag. If true, the features are flattened before computing their shape.
    :return: the shape of the features at the end of the function.
    """
    batch_size = 1
    rand_inp = torch.rand(batch_size, *shape)
    if act_args is not None:
        output = activation(rand_inp, *act_args)
    else:
        output = activation(rand_inp)
    if flatten:
        return output.view(batch_size, -1).size(1)
    return output.shape[1:]


def append_layers_with_reduced_filters(
        sequential: torch.nn.Module, feature_layer: str,
        copy_cnn: torch.nn.Sequential, requires_grad: bool = False,
        filters_to_keep: Optional[Union[List[int], np.ndarray, torch.Tensor]] = None,
        _verbose=False
) -> bool:
    """
    Function that (recursively) flattens the sequence of layers in a Sequential module up to a given feature layer.
    If the parameter filters_to_keep is provided, the convolutional and batch-norm layers are sub-sampled by keeping
    only the provided indices.
    Warnings:
        1) Actually the function does not allow different filters for the various convolutions.
        2) The function does not enter into specific blocks, such as the Bottleneck block of ResNet.
    :param sequential: the original CNN.
    :param feature_layer: the last layer to keep.
    :param copy_cnn: the sequential object where to new NN is created.
    :param requires_grad: requires_grad parameter (see PyTorch doc).
    :param filters_to_keep: optional list of layers to be kept in each convolutional and bn layer.
    :param _verbose: whether to display or not verbose logging.
    :return: true if the feature_layer has been found, false otherwise.
    """
    if _verbose:
        print('Called append_layers_with_reduced_filters!')
    # Flag that is true only when the feature layer has been found.
    found = False
    num_filters = -1
    if filters_to_keep is not None:
        num_filters = len(filters_to_keep)
        filters_to_keep = np.array(filters_to_keep).astype(np.int32)

    previous_channels = -1
    for ly_ in sequential.named_children():
        if _verbose:
            print('Processing layer {}'.format(ly_[0]))
        # Continue only if !found
        if not found:
            if isinstance(ly_[1], torch.nn.Sequential):
                # Recursive call.
                if _verbose:
                    print('Recursive call.')
                found = append_layers_with_reduced_filters(
                    sequential=ly_[1], feature_layer=feature_layer,
                    copy_cnn=copy_cnn, requires_grad=requires_grad,
                    filters_to_keep=filters_to_keep, _verbose=_verbose
                )
            # elif "conv" in ly_[0]:
            elif isinstance(ly_[1], torch.nn.Conv2d):
                if _verbose:
                    print('Found Conv Layer! Reducing it.')
                # Get the conv layer
                conv_ = ly_[1]
                if num_filters > 0:
                    # Modify it.
                    if previous_channels > 0:
                        conv_.in_channels = previous_channels
                    conv_.out_channels = num_filters
                    previous_channels = conv_.out_channels
                    conv_.weight = torch.nn.Parameter(
                        conv_.weight[filters_to_keep], requires_grad=requires_grad
                    )
                    if conv_.bias is not None:
                        conv_.bias = torch.nn.Parameter(
                            conv_.bias[filters_to_keep], requires_grad=requires_grad
                        )
                # Save it.
                copy_cnn.add_module(ly_[0], conv_)
            # elif "norm" in ly_[0]:
            elif isinstance(ly_[1], torch.nn.BatchNorm2d):
                if _verbose:
                    print('Found BatchNorm Layer! Reducing it.')
                # Get the conv layer
                norm_ = ly_[1]
                if num_filters > 0:
                    # Modify it.
                    norm_.num_features = num_filters
                    norm_.weight = torch.nn.Parameter(
                        norm_.weight[filters_to_keep], requires_grad=requires_grad
                    )
                    norm_.bias = torch.nn.Parameter(
                        norm_.bias[filters_to_keep], requires_grad=requires_grad
                    )
                    if norm_.track_running_stats:
                        norm_.running_mean = norm_.running_mean[filters_to_keep]
                        norm_.running_var = norm_.running_var[filters_to_keep]
                # Save it.
                copy_cnn.add_module(ly_[0], norm_)
            else:
                if _verbose:
                    print('Found another layer.')
                copy_cnn.add_module(ly_[0], ly_[1])
            if ly_[0] == feature_layer:
                if _verbose:
                    print('Found the feature layer, stopping!')
                return True
    # If not found return false
    # if _verbose:
    #     print('Created the following CNN: ')
    #     print(copy_cnn)
    return found
