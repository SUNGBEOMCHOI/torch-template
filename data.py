import torch

def get_train_dataset():
    """
    Return dataset for training and validation

    Args:

    Returns:
        train_dataset
        valid_dataset
    """
    raise NotImplementedError

def get_test_dataset():
    """
    Return dataset for test

    Args:

    Returns:
        test_dataset
    """
    raise NotImplementedError

def get_dataloader(dataset, batch_size):
    """
    Return torch dataloader for dataset

    Args:
        dataset: Torch dataset
        batch_size

    Returns:
        data_loader: Torch data loader
    """
    raise NotImplementedError