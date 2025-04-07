from collections import OrderedDict
from typing import Callable, List, Tuple, Union

import torch
import torchinfo
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from tqdm import tqdm

from NeuralTSNE.TSNE.Helpers import x2p
from NeuralTSNE.TSNE.CostFunctions import CostFunctions
from NeuralTSNE.TSNE.NeuralNetwork import NeuralNetwork, BaseModel

from NeuralTSNE.Utils import does_sum_up_to


class ParametricTSNE:
    """
    Parametric t-SNE implementation using a neural network model.

    Parameters
    ----------
    `loss_fn` : `str`
        Loss function for t-SNE. Currently supports `kl_divergence`.
    `perplexity` : `int`
        Perplexity parameter for t-SNE.
    `batch_size` : `int`
        Batch size for training.
    `early_exaggeration_epochs` : `int`
        Number of epochs for early exaggeration.
    `early_exaggeration_value` : `float`
        Early exaggeration factor.
    `max_iterations` : `int`
        Maximum number of iterations for optimization.
    `n_components` : `int`, optional
        Number of components in the output. Defaults to `None`.
    `features` : `int`, optional
        Number of input features. Defaults to `None`.
    `multipliers` : `List[float]`, optional
        List of multipliers for hidden layers in the neural network. Defaults to `None`.
    `n_jobs` : `int`, optional
        Number of workers for data loading. Defaults to `0`.
    `tolerance` : `float`, optional
        Tolerance level for convergence. Defaults to `1e-5`.
    `force_cpu` : `bool`, optional
        Force using CPU even if GPU is available. Defaults to `False`.
    `model` : `Union[NeuralNetwork, nn.Module, OrderedDict]`, optional
        Predefined model. Defaults to `None`.
    """

    def __init__(
        self,
        loss_fn: str,
        perplexity: int,
        batch_size: int,
        early_exaggeration_epochs: int,
        early_exaggeration_value: float,
        max_iterations: int,
        n_components: Union[int, None] = None,
        features: Union[int, None] = None,
        multipliers: Union[List[float], None] = None,
        n_jobs: int = 0,
        tolerance: float = 1e-5,
        force_cpu: bool = False,
        model: Union[NeuralNetwork, nn.Module, OrderedDict, None] = None,
    ):
        if model is None and (
            features is None or n_components is None or multipliers is None
        ):
            raise AttributeError(
                "Either a model or features, n_components, and multipliers must be provided."
            )
        if force_cpu or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        self.model = None
        if model is None:
            self.model = NeuralNetwork(features, n_components, multipliers).to(
                self.device
            )
        elif isinstance(model, (NeuralNetwork, BaseModel)):
            self.model = model.to(self.device)
        elif isinstance(model, (OrderedDict, nn.Sequential)):
            self.model = NeuralNetwork(pre_filled_layers=model).to(self.device)

        features = self.model.in_features

        torchinfo.summary(
            self.model,
            input_size=(batch_size, 1, features),
            col_names=(
                "input_size",
                "output_size",
                "num_params",
                "kernel_size",
                "mult_adds",
            ),
        )

        self.perplexity = perplexity
        self.batch_size = batch_size
        self.early_exaggeration_epochs = early_exaggeration_epochs
        self.early_exaggeration_value = early_exaggeration_value
        self.n_jobs = n_jobs
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        self.loss_fn = self.set_loss_fn(loss_fn)

    def set_loss_fn(self, loss_fn: str) -> Callable:
        """
        Set the loss function based on the provided string.

        Parameters
        ----------
        `loss_fn` : `str`
            String indicating the desired loss function.

        Returns
        -------
        `Callable`
            Corresponding loss function.

        Note
        ----
        Currently supports `kl_divergence` as the loss function.
        """
        fn = CostFunctions(loss_fn)
        self.loss_fn = fn
        return fn

    def save_model(self, filename: str):
        """
        Save the model's state dictionary to a file.

        Parameters
        ----------
        `filename` : `str`
            Name of the file to save the model.
        """
        torch.save(self.model.state_dict(), filename)

    def read_model(self, filename: str):
        """
        Load the model's state dictionary from a file.

        Parameters
        ----------
        `filename` : `str`
            Name of the file to load the model.
        """
        self.model.load_state_dict(torch.load(filename))

    def split_dataset(
        self,
        X: torch.Tensor,
        y: torch.Tensor = None,
        train_size: float = None,
        test_size: float = None,
    ) -> Tuple[Union[DataLoader, None], Union[DataLoader, None]]:
        """
        Split the dataset into training and testing set

        Parameters
        ----------
        `X` : `torch.Tensor`
            Input data tensor.
        `y` : `torch.Tensor`, optional
            Target tensor. Default is `None`.
        `train_size` : `float`, optional
            Proportion of the dataset to include in the training set.
        `test_size` : `float`, optional
            Proportion of the dataset to include in the testing set.

        Returns
        -------
        `Tuple[DataLoader | None, DataLoader | None]`
            Tuple containing training and testing dataloaders.

        Note
        ----
        Splits the input data into training and testing sets, and returns corresponding dataloaders.
        """
        train_size, test_size = self._determine_train_test_split(train_size, test_size)
        if y is None:
            dataset = TensorDataset(X)
        else:
            dataset = TensorDataset(X, y)
        train_size = int(train_size * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        if train_size == 0:
            train_dataset = None
        if test_size == 0:
            test_dataset = None

        return self.create_dataloaders(train_dataset, test_dataset)

    def _determine_train_test_split(
        self, train_size: float, test_size: float
    ) -> Tuple[float, float]:
        """
        Determine the proportions of training and testing sets.

        Parameters
        ----------
        `train_size` : `float`
            Proportion of the dataset to include in the training set.
        `test_size` : `float`
            Proportion of the dataset to include in the testing set.

        Returns
        -------
        `Tuple[float, float]`
            Tuple containing the determined proportions.
        """
        if train_size is None and test_size is None:
            train_size = 0.8
            test_size = 1 - train_size
        elif train_size is None:
            train_size = 1 - test_size
        elif test_size is None:
            test_size = 1 - train_size
        elif not does_sum_up_to(train_size, test_size, 1):
            test_size = 1 - train_size
        return train_size, test_size

    def create_dataloaders(
        self, train: Dataset, test: Dataset
    ) -> Tuple[Union[DataLoader, None], Union[DataLoader, None]]:
        """
        Create dataloaders for training and testing sets.

        Parameters
        ----------
        `train` : `Dataset`
            Training dataset.
        `test` : `Dataset`
            Testing dataset.

        Returns
        -------
        `Tuple[DataLoader | None, DataLoader | None]`
            Tuple containing training and testing dataloaders.
        """
        train_loader = (
            DataLoader(
                train,
                batch_size=self.batch_size,
                drop_last=True,
                pin_memory=False if self.device == "cpu" else True,
                num_workers=self.n_jobs if self.device == "cpu" else 0,
            )
            if train is not None
            else None
        )
        test_loader = (
            DataLoader(
                test,
                batch_size=self.batch_size,
                drop_last=False,
                pin_memory=False if self.device == "cpu" else True,
                num_workers=self.n_jobs if self.device == "cpu" else 0,
            )
            if test is not None
            else None
        )
        return train_loader, test_loader

    def _calculate_P(self, dataloader: DataLoader) -> torch.Tensor:
        """
        Calculate joint probability matrix P.

        Parameters
        ----------
        `dataloader` : `DataLoader`
            Dataloader for the dataset.

        Returns
        -------
        `torch.Tensor`
            Joint probability matrix P.
        """
        n = len(dataloader.dataset)
        P = torch.zeros((n, self.batch_size), device=self.device)
        for i, (X, *_) in tqdm(
            enumerate(dataloader),
            unit="batch",
            total=len(dataloader),
            desc="Calculating P",
            leave=True,
            position=0
        ):
            batch = x2p(X, self.perplexity, self.tolerance)
            batch[torch.isnan(batch)] = 0
            batch = batch + batch.mT
            batch = batch / batch.sum()
            batch = torch.maximum(
                batch.to(self.device), torch.tensor([1e-12], device=self.device)
            )
            P[i * self.batch_size : (i + 1) * self.batch_size] = batch
        return P
