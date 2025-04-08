from typing import Any, List, Tuple, Union

import torch
import torch.optim as optim

import pytorch_lightning as L

from NeuralTSNE.TSNE import ParametricTSNE


class DimensionalityReduction(L.LightningModule):
    """
    Lightning Module for training a neural network-based
    Parametric t-SNE dimensionality reduction model.

    Parameters
    ----------
    `tsne` : `ParametricTSNE`
        Parametric t-SNE model for feature extraction.
    `shuffle` : `bool`
        Flag indicating whether to shuffle data during training.
    `optimizer` : `str`, optional
        Optimizer for training. Defaults to `adam`.
    `lr` : `float`, optional
        Learning rate for the optimizer. Defaults to `1e-3`.

    Note
    ----
    This class defines a Lightning Module for training a neural network-based
    Parametric t-SNE dimensionality reduction model for feature extraction.
    It includes methods for the training step, configuring optimizers, and
    handling the training process.
    """

    def __init__(
        self,
        tsne: "ParametricTSNE",
        shuffle: bool,
        optimizer: str = "adam",
        lr: float = 1e-3,
    ):
        super().__init__()
        self.tsne = tsne
        self.batch_size = tsne.batch_size
        self.model = self.tsne.model
        self.loss_fn = tsne.loss_fn
        self.exaggeration_epochs = tsne.early_exaggeration_epochs
        self.exaggeration_value = tsne.early_exaggeration_value
        self.shuffle = shuffle
        self.lr = lr
        self.optimizer = optimizer
        self.reset_exaggeration_status()

    def reset_exaggeration_status(self):
        """
        Reset exaggeration status based on the number of exaggeration epochs.
        """
        self.has_exaggeration_ended = True if self.exaggeration_epochs == 0 else False

    def training_step(
        self,
        batch: Union[
            torch.Tensor, Tuple[torch.Tensor, ...], List[Union[torch.Tensor, Any]]
        ],
        batch_idx: int,
    ):
        """
        Perform a single training step.

        Parameters
        ----------
        `batch` : `Union[torch.Tensor, Tuple[torch.Tensor, ...], List[Union[torch.Tensor, Any]]]`
            Input batch.
        `batch_idx` : `int`
            Index of the current batch.

        Returns
        -------
        `Dict[str, torch.Tensor]`
            Dictionary containing the `loss` value.

        Note
        ----
        This method defines a single training step for the dimensionality reduction model. It computes the loss using
        the model's `logits` and the conditional probability matrix `_P_batch`.
        """
        x = batch[0]
        _P_batch = self.P_current[
            batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
        ]

        if self.shuffle:
            p_idxs = torch.randperm(x.shape[0])
            x = x[p_idxs]
            _P_batch = _P_batch[p_idxs, :]
            _P_batch = _P_batch[:, p_idxs]

        logits = self.model(x)
        loss = self.loss_fn(
            logits,
            _P_batch,
            {"device": self.tsne.device, "batch_size": self.batch_size},
        )
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def validation_step(
        self,
        batch: Union[
            torch.Tensor, Tuple[torch.Tensor, ...], List[Union[torch.Tensor, Any]]
        ],
        batch_idx: int,
        dataloader_idx: Union[int, None] = None,
    ):
        """
        Perform a single validation step.

        Parameters
        ----------
        `batch` : `Union[torch.Tensor, Tuple[torch.Tensor, ...], List[Union[torch.Tensor, Any]]]`
            Input batch.
        `batch_idx`
            Index of the current batch.
        `dataloader_idx` : optional
            Index of the dataloader

        Returns
        -------
        `Dict[str, torch.Tensor]`
            Dictionary containing the `loss` value.

        Note
        ----
        This method defines a single validation step for the dimensionality reduction model. It computes the loss using
        the model's `logits` and the conditional probability matrix `_P_batch`.
        """
        x = batch[0]
        if dataloader_idx is not None:
            _P_batch = self.val_P[dataloader_idx][
                batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
            ]
        else:
            _P_batch = self.val_P[0][
                batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
            ]
        logits = self.model(x)
        loss = self.loss_fn(
            logits,
            _P_batch,
            {"device": self.tsne.device, "batch_size": self.batch_size},
        )
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def _set_optimizer(
        self, optimizer: str, optimizer_params: dict
    ) -> torch.optim.Optimizer:
        """
        Set the optimizer based on the provided string.

        Parameters
        ----------
        `optimizer` : `str`
            String indicating the desired optimizer.
        `optimizer_params` : `dict`
            Dictionary containing optimizer parameters.

        Returns
        -------
        `torch.optim.Optimizer`
            Initialized optimizer.

        Note
        ----
        This method initializes and returns the desired optimizer based on the provided string.
        """
        if optimizer == "adam":
            return optim.Adam(self.model.parameters(), **optimizer_params)
        elif optimizer == "sgd":
            return optim.SGD(self.model.parameters(), **optimizer_params)
        elif optimizer == "rmsprop":
            return optim.RMSprop(self.model.parameters(), **optimizer_params)
        else:
            raise ValueError("Unknown optimizer")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.

        Returns
        -------
        `torch.optim.Optimizer`
            Configured optimizer.

        Note
        ----
        This method configures and returns the optimizer for training based on the specified parameters.
        """
        return self._set_optimizer(self.optimizer, {"lr": self.lr})

    def on_train_start(self) -> None:
        """
        Perform actions at the beginning of the training process.

        Note
        ----
        This method is called at the start of the training process and calculates the joint
        probability matrix P based on the training dataloader.
        """
        if not hasattr(self, "P"):
            self.P = self.tsne._calculate_P(self.trainer.train_dataloader)

    def on_train_epoch_start(self) -> None:
        """
        Perform actions at the start of each training epoch.

        Note
        ----
        This method is called at the start of each training epoch. If exaggeration is enabled and has
        not ended, it modifies the joint probability matrix for the current epoch.
        """
        if self.current_epoch > 0 and self.has_exaggeration_ended:
            return
        if (
            self.exaggeration_epochs > 0
            and self.current_epoch < self.exaggeration_epochs
        ):
            if not hasattr(self, "P_multiplied"):
                self.P_multiplied = self.P.clone()
                self.P_multiplied *= self.exaggeration_value
            self.P_current = self.P_multiplied
        else:
            self.P_current = self.P
            self.has_exaggeration_ended = True

    def on_train_epoch_end(self) -> None:
        """
        Perform actions at the end of each training epoch.

        Note
        ----
        This method is called at the end of each training epoch. If exaggeration has ended and
        P_multiplied exists, it is deleted to free up memory.
        """
        if hasattr(self, "P_multiplied") and self.has_exaggeration_ended:
            del self.P_multiplied

    def on_validation_start(self) -> None:
        """
        Perform actions at the beginning of the validation process.

        Note
        ----
        This method is called at the start of the validation process and calculates the joint
        probability matrix P for each validation dataloader.
        """
        if not hasattr(self, "val_P"):
            self.val_P = [
                self.tsne._calculate_P(loader)
                for loader in self.trainer.val_dataloaders
            ]

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """
        Perform a single step during the prediction process.

        Parameters
        ----------
        `batch`
            Input batch.
        `batch_idx`
            Index of the current batch.
        `dataloader_idx` : optional
            Index of the dataloader

        Returns
        -------
        `torch.Tensor`
            Model predictions.

        Note
        ----
        This method is called during the prediction process and returns the model's predictions for the input batch.
        """
        return self.model(batch[0])
