import typing as tp

import lightning as L
import torch
from audiomanifolds.embeddings.interfaces.base import AudioEmbedder
from torch import nn

from .invertibles import Identity, LinearTriangular, PlanarFlow, RadialFlow


def _build_disentangler(
    disentangler_type: tp.Literal["identity", "triangular", "radialflow", "planarflow"],
    disentangler_num_layers: int,
    disentangler_nonlinearity: tp.Literal["identity", "tanh", "leaky_relu", "sigmoid"],
    embedding_dim: int,
) -> nn.Module:
    non_linearities = {
        "identity": nn.Identity,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
    }
    if disentangler_type == "identity":
        return Identity()
    elif disentangler_type == "triangular":
        layers = [LinearTriangular(embedding_dim)]
        for _ in range(disentangler_num_layers - 1):
            layers.append(non_linearities[disentangler_nonlinearity]())
            layers.append(LinearTriangular(embedding_dim))

        return nn.Sequential(*layers)
    elif disentangler_type == "radialflow":
        layers = [RadialFlow(embedding_dim)]
        for _ in range(disentangler_num_layers - 1):
            layers.append(non_linearities[disentangler_nonlinearity]())
            layers.append(RadialFlow(embedding_dim))
        return nn.Sequential(*layers)
    elif disentangler_type == "planarflow":
        layers = [PlanarFlow(embedding_dim)]
        for _ in range(disentangler_num_layers - 1):
            layers.append(non_linearities[disentangler_nonlinearity]())
            layers.append(PlanarFlow(embedding_dim))
        return nn.Sequential(*layers)
    else:
        raise ValueError(f"Unsupported disentangler type: {disentangler_type}")


class Disentangler(L.LightningModule):
    def __init__(
        self,
        encoder: AudioEmbedder,
        disentangler_type: tp.Literal[
            "identity", "triangular", "radialflow", "planarflow"
        ],
        disentangler_num_layers: int,
        disentangler_nonlinearity: tp.Literal["identity",],
        dimensions_per_aug: list[int],
    ):
        """

        Args:
            encoder (nn.Module): Module that extracts representations from input audio
            dimensions_per_aug (list[int]): Number of dimensions to allocate for each effect.
        """

        super().__init__()
        self.dimensions_per_aug = dimensions_per_aug
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.encoder_dim = self.encoder.embedding_dim
        self.encoder = None
        if self.encoder_dim is None:
            raise ValueError("Encoder must have a defined embedding dimension.")
        self.encoder_dim: int  # for the type checker

        self.disentangler: nn.Module = _build_disentangler(
            disentangler_type,
            disentangler_num_layers,
            disentangler_nonlinearity,
            self.encoder_dim,
        )

    def _make_mask(self, aug_index: int) -> torch.Tensor:
        """
        Creates a mask for the given augmentation index.

        Args:
            aug_index (int): Index of the augmentation for which to create the mask.
        Returns:
            torch.Tensor: A binary mask of shape (encoder_dim,) where the dimensions allocated for the specified augmentation are set to 0 and the rest are set to 1.
        """
        mask = torch.ones(self.encoder_dim)
        start_dim = sum(self.dimensions_per_aug[:aug_index])
        end_dim = start_dim + self.dimensions_per_aug[aug_index]
        mask[start_dim:end_dim] = 0
        return mask

    def _make_multiple_masks(self, aug_indices: list[int]) -> torch.Tensor:
        """
        Creates a combined mask for the given augmentation indices.

        Args:
            aug_indices (list[int]): Indices of the augmentations for which to create the mask.

        Returns:
            torch.Tensor: A binary mask of shape (num_masks, encoder_dim,) where the dimensions allocated for the specified augmentations are set to 0 and the rest are set to 1.
        """
        mask = torch.ones(len(aug_indices), self.encoder_dim)
        for i, aug_index in enumerate(aug_indices):
            start_dim = sum(self.dimensions_per_aug[:aug_index])
            end_dim = start_dim + self.dimensions_per_aug[aug_index]
            mask[i, start_dim:end_dim] = 0
        return mask

    def forward(self, x: tuple[torch.Tensor, float]) -> torch.Tensor:
        """Computes a representation for the provided audio and returns a disentangled version of it

        Args:
            x (tuple[torch.Tensor, float]): A tuple containing the audio tensor and the sample rate. The audio tensor should have shape (*batch_size, num_samples).

        Returns:
            torch.Tensor: A disentangled representation of the input audio. (*batch_size, embedding_dim)
        """
        raise NotImplementedError("Delete the line setting self.encoder to None")
        embedding: torch.Tensor = self.encoder(x)
        disentangled_embedding = self.disentangler(embedding)
        return disentangled_embedding

    def training_step(self, batch: dict[str, torch.Tensor | list[int]], *args):
        x = batch["embedding"]  # (batch_size, num_augs=2, embedding_dim)
        augmentation_indices: list[int] = batch["augmentation_index"]  # (batch_size,)
        x_d = self.disentangler(x)
        masks = self._make_multiple_masks(augmentation_indices).to(
            x.device
        )  # (batch_size, embedding_dim)
        losses = ((x_d[..., 1, :] - x_d[..., 0, :]) ** 2) * masks
        losses = losses.sum(dim=-1)
        self.log(
            "train_loss",
            losses.mean(),
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=len(batch["embedding"]),
        )
        return losses.mean()

    def validation_step(self, batch: dict[str, torch.Tensor | list[int]], *args):
        x = batch["embedding"]
        augmentation_indices: list[int] = batch["augmentation_index"]
        x_d = self.disentangler(x)
        masks = self._make_multiple_masks(augmentation_indices).to(x.device)
        losses = ((x_d[..., 1, :] - x_d[..., 0, :]) ** 2) * masks
        losses = losses.sum(dim=-1)
        self.log(
            "val_loss",
            losses.mean(),
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=len(batch["embedding"]),
        )

        in_mask_variance = ((x_d[..., 1, :] - x_d[..., 0, :]) ** 2) * (1 - masks)
        in_mask_variance = in_mask_variance.sum(dim=-1)
        self.log(
            "val_in_mask_variance",
            in_mask_variance.mean(),
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=len(batch["embedding"]),
        )

        return losses.mean()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.disentangler.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=2
        )
        return {"optimizer": opt, "lr_scheduler": sched, "monitor": "val_loss"}

    def predict_step(self, batch: dict[str, torch.Tensor | list[int]], *args):
        x = batch["embedding"]
        x_d = self.disentangler(x)
        return {
            "original_embedding": x,
            "disentangled_embedding": x_d,
            "augmentation_index": batch["augmentation_index"],
        }
