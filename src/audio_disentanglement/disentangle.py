import typing as tp

import lightning as L
import numpy as np
import torch
from torch import nn

from .invertibles import (
    CouplingFlow,
    FullRank,
    Identity,
    Invertible,
    LinearTriangular,
    PlanarFlow,
    RadialFlow,
    RandPerm,
)
from .util import ConfigNamespace

torch.set_float32_matmul_precision("medium")


def _build_disentangler(
    config: ConfigNamespace,
    embedding_dim: int,
) -> nn.ModuleList:
    disentangler_type = config.disentangler_type
    disentangler_num_layers = config.disentangler_num_layers
    partition_strategy = config.coupling_flow_partition_strategy
    coupling_flow_num_layers = config.coupling_flow_num_hidden_layers
    coupling_flow_hidden_layer_size = config.coupling_flow_hidden_layer_size

    layers = []
    if disentangler_type == "identity":
        layers.append(Identity())
    elif disentangler_type == "triangular":
        layers.append(LinearTriangular(embedding_dim))
        for _ in range(disentangler_num_layers - 1):
            layers.append(LinearTriangular(embedding_dim))
    elif disentangler_type == "radialflow":
        layers.append(RadialFlow(embedding_dim))
        for _ in range(disentangler_num_layers - 1):
            layers.append(FullRank(embedding_dim))
            layers.append(RadialFlow(embedding_dim))
    elif disentangler_type == "planarflow":
        layers.append(PlanarFlow(embedding_dim))
        for _ in range(disentangler_num_layers - 1):
            layers.append(FullRank(embedding_dim))
            layers.append(PlanarFlow(embedding_dim))
    elif disentangler_type == "couplingflow":
        embedding_dim_num_bits = (
            int(np.ceil(np.log2(embedding_dim))) - 1
        )  # bits that vary in 0..<embedding_dim
        layers = [
            CouplingFlow(
                embedding_dim,  # equivalent to partitioning on the highest bit 1<<embedding_dim_num_bits
                num_hidden_layers=coupling_flow_num_layers,
                hidden_layer_size=coupling_flow_hidden_layer_size,
            )
        ]
        for layer_idx in range(1, disentangler_num_layers):
            partition = None
            if partition_strategy == "random":
                layers.append(RandPerm(embedding_dim))
            elif partition_strategy == "haar":
                cycle_size = embedding_dim_num_bits * 2
                i = layer_idx % cycle_size
                shift_amount = embedding_dim_num_bits - i // 2
                should_flip = i % 2 == 1
                partition = (np.arange(embedding_dim) & (1 << shift_amount)).astype(
                    bool
                )
                partition = ~partition if should_flip else partition
            layers.append(
                CouplingFlow(
                    embedding_dim,
                    partition=partition,
                    num_hidden_layers=coupling_flow_num_layers,
                    hidden_layer_size=coupling_flow_hidden_layer_size,
                )
            )
    else:
        raise ValueError(f"Unsupported disentangler type: {disentangler_type}")

    return nn.ModuleList(layers)


class Disentangler(L.LightningModule):
    def __init__(
        self,
        config: ConfigNamespace,
        encoder_dim: int,
    ):
        """

        Args:
            encoder (nn.Module): Module that extracts representations from input audio
            dimensions_per_aug (list[int]): Number of dimensions to allocate for each effect.
        """

        super().__init__()
        self.config = config
        self.dimensions_per_aug = [
            config.dims_per_aug[aug_name] for aug_name in config.augmentations
        ]
        self.encoder_dim = encoder_dim
        self.encoder = None
        if self.encoder_dim is None:
            raise ValueError("Encoder must have a defined embedding dimension.")
        self.encoder_dim: int  # for the type checker

        self.disentangler: nn.ModuleList = _build_disentangler(
            config=config,
            embedding_dim=self.encoder_dim,
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

    def disentangle(
        self, x: torch.Tensor, return_jacobians: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        jacobians = []
        for layer in self.disentangler:
            if return_jacobians and hasattr(layer, "log_det_jacobian"):
                jacobians.append(layer.log_det_jacobian(x))
            x = layer(x)
        if return_jacobians:
            return x, torch.stack(jacobians, dim=0)
        return x

    def training_step(self, batch: dict[str, torch.Tensor | list[int]], *args):
        x = batch["embedding"]  # (batch_size, num_augs=2, embedding_dim)
        augmentation_indices: list[int] = batch["augmentation_index"]  # (batch_size,)
        x_d = self.disentangle(x, return_jacobians=False)
        masks = self._make_multiple_masks(augmentation_indices).to(
            x.device
        )  # (batch_size, embedding_dim)
        # variance
        losses = (x_d * masks[:, None, :]).var(dim=1).sum(dim=-1) / masks.sum(
            dim=-1
        )  # (batch_size,)
        # losses = ((x_d[..., 1, :] - x_d[..., 0, :]) ** 2) * masks
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
        x_d, det_jacobians = self.disentangle(x, return_jacobians=True)
        # Sum across layers to get log determinant of the entire transformation's jacobian
        # Average across samples for reporting
        det_jacobians = det_jacobians.sum(dim=0).mean()
        self.log(
            "log_det_jacobian",
            det_jacobians,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        masks = self._make_multiple_masks(augmentation_indices).to(x.device)
        losses = ((x_d[..., 1, :] - x_d[..., 0, :]) ** 2) * masks
        losses = losses.mean(dim=-1)
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
            "permitted_variance",
            in_mask_variance.mean(),
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=len(batch["embedding"]),
        )

        # Log the norm of the weights over time
        for i, layer in enumerate(self.disentangler):
            if isinstance(layer, Invertible):
                sum_params = sum(
                    torch.square(p).sum().cpu().item() for p in layer.parameters()
                )
                num_params = sum(p.numel() for p in layer.parameters())

                if num_params == 0:
                    continue  # Skip logging for layers with no parameters

                layer_type = type(layer).__name__
                self.log(
                    f"{layer_type}_{i}_weight_norm",
                    np.sqrt(sum_params / num_params),
                    on_epoch=True,
                    on_step=False,
                    sync_dist=True,
                )

        return losses.mean()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.disentangler.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10
        )
        return {"optimizer": opt, "lr_scheduler": sched, "monitor": "val_loss"}

    def predict_step(self, batch: dict[str, torch.Tensor | list[int]], *args):
        x = batch["embedding"]
        x_d = self.disentangle(x, return_jacobians=False)
        return {
            "original_embedding": x.cpu(),
            "disentangled_embedding": x_d.cpu(),
            "augmentation_index": batch["augmentation_index"],
        }
