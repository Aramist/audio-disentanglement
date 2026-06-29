from functools import partial
from pathlib import Path

import h5py
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split


def collate_fn(batch: dict[str, np.ndarray], ordered_aug_names: list[str]):
    embeddings = torch.from_numpy(
        np.stack([item["embedding"] for item in batch], axis=0)
    ).float()
    augmentation_names = [item["augmentation_name"] for item in batch]
    aug_indices = [ordered_aug_names.index(name) for name in augmentation_names]
    return {"embedding": embeddings, "augmentation_index": aug_indices}


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        hdf5_path: Path,
        augmentation_name: str,
        inference_mode: bool = False,
    ):
        self.hdf5_path = hdf5_path
        self.augmentation_name = augmentation_name
        self.inference_mode = inference_mode
        self.rng = np.random.default_rng()
        self.hdf5_handle = None

        with h5py.File(self.hdf5_path, "r") as f:
            # "embedding" (num_sounds, num_augs_per_sound, embedding_dim)
            self.length = f["embedding"].shape[0]
            self.augs_per_sound = f["embedding"].shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.hdf5_handle is None:
            self.hdf5_handle = h5py.File(self.hdf5_path, "r")

        if self.inference_mode:
            embedding = self.hdf5_handle["embedding"][
                idx
            ]  # (num_augs_per_sound, embedding_dim)
        else:
            rand_indices = self.rng.choice(self.augs_per_sound, size=2, replace=False)
            # hdf5 doesn't like advanced indexing with non-sorted arrays
            embedding = np.stack(
                [
                    self.hdf5_handle["embedding"][idx, rand_indices[0]],
                    self.hdf5_handle["embedding"][idx, rand_indices[1]],
                ],
                axis=0,
            )  # (2, embedding_dim)

        return {"embedding": embedding, "augmentation_name": self.augmentation_name}

    def __del__(self):
        if self.hdf5_handle is not None:
            self.hdf5_handle.close()


class EmbeddingDataModule(LightningDataModule):
    def __init__(
        self,
        hdf5_paths: dict[str, Path],
        batch_size: int = 32,
        num_workers: int = 4,
        random_seed: int = 42,
    ):
        super().__init__()
        self.hdf5_paths = hdf5_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.ordered_aug_names = list(hdf5_paths.keys())

    def setup(self, stage=None):
        self.datasets = {
            aug_name: EmbeddingDataset(
                hdf5_path=path,
                augmentation_name=aug_name,
                inference_mode=False,
            )
            for aug_name, path in self.hdf5_paths.items()
        }

        rng = torch.Generator()
        rng.manual_seed(self.random_seed)
        train_dsets, val_dsets = [], []
        for _, dset in self.datasets.items():
            num_train = int(0.9 * len(dset))
            train, val = random_split(
                dset,
                [num_train, len(dset) - num_train],
                generator=rng,
            )
            train_dsets.append(train)
            val_dsets.append(val)

        self.train_dataset = ConcatDataset(train_dsets)
        self.val_dataset = ConcatDataset(val_dsets)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=partial(collate_fn, ordered_aug_names=self.ordered_aug_names),
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=partial(collate_fn, ordered_aug_names=self.ordered_aug_names),
            persistent_workers=True,
        )

    def predict_dataloader(self):
        self.inference_dataset = ConcatDataset(
            [
                EmbeddingDataset(
                    hdf5_path=self.hdf5_paths[aug_name],
                    augmentation_name=aug_name,
                    inference_mode=True,
                )
                for aug_name in self.ordered_aug_names
            ]
        )
        return DataLoader(
            self.inference_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=partial(collate_fn, ordered_aug_names=self.ordered_aug_names),
        )


def load_datamodule(data_dir: Path, model_name: str, **kwargs) -> EmbeddingDataModule:
    fmt = "BSD10k_{model_name}_{aug_name}.h5"
    hdf5_paths = {
        aug_name: data_dir / fmt.format(model_name=model_name, aug_name=aug_name)
        for aug_name in [
            "pitch_shifting",
            "time_stretching",
            "gain",
        ]
    }
    return EmbeddingDataModule(hdf5_paths=hdf5_paths, **kwargs)
