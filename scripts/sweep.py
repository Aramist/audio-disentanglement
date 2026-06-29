import argparse
import dataclasses
import os
import typing as tp
from pathlib import Path

import lightning as L
import lightning.pytorch.loggers
import numpy as np
import torch
import wandb
from audiomanifolds.embeddings import (
    AudioEmbedder,
    CLAPAudioEmbedder,
    EncodecEmbedder,
    PannEmbedder,
)
from lightning.pytorch import callbacks

from audio_disentanglement.dataloading import load_datamodule
from audio_disentanglement.disentangle import Disentangler
from audio_disentanglement.util import ConfigNamespace

DEFAULT_CONFIG = ConfigNamespace()


def rand_ascii(length: int = 8) -> str:
    """Generate a random ASCII string of the given length."""
    return "".join(
        np.random.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), size=length)
    )


def retrieve_encoder(encoder_name: str) -> AudioEmbedder:
    if encoder_name == "CLAP":
        # return CLAPAudioEmbedder.from_pretrained()
        return CLAPAudioEmbedder()
    elif encoder_name == "PANN":
        # return PannEmbedder.from_pretrained()
        return PannEmbedder()
    elif encoder_name == "encodec":
        # return EncodecEmbedder.from_pretrained()
        return EncodecEmbedder()
    else:
        raise ValueError(f"Unsupported encoder name: {encoder_name}")


def make_trainer(config: ConfigNamespace, save_directory: Path, **kwargs) -> L.Trainer:
    return L.Trainer(
        max_steps=config.num_optimization_steps,
        default_root_dir=save_directory,
        callbacks=[
            # Save the best model based on validation accuracy
            callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=False,
                verbose=False,
            ),
            # End training if validation accuracy does not improve
            callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5),
            # End training if weights explode
            callbacks.EarlyStopping(
                monitor="train_loss",
                check_finite=True,
                mode="min",
                verbose=False,
                patience=100000,  # only looking to stop if non-finite
            ),
        ],
        gradient_clip_val=1.0 if config.clip_gradients else 0.0,
        num_sanity_val_steps=0,
        **kwargs,
    )


def train_model(
    encoding_model_name: str,
    data_dir: Path,
    save_dir: Path,
    experiment: wandb.Run,
    *,
    sweep_name: str = "audio-disentangle-sweep",
):
    logger = lightning.pytorch.loggers.WandbLogger(
        project=sweep_name,
        save_dir=save_dir / "logs",
        log_model=False,
        experiment=experiment,
    )
    model_config_dict = dataclasses.asdict(DEFAULT_CONFIG)
    model_config_dict.update(experiment.config)  # Load wandb sweep hypers
    model_config = ConfigNamespace.from_config_dict(model_config_dict)
    print(f"Training model with config: {model_config}")

    trainer = make_trainer(model_config, save_directory=save_dir, logger=logger)

    datamodule = load_datamodule(
        data_dir, model_name=encoding_model_name, batch_size=model_config.batch_size
    )
    aug_names = datamodule.ordered_aug_names
    dims_per_aug = [model_config.dims_per_aug[aug_name] for aug_name in aug_names]
    model = Disentangler(
        encoder=retrieve_encoder(encoding_model_name),
        disentangler_type=model_config.disentangler_type,
        disentangler_num_layers=model_config.disentangler_num_layers,
        disentangler_nonlinearity=model_config.disentangler_nonlinearity,
        dimensions_per_aug=dims_per_aug,
    )
    trainer.fit(model, datamodule=datamodule)
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a sweep over invertable models to disentangle audio embeddings."
    )
    parser.add_argument(
        "--sweep_name",
        type=str,
        default="audio-disentangle-sweep",
        help="Name of the wandb sweep.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to the input data for training (directory).",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path(".") / "sweep_runs",
        help="Directory to save the sweep runs.",
    )
    parser.add_argument("--encoder", type=str, choices=["CLAP", "PANN", "encodec"])
    args = parser.parse_args()
    if args.data is None:
        raise ValueError("Must provide --data")

    args = parser.parse_args()

    while (save_path := args.save_dir / rand_ascii()).exists():
        pass
    save_path.mkdir(parents=True, exist_ok=False)

    exp = wandb.init(
        dir=str(save_path),
    )
    train_model(
        encoding_model_name=args.encoder,
        data_dir=args.data,
        save_dir=save_path,
        experiment=exp,
        sweep_name=args.sweep_name,
    )
