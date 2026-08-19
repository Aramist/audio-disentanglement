import argparse
import dataclasses
import json
import os
import typing as tp
from pathlib import Path

import lightning as L
import lightning.pytorch.loggers
import numpy as np
import torch

# from audiomanifolds.embeddings import (
#     AudioEmbedder,
#     CLAPAudioEmbedder,
#     EncodecEmbedder,
#     PannEmbedder,
# )
from lightning.pytorch import callbacks
from lightning.pytorch.callbacks import BasePredictionWriter

from .dataloading import EmbeddingDataModule, load_datamodule
from .disentangle import Disentangler
from .util import ConfigNamespace

DEFAULT_CONFIG = ConfigNamespace()


class InferencWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: Path,
        ordered_aug_names: list[str],
        output_format: str = "batch_{}.npz",
    ):
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self.output_format = output_format
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ordered_aug_names = ordered_aug_names

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction: tp.Sequence[dict[str, torch.Tensor]],
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        output_path = self.output_dir / self.output_format.format(batch_idx)

        aug_names = list(map(str.encode, self.ordered_aug_names))
        np.savez(
            output_path,
            original_embeddings=prediction["original_embedding"].cpu().numpy(),
            disentangled_embeddings=prediction["disentangled_embedding"].cpu().numpy(),
            augmentation_indices=np.array(prediction["augmentation_index"]),
            augmentation_names=np.array(aug_names),
        )


def retrieve_encoder_dim(encoder_name: str) -> int:
    if encoder_name == "CLAP":
        # return CLAPAudioEmbedder()
        return 512
    elif encoder_name == "PANN":
        # return PannEmbedder()
        return 2048
    elif encoder_name == "encodec":
        # return EncodecEmbedder()
        return 128
    else:
        raise ValueError(f"Unsupported encoder name: {encoder_name}")


def make_trainer(config: ConfigNamespace, save_directory: Path, **kwargs) -> L.Trainer:
    num_nodes = int(os.getenv("SLURM_NNODES", 1))
    additional_callbacks = kwargs.get("callbacks", [])
    del kwargs["callbacks"]  # Remove callbacks from kwargs to avoid duplication
    return L.Trainer(
        max_steps=config.num_optimization_steps,
        num_nodes=num_nodes,
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
            callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=100),
            # End training if weights explode
            callbacks.EarlyStopping(
                monitor="train_loss",
                check_finite=True,
                mode="min",
                verbose=False,
                patience=100000,  # only looking to stop if non-finite
            ),
            *additional_callbacks,
        ],
        gradient_clip_val=1.0 if config.clip_gradients else 0.0,
        num_sanity_val_steps=0,
        **kwargs,
    )


def train(
    encoding_model_name: str,
    datamodule: EmbeddingDataModule,
    save_dir: Path,
    run_name: str | None = None,
    config: ConfigNamespace = DEFAULT_CONFIG,
    **kwargs,
):
    logger = lightning.pytorch.loggers.WandbLogger(
        project="audio-disentangle",
        name=run_name,
        save_dir=save_dir / "logs",
        log_model=False,
    )
    # Save model config to save_dir
    with open(save_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=4)

    model = Disentangler(
        config=config,
        encoder_dim=retrieve_encoder_dim(encoding_model_name),
    )

    trainer = make_trainer(config, save_directory=save_dir, logger=logger, **kwargs)
    trainer.fit(model, datamodule=datamodule)
    return trainer


def infer(
    trainer: L.Trainer,
    encoding_model_name: str,
    datamodule: EmbeddingDataModule,
    save_dir: Path,
    config: ConfigNamespace = DEFAULT_CONFIG,
):
    ckpt_path = None
    if trainer.checkpoint_callback:
        ckpt_path = trainer.checkpoint_callback.best_model_path
    if not ckpt_path:
        ckpt_path = find_existing_checkpoint(save_dir)

    # I don't trust load_from_checkpoint
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model = Disentangler(
        config=config,
        encoder_dim=retrieve_encoder_dim(encoding_model_name),
    )
    model.load_state_dict(state_dict, strict=True)

    # Sanity check: run validation first to make sure the model is working and performant
    val_score = trainer.validate(model, datamodule=datamodule)
    print(f"(Sanity check) Validation loss before inference: {val_score}")
    trainer.predict(
        model,
        datamodule=datamodule,
        return_predictions=False,
    )


def find_existing_checkpoint(save_dir: Path) -> Path | None:
    """Checks the save directory for existing checkpoints and returns the path to the most recent one, if it exists."""
    checkpoint_dir = save_dir
    if not checkpoint_dir.exists():
        return None
    checkpoint_files = list(checkpoint_dir.rglob("*.ckpt"))
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
    return latest_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an invertable model to disentangle audio embeddings."
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to the input data for training (directory).",
    )
    parser.add_argument("--encoder", type=str, choices=["CLAP", "PANN", "encodec"])
    parser.add_argument(
        "--save-path",
        type=Path,
        help="Directory where model checkpoints and logs will be saved.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for the training run (for logging purposes).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON configuration file for training parameters.",
    )
    args = parser.parse_args()
    if args.data is None:
        raise ValueError("Must provide --data")
    if args.save_path is None:
        raise ValueError("Must provide --save-path")
    if args.run_name is None:
        args.run_name = f"{args.save_path.stem}_{int(torch.randn(1).item() * 1e6)}"
    if args.config is not None:
        with open(args.config, "r") as ctx:
            config = json.load(ctx)
        model_config_dict = dataclasses.asdict(DEFAULT_CONFIG)
        model_config_dict.update(config)  # Use provided config to override defaults
        model_config = ConfigNamespace.from_config_dict(model_config_dict)
        print("Loaded configuration:")
        for key, value in dataclasses.asdict(model_config).items():
            print(f"  {key}: {value}")
    else:
        model_config = DEFAULT_CONFIG

    datamodule = load_datamodule(
        args.data,
        augmentation_names=model_config.augmentations,
        model_name=args.encoder,
        batch_size=model_config.batch_size,
        num_training_samples_per_sound=model_config.num_training_samples_per_sound,
    )
    writer = InferencWriter(
        output_dir=args.save_path / f"{args.encoder}_disentangled_embeddings",
        ordered_aug_names=datamodule.ordered_aug_names,
    )
    if (ckpt_path := find_existing_checkpoint(args.save_path)) is not None:
        print(
            f"Found existing checkpoint at {ckpt_path}. Skipping training and proceeding to inference."
        )

        trainer = make_trainer(
            model_config, save_directory=args.save_path, logger=None, callbacks=[writer]
        )
    else:
        trainer = train(
            args.encoder,
            datamodule,
            args.save_path,
            run_name=args.run_name,
            config=model_config,
            callbacks=[writer],
        )
    infer(trainer, args.encoder, datamodule, args.save_path, config=model_config)
