import argparse
import os
import typing as tp
from pathlib import Path

import lightning as L
import lightning.pytorch.loggers
import numpy as np
import torch
from audiomanifolds.embeddings import (
    AudioEmbedder,
    CLAPAudioEmbedder,
    EncodecEmbedder,
    PannEmbedder,
)
from lightning.pytorch import callbacks

from .dataloading import load_datamodule
from .disentangle import Disentangler
from .util import ConfigNamespace

DEFAULT_CONFIG = ConfigNamespace()


def retrieve_encoder(encoder_name: str) -> AudioEmbedder:
    if encoder_name == "CLAP":
        return CLAPAudioEmbedder.from_pretrained()
    elif encoder_name == "PANN":
        return PannEmbedder.from_pretrained()
    elif encoder_name == "encodec":
        return EncodecEmbedder.from_pretrained()
    else:
        raise ValueError(f"Unsupported encoder name: {encoder_name}")


def make_trainer(config: ConfigNamespace, save_directory: Path, **kwargs) -> L.Trainer:
    num_nodes = int(os.getenv("SLURM_NNODES", 1))
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
        num_sanity_val_steps=2,
        **kwargs,
    )


def train(
    encoding_model_name: str,
    data_dir: Path,
    save_dir: Path,
    run_name: str | None = None,
    config: ConfigNamespace = DEFAULT_CONFIG,
):
    logger = lightning.pytorch.loggers.WandbLogger(
        project="audio-disentangle", name=run_name, save_dir=save_dir / "logs"
    )
    datamodule = load_datamodule(
        data_dir,
        model_name=encoding_model_name,
        batch_size=config.batch_size,
    )
    aug_names = datamodule.ordered_aug_names
    dims_per_aug = [config.dims_per_aug[aug_name] for aug_name in aug_names]
    model = Disentangler(
        encoder=retrieve_encoder(encoding_model_name),
        disentangler_type="triangular",
        dimensions_per_aug=dims_per_aug,
    )

    trainer = make_trainer(config, save_directory=save_dir, logger=logger)
    trainer.fit(model, datamodule=datamodule)
    return trainer


def infer(
    trainer: L.Trainer,
    encoding_model_name: str,
    data_dir: Path,
    save_dir: Path,
    config: ConfigNamespace = DEFAULT_CONFIG,
):
    datamodule = load_datamodule(
        data_dir,
        model_name=encoding_model_name,
        batch_size=config.batch_size,
    )

    ckpt_path = None
    if trainer.checkpoint_callback:
        ckpt_path = trainer.checkpoint_callback.best_model_path
    if not ckpt_path:
        ckpt_path = find_existing_checkpoint(save_dir)
    aug_names = datamodule.ordered_aug_names
    dims_per_aug = [config.dims_per_aug[aug_name] for aug_name in aug_names]
    model = Disentangler(
        encoder=retrieve_encoder(encoding_model_name),
        disentangler_type="triangular",
        dimensions_per_aug=dims_per_aug,
    )
    # I don't trust load_from_checkpoint
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=True)

    # Sanity check: run validation first to make sure the model is working and performant
    val_score = trainer.validate(model, datamodule=datamodule)
    print(f"(Sanity check) Validation loss before inference: {val_score}")

    preds: tp.Sequence[dict[str, torch.Tensor]] = trainer.predict(
        model, datamodule=datamodule
    )

    output_path = save_dir / f"{encoding_model_name}_disentangled_embeddings.npz"
    aug_names = list(map(str.encode, datamodule.ordered_aug_names))
    np.savez(
        output_path,
        original_embeddings=torch.cat([pred["original_embedding"] for pred in preds])
        .cpu()
        .numpy(),
        disentangled_embeddings=torch.cat(
            [pred["disentangled_embedding"] for pred in preds]
        )
        .cpu()
        .numpy(),
        augmentation_indices=np.concatenate(
            [pred["augmentation_index"] for pred in preds]
        ),
        augmentation_names=np.array(aug_names),
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
    args = parser.parse_args()
    if args.data is None:
        raise ValueError("Must provide --data")
    if args.save_path is None:
        raise ValueError("Must provide --save-path")
    if args.run_name is None:
        args.run_name = f"{args.save_path.stem}_{int(torch.randn(1).item() * 1e6)}"

    if (ckpt_path := find_existing_checkpoint(args.save_path)) is not None:
        print(
            f"Found existing checkpoint at {ckpt_path}. Skipping training and proceeding to inference."
        )

        trainer = make_trainer(
            DEFAULT_CONFIG, save_directory=args.save_path, logger=None
        )
    else:
        trainer = train(args.encoder, args.data, args.save_path, run_name=args.run_name)
    infer(trainer, args.encoder, args.data, args.save_path)
