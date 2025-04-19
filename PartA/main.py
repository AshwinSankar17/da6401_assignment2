import os
import json
import torch
import wandb
import random
import numpy as np


import pytorch_lightning as pl
from model import NeuralNetwork
from argparse import ArgumentParser

from plot import plot_predictions


def seed_everything(seed: int = 42):
    """
    Set the random seed for reproducibility across different libraries.

    Args:
        seed (int): Seed value to use. Default is 42.
    """
    # Set Python's random seed
    random.seed(seed)

    # Set NumPy's random seed
    np.random.seed(seed)

    # Set PyTorch's random seed
    torch.manual_seed(seed)

    # Set CUDA's random seed if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Set environment variable for potential other libraries
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Print confirmation message
    print(f"Random seed set to {seed}")


def main(args):
    """
        Train a neural network using the specified hyperparameters.
    """
    nn = NeuralNetwork(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        learning_rate=args.learning_rate,
        batch_norm=args.batch_norm,
        activation=args.activation,
        kernel_strategy=args.filter_strategy,
        dropout=args.dropout,
        num_classes=args.num_classes,
        hidden_size=args.hidden_size,
        dataset_path=args.dataset_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        augmentation=args.augmentation,
    )
    # nn = NeuralNetwork.load_from_checkpoint(
    #     "./DA6401-Assignment2/93b2wolo/checkpoints/last.ckpt",
    #     in_channels=args.in_channels,
    #     out_channels=args.out_channels,
    #     kernel_size=args.kernel_size,
    #     stride=args.stride,
    #     padding=args.padding,
    #     learning_rate=args.learning_rate,
    #     batch_norm=args.batch_norm,
    #     activation=args.activation,
    #     kernel_strategy=args.filter_strategy,
    #     dropout=args.dropout,
    #     num_classes=args.num_classes,
    #     hidden_size=args.hidden_size,
    #     dataset_path=args.dataset_path,
    #     num_workers=args.num_workers,
    #     batch_size=args.batch_size,
    #     augmentation=args.augmentation,
    # )
    logger = pl.loggers.WandbLogger(project="DA6401-Assignment2")
    logger.watch(nn)
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val/acc", mode="max", save_top_k=3, save_last=True
        )
    ]
    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        devices=1,
        precision="bf16-mixed",
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(nn)
    if args.test:
        trainer.test(nn)
    
    plot_predictions(nn, nn.test_dataloader(), nn.test_dataset.classes)


def sweep(args):
    """
        Run sweep with the provided sweep configuration.
    """
    wandb.init()
    config = wandb.config
    logger = pl.loggers.WandbLogger()

    wandb.run.name = (
        f"{config.activation}_act-"
        f"{config.out_channels}oc-"
        f"{config.kernel_size}k-"
        # f"{config.stride}s-"
        f"{config.filter_strategy}_fs-"
        f"{config.batch_size}bs-"
        f"{'bn' if config.batch_norm else 'no_bn'}-"
        f"{'aug' if config.augmentation else 'no_aug'}"
        f"{config.n_epochs}epochs"
    )

    nn = NeuralNetwork(
        config.in_channels,
        config.out_channels,
        config.kernel_size,
        1,
        0,
        learning_rate=config.learning_rate,
        batch_norm=config.batch_norm,
        activation=config.activation,
        kernel_strategy=config.filter_strategy,
        dropout=config.dropout,
        num_classes=args.num_classes,
        hidden_size=config.hidden_size,
        dataset_path=args.dataset_path,
        num_workers=args.num_workers,
        batch_size=config.batch_size,
        augmentation=config.augmentation,
    )

    logger.watch(nn)
    # nn = torch.compile(nn)

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        devices=1,
        precision="bf16-mixed",
        logger=logger,
        log_every_n_steps=10,
    )
    trainer.fit(nn)


if __name__ == "__main__":
    parser = ArgumentParser(description="Neural Network Hyperparameters")

    parser.add_argument(
        "--n_epochs", type=int, default=20, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--in_channels", type=int, default=3, help="Number of input channels"
    )
    parser.add_argument(
        "--out_channels", type=int, default=64, help="Number of output channels"
    )
    parser.add_argument(
        "--kernel_size", type=int, default=7, help="Size of convolution kernel"
    )
    parser.add_argument("--stride", type=int, default=1, help="Stride of convolution")
    parser.add_argument("--padding", type=int, default=0, help="Padding of convolution")
    parser.add_argument(
        "--batch_norm", type=bool, default=True, help="Use batch normalization"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="mish",
        choices=["relu", "gelu", "silu", "mish"],
        help="Activation function",
    )
    parser.add_argument(
        "--filter_strategy",
        type=str,
        default="double",
        choices=["same", "double", "half"],
        help="Filter strategy",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout probability"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.00020112284928963905, help="Learning rate"
    )
    parser.add_argument(
        "--num_classes", type=int, default=10, help="Number of classes in output"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=512, help="Size of hidden layer"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../inaturalist_12K/",
        help="Path to dataset",
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for dataloader"
    )
    parser.add_argument(
        "--augmentation", type=bool, default=False, help="Use data augmentation"
    )
    parser.add_argument("--test", action="store_true", help="Flag to indicate whether Test mode should be activated")
    parser.add_argument(
        "--do_sweep",
        action="store_true",
        help="Flag to indicate if a sweep should be run",
    )
    parser.add_argument(
        "--sweep_config_path",
        type=str,
        default="sweep_config.json",
        help="Path to the W&B sweep config file",
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
        help="Sweep id of already started sweep",
    )

    args = parser.parse_args()
    seed_everything()
    if not args.do_sweep:
        main(args)
    else:
        assert (
            os.path.exists(args.sweep_config_path) 
            and os.path.isfile(args.sweep_config_path)
        ), f"{os.path.exists(args.sweep_config_path)} or {os.path.isfile(args.sweep_config_path)}"
        sweep_config = json.load(open(args.sweep_config_path))

        sweep_id = args.sweep_id
        if not sweep_id:
            sweep_id = wandb.sweep(sweep_config, project="DA6401-Assignment2")

        wandb.agent(sweep_id=sweep_id, function=lambda: sweep(args), count=100, project="DA6401-Assignment2")
