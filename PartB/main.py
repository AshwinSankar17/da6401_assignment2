import os
import torch
import wandb
import random
import numpy as np

from model import FineTuneModel
from argparse import ArgumentParser
import pytorch_lightning as pl

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
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Set environment variable for potential other libraries
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Print confirmation message
    print(f"Random seed set to {seed}")

def main(args):
    nn = FineTuneModel(
        args.batch_size,
        args.learning_rate,
        freeze_block=args.freeze_block,
        num_classes=args.num_classes,
        augmentation=args.augmentation,
        dataset_path=args.dataset_path,
        num_workers=args.num_workers,
    )
    
    run_name = (
        f"bs{args.batch_size}-lr{args.learning_rate:.0e}"
        f"-freeze{args.freeze_block}-aug{'T' if args.augmentation else 'F'}"
    )

    logger = pl.loggers.WandbLogger(project="DA6401-Assignment2", name=run_name)
    logger.watch(nn)
    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor='val/acc', mode='max', save_top_k=1, save_last=True),
        pl.callbacks.EarlyStopping(monitor='val/acc', mode='max', patience=5, verbose=True)
    ]

    trainer = pl.Trainer(accelerator="gpu", max_epochs=args.n_epochs, strategy="ddp", devices=-1, precision='bf16-mixed', logger=logger, callbacks=callbacks, log_every_n_steps=10)
    trainer.fit(nn)
    if args.test:
        trainer = pl.Trainer(devices=1, accelerator="gpu", precision='bf16-mixed', logger=logger)
        trainer.test(nn)

if __name__ == "__main__":
    parser = ArgumentParser(description='Fine-tune Model Hyperparameters')

    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for dataloader')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument("--freeze_block", type=str, default="layer4", help="Freeze up to which block.")
    parser.add_argument('--num_classes', type=int, default=10, help='Number of output classes')
    parser.add_argument('--augmentation', type=bool, default=True, help='Use data augmentation')
    parser.add_argument('--dataset_path', type=str, default='../inaturalist_12K/', help='Path to dataset')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for dataloader')
    parser.add_argument('--test', action="store_true", help='Test mode')

    args = parser.parse_args()

    seed_everything()

    main(args)