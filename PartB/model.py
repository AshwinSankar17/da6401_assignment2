import os
import math

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import random_split

import torchvision
from torchvision.models import ResNet50_Weights
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from torchmetrics.classification import MulticlassAccuracy

from typing import Any


def get_datasets(augmentation: bool = True, path: str = "../inaturalist_12K/"):
    """
    Loads and preprocesses the iNaturalist 12K dataset with optional data augmentation.

    Args:
        augmentation (bool): Whether to apply data augmentation. Defaults to True.
        path (str): Path to the root directory containing 'train' and 'val' folders. Defaults to "../inaturalist_12K/".

    Returns:
        Tuple[Dataset, Dataset, Dataset]: A tuple containing the training, validation, and test datasets.
    """
    normalize = transforms.Normalize(
        mean=[0, 0, 0], std=[1, 1, 1]
    )

    augs = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # less common in natural images
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.RandomAffine(
            degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5
        ),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    ]

    ts = [transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0), ratio=(0.75, 1.33))]
    if augmentation:
        ts += augs
    ts += [transforms.ToTensor(), normalize]

    transform = transforms.Compose(ts)

    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "val")

    dataset = ImageFolder(train_path, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_dataset = ImageFolder(test_path, transform=test_transform)

    return train_dataset, val_dataset, test_dataset


def freeze_up_to_block(model, last_block_to_freeze):
    """
    last_block_to_freeze: string - 'conv1', 'bn1', 'layer1', 'layer2', 'layer3', or 'layer4'
    Will freeze all blocks up to and including the specified block
    """
    if last_block_to_freeze is None:
        return model

    # Define the blocks in order
    block_order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]

    # Find the index of the specified block
    if last_block_to_freeze in block_order:
        max_idx = block_order.index(last_block_to_freeze)
    else:
        raise ValueError(
            f"Block {last_block_to_freeze} not found. Use one of {block_order}"
        )

    # Freeze all blocks up to and including the specified block
    blocks_to_freeze = block_order[: max_idx + 1]

    for name in blocks_to_freeze:
        if hasattr(model, name):
            for param in getattr(model, name).parameters():
                param.requires_grad = False
            # print(f"Froze {name}")

    return model


class FineTuneModel(pl.LightningModule):
    """
    A fine-tuning wrapper around ResNet-50 using PyTorch Lightning.

    This model freezes all layers up to a specified block and replaces 
    the final fully connected layer to adapt to a new classification task.

    Args:
        batch_size (int): Batch size for training and evaluation.
        learning_rate (float): Learning rate for the optimizer.
        freeze_block (str): Name of the last block to freeze in ResNet ('layer1', 'layer2', etc.).
        num_classes (int): Number of output classes for classification.
        augmentation (bool): Whether to apply data augmentation.
        dataset_path (str): Path to the dataset directory.
        num_workers (int): Number of workers for data loading.
    """
    def __init__(
        self,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        freeze_block: str = "layer4",
        num_classes: int = 10,
        augmentation: bool = True,
        dataset_path: str = "../inaturalist_12K",
        num_workers: int = 16,
    ) -> None:
        super().__init__()
        # self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.augmentation = augmentation
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model = freeze_up_to_block(
            torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
            freeze_block,
        )

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=self.num_classes).to(
            self.device
        )
        self.train_dataset, self.val_dataset, self.test_dataset = get_datasets(
            self.augmentation, self.dataset_path
        )

        self.val_mem = {"loss": [], "y_hats": [], "ys": []}
        self.test_mem = {"loss": [], "y_hats": [], "ys": []}

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, *args):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, *args) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # Store predictions and ground truth for later
        self.val_mem["loss"].append(loss)
        self.val_mem["y_hats"].append(y_hat.softmax(dim=-1))
        self.val_mem["ys"].append(y)

        return loss

    def on_validation_epoch_end(self):
        # Calculate average loss
        avg_loss = torch.tensor(self.val_mem["loss"]).mean()

        # Concatenate all predictions and labels
        y_hats = torch.cat(self.val_mem["y_hats"], dim=0)
        ys = torch.cat(self.val_mem["ys"], dim=0)

        # Calculate accuracy once on the entire validation set
        val_acc = self.accuracy(y_hats, ys)

        # Log metrics
        self.log("val/loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", val_acc, prog_bar=True, sync_dist=True)

        # Clear memory for next epoch
        self.val_mem = {"loss": [], "y_hats": [], "ys": []}

    def test_step(self, batch, *args) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # Store predictions and ground truth for later
        self.test_mem["loss"].append(loss)
        self.test_mem["y_hats"].append(y_hat.softmax(dim=-1))
        self.test_mem["ys"].append(y)

        return loss

    def on_test_epoch_end(self):
        # Calculate average loss
        avg_loss = torch.tensor(self.test_mem["loss"]).mean()

        # Concatenate all predictions and labels
        y_hats = torch.cat(self.test_mem["y_hats"], dim=0)
        ys = torch.cat(self.test_mem["ys"], dim=0)

        # Calculate accuracy once on the entire test set
        test_acc = self.accuracy(y_hats, ys)

        # Log metrics
        self.log("test/loss", avg_loss)
        self.log("test/acc", test_acc)

        # Clear memory
        self.test_mem = {"loss": [], "y_hats": [], "ys": []}

    def configure_optimizers(self) -> Any:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, fused=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
