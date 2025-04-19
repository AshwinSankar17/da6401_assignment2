import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from torchmetrics.classification import MulticlassAccuracy

from typing import Any, Literal, Optional


def get_datasets(augmentation: bool = True, path: str = "../inaturalist_12K/"):
    ts = [transforms.Resize((256, 256))]
    if augmentation:
        ts += [transforms.AutoAugment()]
    ts += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4749, 0.4611, 0.3898], std=[0.1947, 0.1887, 0.1850]
        ),
    ]

    transform = transforms.Compose(ts)

    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "val")
    dataset = ImageFolder(train_path, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    test_dataset = ImageFolder(test_path, transform=transform)
    return train_dataset, val_dataset, test_dataset


class ConvolutionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        batch_norm: bool = True,
        activation: Literal["relu", "gelu", "silu", "mish"] = "relu",
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.activation_name = activation
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        activation_fn = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "mish": nn.Mish(),
        }.get(activation, nn.ReLU())
        layers.append(activation_fn)

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu" if self.activation_name == "relu" else "linear",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.block(x)


class CNNBase(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 32,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        batch_norm: bool = True,
        activation: Literal["relu", "gelu", "silu", "mish"] = "relu",
        kernel_strategy: Literal["same", "double", "half"] = "same",
        dropout: Optional[float] = 0.25,  # You can tune this value
    ):
        super().__init__()

        coeff = {"same": 1.0, "double": 2.0, "half": 0.5}[kernel_strategy]
        self.layers = nn.ModuleList()
        for _ in range(5):
            block = ConvolutionBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                batch_norm=batch_norm,
                activation=activation,
                dropout=dropout,
            )
            self.layers.append(block)
            in_channels = out_channels
            out_channels = max(1, int(out_channels * coeff))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ClassifierHead(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        in_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        activation: Literal["relu", "gelu", "silu", "mish"] = "relu",
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

        self.activation = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "mish": nn.Mish(),
        }.get(activation, nn.ReLU())

        self.activation_name = activation

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight,
                    nonlinearity="relu" if self.activation_name == "relu" else "linear",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(self.dropout(x))  # raw logits
        return x


class NeuralNetwork(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,  # Convolutional Layers
        kernel_size: int,
        stride: int,
        padding: int,
        learning_rate: float = 1e-3,
        batch_norm: bool = True,
        activation: Literal["relu", "gelu", "silu", "mish"] = "relu",
        kernel_strategy: Literal["same", "double", "half"] = "same",
        dropout: float = 0.0,
        num_classes: int = 10,
        hidden_size: int = 64,  # Fully-Connected Layers
        dataset_path: str = "../inaturalist_12K/",
        num_workers: int = 2,
        batch_size: int = 32,
        augmentation: bool = True,  # Datasets
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        # self.save_hyperparameters()
        self.cnn = CNNBase(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            batch_norm,
            activation,
            kernel_strategy,
        )
        in_size = self.get_in_size()
        self.classifier = ClassifierHead(
            num_classes, in_size, hidden_size, dropout, activation
        )
        self.accuracy = MulticlassAccuracy(num_classes=num_classes).to("cuda")
        self.train_dataset, self.val_dataset, self.test_dataset = get_datasets(
            augmentation, dataset_path
        )
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_mem = {"loss": [], "y_hats": [], "ys": []}
        self.test_mem = {"loss": [], "y_hats": [], "ys": []}

    def get_in_size(self):
        x = torch.randn(1, 3, 256, 256)
        x = self.cnn(x)
        return x.numel()

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, *_) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat.argmax(dim=-1), y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, *_) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # Store predictions and ground truth for later
        self.val_mem["loss"].append(loss)
        self.val_mem["y_hats"].append(y_hat)
        self.val_mem["ys"].append(y)

        return loss

    def on_validation_epoch_end(self):
        # Calculate average loss
        avg_loss = torch.tensor(self.val_mem["loss"]).mean()

        # Concatenate all predictions and labels
        y_hats = torch.cat(self.val_mem["y_hats"], dim=0)
        ys = torch.cat(self.val_mem["ys"], dim=0)

        # Calculate accuracy once on the entire validation set
        val_acc = self.accuracy(y_hats.argmax(dim=-1), ys)

        # Log metrics
        self.log("val/loss", avg_loss, prog_bar=True)
        self.log("val/acc", val_acc, prog_bar=True)

        # Clear memory for next epoch
        self.val_mem = {"loss": [], "y_hats": [], "ys": []}

    def test_step(self, batch, *_) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # Store predictions and ground truth for later
        self.test_mem["loss"].append(loss)
        self.test_mem["y_hats"].append(y_hat)
        self.test_mem["ys"].append(y)

        return loss

    def on_test_epoch_end(self):
        # Calculate average loss
        avg_loss = torch.tensor(self.test_mem["loss"]).mean()

        # Concatenate all predictions and labels
        y_hats = torch.cat(self.test_mem["y_hats"], dim=0)
        ys = torch.cat(self.test_mem["ys"], dim=0)

        # Calculate accuracy once on the entire test set
        test_acc = self.accuracy(y_hats.argmax(dim=-1), ys)

        # Log metrics
        self.log("test/loss", avg_loss)
        self.log("test/acc", test_acc)

        # Clear memory
        self.test_mem = {"loss": [], "y_hats": [], "ys": []}

    def configure_optimizers(self) -> None:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, fused=True
        )
        return optimizer

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
