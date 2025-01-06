import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from deepforest.main import deepforest
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    Callback
)
from pytorch_lightning.loggers import TensorBoardLogger


# Visualization Function
def visualize_before_after(img_original, img_augmented, bboxes_original, bboxes_augmented):
    """
    Visualize an image before and after augmentation along with bounding boxes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for ax, img, bboxes, title in zip(
        axes,
        [img_original, img_augmented],
        [bboxes_original, bboxes_augmented],
        ["Before Augmentation", "After Augmentation"]
    ):
        if isinstance(img, np.ndarray) and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        elif isinstance(img, torch.Tensor) and img.shape[0] == 3:
            img = img.permute(1, 2, 0).detach().cpu().numpy()

        ax.imshow(img)
        for box in bboxes:
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)
        ax.set_title(title)
        ax.axis("off")
    plt.show()


# Data Preprocessing
def preprocess_data(train_csv, val_csv, default_label="Tree"):
    """
    Fill missing labels and standardize dataset structure.
    """
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    for df in [train_df, val_df]:
        df["label"] = df["label"].fillna(default_label).str.capitalize()

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)


# Selective Augmentation
log_augmentation_details_once = True


def selective_augmentation(batch, train_transform, augmentation_ratio):
    """
    Apply augmentation to a subset of images in the batch based on augmentation_ratio.
    """
    global log_augmentation_details_once

    images, targets = batch
    batch_size = len(images)
    num_augmented = int(batch_size * augmentation_ratio)
    indices_to_augment = random.sample(range(batch_size), num_augmented)

    augmented_images = []
    for i, img in enumerate(images):
        if torch.isnan(img).any():
            print(f"NaN detected in raw image at index {i}, replacing with 0.")
            img[torch.isnan(img)] = 0
        augmented_images.append(train_transform(img) if i in indices_to_augment else img)

    if log_augmentation_details_once:
        print(f"Augmentation Settings:")
        print(f" - Total images in batch: {batch_size}")
        print(f" - Number of images to augment: {num_augmented}")
        print(f" - Augmented image indices (example): {indices_to_augment}")
        log_augmentation_details_once = False

    return augmented_images, targets


# Data Augmentation Pipelines
def get_train_transform():
    """
    Define training-specific augmentations.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.ToTensor(),
    ])


def get_validation_transform():
    """
    Define validation-specific transformations.
    """
    return transforms.Compose([
        transforms.ToTensor(),  # Validation doesn't require augmentations
    ])


# Initialize Model
def initialize_model(args):
    """
    Initialize the DeepForest model with custom training and validation transformations.
    """
    model = deepforest()
    model.use_release()

    # Set training configuration
    model.config["train"]["csv_file"] = args["train_csv"]
    model.config["train"]["root_dir"] = os.path.dirname(args["train_csv"])
    model.config["validation"]["csv_file"] = args["val_csv"]
    model.config["validation"]["root_dir"] = os.path.dirname(args["val_csv"])
    model.config["batch_size"] = args["batch_size"]
    model.config["num_classes"] = args["num_classes"]  # Set the number of classes

    model.train_transform = args.get("train_transform", None)
    model.val_transform = args.get("val_transform", None)

    # Set optimizer
    optimizer_type = args.get("optimizer_type", "SGD").lower()
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": lambda params: torch.optim.SGD(params, lr=args["learning_rate"], momentum=args.get("momentum", 0.9))
    }
    if optimizer_type not in optimizers:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    model.optimizer = optimizers[optimizer_type](model.model.parameters())

    # define monitor
    monitor_metric = args["monitor"]
    mode = "min" if monitor_metric in ["val_classification"] else "max"
    # Set learning rate scheduler
    model.scheduler = ReduceLROnPlateau(
        optimizer=model.optimizer,
        mode=mode,
        patience= args["optimizer_patience"],
        verbose=True,
    )

    return model

# Custom Callbacks
class IOULogger(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if "iou" in trainer.callback_metrics:
            iou_value = trainer.callback_metrics["iou"]
            if isinstance(iou_value, torch.Tensor):  # Check if it's a tensor
                iou_value = iou_value.item()  # Convert tensor to float
            print(f"[Epoch {trainer.current_epoch}] IoU: {iou_value} (Type: {type(iou_value)})")
            assert isinstance(iou_value, float), "IoU metric must be a float."

class LearningRateLogger(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print(f"[Epoch {trainer.current_epoch}] Learning Rate: {trainer.optimizers[0].param_groups[0]['lr']}")

# Initialize Trainer
def initialize_trainer(args):
    """
    Initialize PyTorch Lightning trainer.
    """
    logger = TensorBoardLogger(save_dir=args["tb_log_dir"])
    monitor = args["monitor"]
    mode = "min" if monitor == "val_classification" else "max"

    checkpoint_callback = ModelCheckpoint(
        dirpath=args["model_save_dir"],
        filename=f"best_model-epoch{{epoch:02d}}-{monitor}{{{monitor}:.4f}}",
        monitor=monitor,
        mode=mode,
        save_top_k=args["save_top_k"],
        verbose=True,
        auto_insert_metric_name=False,
    )

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    if args["early_stop"]:
        trainer = Trainer(
            max_epochs=args["epochs"],
            # check_val_every_n_epoch=args["check_val_every_n_epoch"],
            callbacks=[checkpoint_callback, IOULogger(), EarlyStopping(
                monitor=monitor,
                patience=args["optimizer_patience"],
                verbose=True,
                mode=mode
            ), LearningRateMonitor(logging_interval="step")],
            logger=logger,
            devices=1,
            accelerator=accelerator,
            precision=16,
        )
    else:
        trainer = Trainer(
            max_epochs=args["epochs"],
            callbacks=[checkpoint_callback, IOULogger()],
            val_check_interval=1.0, # 0.25 Runs validation after completing 25% of the training epoch. or 'int' Runs validation every 10 training batches.
            check_val_every_n_epoch=1, # Perform a validation loop after every `N` training epochs
            accelerator=accelerator,
            precision=16,
        )

    return trainer
