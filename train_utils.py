import os
import random
import torch
import pandas as pd
from torchvision import transforms
from deepforest.main import deepforest
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mlflow
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    Callback
)

# -------------------------------
# Helper Functions for MLflow
# -------------------------------
def ensure_directory_exists(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Created missing directory: {path}")
def mlflow_log_params(params_dict):
    """Logs all hyperparameters to MLflow."""
    for key, value in params_dict.items():
        mlflow.log_param(key, value)
def log_metrics_during_training(epoch, metrics):
    """
    Logs all available training metrics at each epoch dynamically.

    - Converts tensor values to Python floats before logging.
    - Handles missing values safely.
    - Logs every available metric from `trainer.callback_metrics`.
    """

    sanitized_metrics = {}
    for key, value in metrics.items():
        safe_key = key.replace("/", "_")  # Replace `/` with `_` to avoid MLflow issues

        try:
            if isinstance(value, torch.Tensor):
                sanitized_metrics[safe_key] = value.item()  # Convert Tensor to float
            elif isinstance(value, (int, float)):
                sanitized_metrics[safe_key] = float(value)  # Ensure all values are float
            else:
                print(f"⚠️ Skipping {key}: Unsupported type {type(value)}")

        except Exception as e:
            print(f"⚠️ Warning: Failed to process metric {key}. Skipping... Error: {e}")

    #  Debugging: Print all available metrics for the epoch
    print(f" Epoch {epoch} - Metrics Available: {list(metrics.keys())}")

    #  Log all sanitized metrics to MLflow
    try:
        mlflow.log_metrics(sanitized_metrics, step=epoch)  # Log all metrics at once
        print(f" Logged metrics for epoch {epoch}: {sanitized_metrics}")

    except Exception as e:
        print(f" Warning: Failed to log metrics for epoch {epoch}. Error: {e}")


class MLFlowLoggingCallback(Callback):
    """Logs all training & validation metrics to MLflow after each epoch and saves as a DataFrame at the end."""

    def __init__(self):
        super().__init__()
        self.epoch_metrics = []  # Store all epoch metrics

    def on_train_epoch_end(self, trainer, pl_module):
        """Logs training metrics at the end of each training epoch."""
        metrics = trainer.callback_metrics  # Get metrics
        epoch = trainer.current_epoch  # Get current epoch

        epoch_metric_entry = {"epoch": epoch}

        #  Extract & ensure values are stored correctly
        for key in [
            "train_classification", "train_bbox_regression", "train_loss",
            "val_classification", "val_bbox_regression", "iou",
            "map", "map_50", "map_75", "mar_100", "Tree_Recall", "Tree_Precision"
        ]:
            value = metrics.get(key, None)  # Use `.get()` to avoid missing keys

            if isinstance(value, torch.Tensor):
                value = value.item()  # Convert Tensor to float

            epoch_metric_entry[key] = value if value is not None else 0  #  Replace NaN with 0

        self.epoch_metrics.append(epoch_metric_entry)  # Store for later logging

        #  Log per-epoch metrics to MLflow immediately
        for key, value in epoch_metric_entry.items():
            if key != "epoch":  # Avoid logging epoch number as metric
                mlflow.log_metric(key, value, step=epoch)

        print(f" MLflow: Logged metrics for epoch {epoch}")

    def on_train_end(self, trainer, pl_module):
        """Logs full training history to MLflow as a table at the end."""
        try:
            if self.epoch_metrics:
                metrics_df = pd.DataFrame(self.epoch_metrics)

                #  Ensure all columns exist & fill missing values with 0
                metrics_df.fillna(0, inplace=True)

                print(" Final training metrics table:")
                #print(metrics_df)  # Debugging: Check table content
                mlflow.log_table(data=metrics_df, artifact_file="training_metrics.json")
                print(" Training Metrics Table Logged to MLflow.")
        except Exception as e:
            print(f" Failed to log training table: {e}")


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
    #model.config["score_thresh"] = args["score_thresh"]


    #  Assign Transforms
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
    mode = "min" if monitor_metric in ["val_classification", "loss", "val_bbox_regression"] else "max"
    # Set learning rate scheduler
    model.scheduler = ReduceLROnPlateau(
        optimizer=model.optimizer,
        mode=mode,
        patience= 20, # args["optimizer_patience"],
        verbose=True,
    )
    return model

# Initialize Trainer
def initialize_trainer(args):
    """
    Initialize PyTorch Lightning trainer.
    """
    # Ensure min_epochs does not exceed max_epochs
    min_epochs = min(args.get("min_epochs", 1), args["epochs"])

    monitor = args["monitor"]
    mode = "min" if monitor in ["val_classification", "loss", "val_bbox_regression"] else "max"
    # logger = TensorBoardLogger("tb_logs", name="deepforest")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args["model_save_dir"],
        filename=f"best_model-epoch{{epoch:02d}}-{monitor}{{{monitor}:.4f}}",
        monitor=monitor,
        mode=mode,
        save_top_k=args["save_top_k"],
        verbose=True,
    )

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    if args["early_stop"]:
        trainer = Trainer(
            max_epochs=args["epochs"],
            min_epochs=min_epochs,
            # check_val_every_n_epoch=args["check_val_every_n_epoch"],
            callbacks=[checkpoint_callback, MLFlowLoggingCallback(), EarlyStopping(
                monitor=monitor,
                patience=args["optimizer_patience"],
                verbose=True,
                mode=mode
            ), LearningRateMonitor(logging_interval="step")],
            logger = TensorBoardLogger("tb_logs", name="deepforest"),
            log_every_n_steps=50,
            val_check_interval=1.0, # 0.25 Runs validation after completing 25% of the training epoch. or 'int' Runs validation every 10 training batches. recommend: "num of batches every epoch" num_of_batch in one epoch
            check_val_every_n_epoch=None, # Perform a validation loop after every `N` training epochs
            devices=1,
            accelerator=accelerator,
            precision=16,
            # gradient_clip_val=0.5,
        )
    else:
        trainer = Trainer(
            max_epochs=args["epochs"],
            min_epochs=min_epochs,
            callbacks=[checkpoint_callback, MLFlowLoggingCallback()],
            val_check_interval=1.0, # 0.25 Runs validation after completing 25% of the training epoch. or 'int' Runs validation every 10 training batches.
            check_val_every_n_epoch=1, # Perform a validation loop after every `N` training epochs
            accelerator=accelerator,
            precision=16,
        )

    return trainer
