import os
import platform
import torch
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from pytorch_lightning import Trainer
from train_utils import (
    ensure_directory_exists,
    mlflow_log_params,
    preprocess_data,
    initialize_model,
    initialize_trainer,
)

def run_train(args_train):
    """
    Executes the training pipeline:
    - Logs dataset metadata
    - Preprocesses data
    - Trains the model (logs per epoch)
    - Saves and registers the model
    """

    model_name = args_train["registered_model_name"]

    with mlflow.start_run(run_name=args_train["run_name"]) as run:
        run_id = run.info.run_id  # Capture Run ID for logging tables
        print(" Starting Training Pipeline...")

        # Log parameters to MLflow
        mlflow_log_params(args_train)

        # Ensure necessary directories exist
        ensure_directory_exists(args_train["tb_log_dir"])
        ensure_directory_exists(args_train["model_save_dir"])

        # Log Dataset to MLflow
        try:
            train_df = pd.read_csv(args_train["train_csv"])
            dataset = mlflow.data.from_pandas(train_df, source=args_train["train_csv"], name="training_dataset")
            mlflow.log_input(dataset, context="training")
            print(" Dataset Logged Successfully.\n")

            # ✅ Log validation dataset if exists
            val_csv = args_train.get("val_csv")
            if val_csv and os.path.exists(val_csv):
                val_df = pd.read_csv(val_csv)
                val_dataset = mlflow.data.from_pandas(val_df, source=val_csv, name="validation_dataset")
                mlflow.log_input(val_dataset, context="validation")
                print("✅ Validation dataset logged successfully.\n")
        except Exception as e:
            print(f" Failed to log dataset: {e}\n")

        # Preprocess Data
        preprocess_data(args_train["train_csv"], args_train["val_csv"])
        print(" Data preprocessing completed.\n")

        # Initialize Model & Trainer
        model = initialize_model(args_train)
        trainer = initialize_trainer(args_train)

        # Train the Model & Log Metrics
        print("🚀 Starting Model Training...\n")
        epoch_metrics = []  # Store metrics for logging table

        try:
            for epoch in range(args_train["epochs"]):
                print(f" Training Epoch {epoch + 1}/{args_train['epochs']}...")

                trainer.fit(model)  # Run training for this epoch
                # Optionally log a sample input and output shape if available
                try:
                    batch = next(iter(model.train_dataloader()))  # or however you load one batch
                    x, y = batch["image"], batch["label"]  # adapt keys to your dataloader
                    x_np, y_np = x.cpu().detach().numpy(), y.cpu().detach().numpy()
                    signature = mlflow.models.infer_signature(x_np, y_np)
                    mlflow.set_tag("input_shape", str(x_np.shape))
                    mlflow.set_tag("output_shape", str(y_np.shape))
                except Exception as e:
                    print(f" Could not log input/output signature: {e}")

                metrics = trainer.callback_metrics  # Retrieve epoch metrics

                # Convert and store metrics
                epoch_metric_entry = {"epoch": epoch}
                for key, value in metrics.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    epoch_metric_entry[key.replace("/", "_")] = value if value is not None else 0  #  Replace NaN with 0

                epoch_metrics.append(epoch_metric_entry)
                print(f" Logged metrics for epoch {epoch}: {epoch_metric_entry}")

                #  Log per-epoch metrics to MLflow
                for key, value in epoch_metric_entry.items():
                    if key != "epoch":
                        mlflow.log_metric(key, value, step=epoch)

            print(" Training Completed Successfully!\n")

        except Exception as e:
            print(f" Training Error: {e}\n")

        # Save Final Checkpoint
        final_checkpoint_path = os.path.join(args_train["model_save_dir"], "final_model_checkpoint.ckpt")
        os.makedirs(args_train["model_save_dir"], exist_ok=True)
        trainer.save_checkpoint(final_checkpoint_path)
        print(f" Final Model Checkpoint Saved: {final_checkpoint_path}\n")

        # Log Model to MLflow
        mlflow.pytorch.log_model(model, artifact_path="models", registered_model_name=model_name)
        print(" Model Logged to MLflow.")

    return model, trainer  # Return model & trainer for further use
