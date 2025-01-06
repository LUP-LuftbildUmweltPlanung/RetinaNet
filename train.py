from train_utils import (
    preprocess_data,
    initialize_model,
    initialize_trainer,
    get_train_transform,
    get_validation_transform,
)
import subprocess
from pytorch_lightning.callbacks import ModelCheckpoint


def run_train(args_train):
    """
    Main function to execute the training pipeline.

    Args:
        args_train (dict): Dictionary containing training arguments.
    """
    # Log all training arguments at the start
    print("Training Arguments:")
    for key, value in args_train.items():
        print(f"{key}: {value}")
    print("\n")

    # Step 1: Preprocessing
    print("Preprocessing data with the following parameters:")
    print(f"Train CSV: {args_train['train_csv']}")
    print(f"Validation CSV: {args_train['val_csv']}")
    print(f"Default Label: {args_train.get('default_label', 'Tree')}")
    preprocess_data(
        args_train["train_csv"],
        args_train["val_csv"],
        default_label=args_train.get("default_label", "Tree"),
    )
    print("Data preprocessing completed.\n")

    # Step 2: Model and Transformations
    print("Initializing model and transformations...")

    # Custom training and validation transformations
    args_train["train_transform"] = get_train_transform()
    print("Train Transform:")
    print(args_train["train_transform"])

    args_train["val_transform"] = get_validation_transform()
    print("Validation Transform:")
    print(args_train["val_transform"])

    # Initialize the model
    model = initialize_model(args_train)
    print("Model Initialization Details:")
    print(f"Model Type: {args_train.get('model_type', 'DeepForest')}")
    print(f"Learning Rate: {args_train['learning_rate']}")
    print(f"Optimizer Type: {args_train['optimizer_type']}")
    print(f"Number of Classes: {args_train.get('num_classes', 'Not Specified')}")
    print("Model initialization completed.\n")

    # Step 3: Trainer
    print("Initializing trainer...")
    trainer = initialize_trainer(args_train)
    print("Trainer Initialization Details:")
    print(f"Checkpoint Directory: {args_train['model_save_dir']}")
    print(f"TensorBoard Log Directory: {args_train['tb_log_dir']}")
    print(f"Monitored Metric: {args_train['monitor']}")
    print(f"Validation Frequency: {args_train['valid_every_n_epochs']} epoch(s)")
    print("Trainer initialization completed.\n")

    # Step 4: TensorBoard
    print("Starting TensorBoard...")
    log_dir = args_train["tb_log_dir"]
    tensorboard_port = args_train["tensorboard_port"]
    try:
        subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", str(tensorboard_port)])
        print(f"TensorBoard is running. Access it at http://localhost:{tensorboard_port}\n")
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}\n")

    # Step 5: Training
    print("Starting training...")
    try:
        trainer.fit(model)
        print("Training completed successfully!\n")
    except Exception as e:
        print(f"Error during training: {e}\n")

    # Step 6: Save Final Checkpoint
    print("Saving the final checkpoint...")
    final_checkpoint_path = f"{args_train['model_save_dir']}/final_model_checkpoint.ckpt"
    try:
        trainer.save_checkpoint(final_checkpoint_path)
        print(f"Final model checkpoint saved at: {final_checkpoint_path}\n")
    except Exception as e:
        print(f"Failed to save the final checkpoint: {e}\n")
