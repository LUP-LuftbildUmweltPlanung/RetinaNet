import os
import time
import mlflow
import logging
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import warnings
import torch
from train import run_train
from validate import run_evaluate
import run_split
from deepforest.main import deepforest
from predict import process_all_tif_files_in_folder
from train_utils import get_train_transform, get_validation_transform
from validate import extract_model_version, load_mlflow_model
from mlflow_config import *
warnings.filterwarnings("ignore")


# Optimize PyTorch precision
torch.set_float32_matmul_precision('medium')

#  Set MLflow request timeout
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "300"

print(f" MLflow Tracking URI Set: {mlflow.get_tracking_uri()}")

# Initialize MLflow Client
client = MlflowClient()

# Define Experiment Name
experiment_name = "DeepForest_ObjectDetection"

# Check if the Experiment Exists
experiment = client.get_experiment_by_name(experiment_name)

# DO NOT specify artifact_location if using remote MinIO
if experiment is None:
    print(f" Experiment '{experiment_name}' not found! Creating a new one...")
    experiment_id = client.create_experiment(name=experiment_name)
    print(f" Created new experiment: {experiment_name} (ID: {experiment_id})")
else:
    experiment_id = experiment.experiment_id
    print(f" Using existing experiment: {experiment_name} (ID: {experiment_id})")

#  Set the Active Experiment
mlflow.set_experiment(experiment_name)

#  Confirm Artifact Location (should point to s3:// if using MinIO bucket)
experiment = client.get_experiment(experiment_id)
print(f"🔍 Experiment '{experiment_name}' Artifact Location: {experiment.artifact_location}")

# Define run_split arguments (Hint: if you got an error try to change "max_empty" or "patch_size" values)
args_split = {
    "annotations": [  # List of paths to shapefile annotations
        r"Path\to\shapefile.shp",
        r"Path\to\shapefile.shp"
    ],
    "image_path": [  # List of paths to corresponding TIFF images
         r"Path\to\raster.tif",
         r"Path\to\raster.tif"
    ],
    "directory_to_save_crops": r"Path\to\output_folder",  # Directory to save cropped image tiles
    "patch_size": 400,  # Size of each crop (e.g., `400x400` pixels)
    "patch_overlap": 0.0,  # Overlap percentage between cropped tiles
    "merge_name": "csv_ref_merged.csv",  # Name of the merged output CSV file
    "split": 0.3,  # Percentage of data to allocate to the test set between [0.0, 1.0]
    "seed": 42,  # Random seed for reproducibility of the split
    "label": {'Tree': 0},  # Mapping of label names to numerical IDs
    "max_empty": 0.2  # Maximum proportion of empty tiles allowed
}

# Define train arguments
args_train = {
    "epochs": 50,  # Total number of epochs to train the model
    "check_val_every_n_epoch": 10,  # Perform validation after every n epochs   "Hint: ``check_val_every_n_epoch=10`` and ``patience=3``, the trainer will perform at least 40 training"
    "early_stop": True, # Treu if you want to apply early stop 
    "optimizer_patience": 20,  # Number of epochs with no improvement to wait before reducing LR
    "save_top_k": -1, # (1) Save the model checkpoints after every epoch ( -1 saved the last and best accuracy model)
    "augmentation_ratio": 0.5,  # Probability of applying data augmentation accourdint to the batch_size
    "batch_size": 4,  # Number of samples per batch
    "monitor": "val_classification",  # Metric to monitor during training: 'val_classification' or 'iou'
    "score_thresh": 0.4,  # Score threshold for filtering predictions
    "nms_thresh": 0.05,  # Non-Max Suppression (NMS) threshold for overlapping predictions
    "optimizer_type": "SGD",  # Optimizer type (e.g., 'SGD', 'Adam', 'AdamW')
    "learning_rate": 0.001,  # Initial learning rate for the optimizer
    "model_save_dir": r"Path\to\save\checkpoints",  # Directory to save model checkpoints
    "tb_log_dir": "tb_logs/deepforest",  # Directory to save TensorBoard logs
    "train_csv": r"Path\to\file\train_csv_ref_merged.csv",  # Path to the training dataset CSV file
    "val_csv": r"Path\to\file\test_csv_ref_merged.csv",  # Path to the validation dataset CSV file
    "default_label": "Tree",  # Default label for missing values in the dataset
    "num_classes": 1,
    "run_name": "training_Berlin_1",
    "registered_model_name": "DeepForest_ObjectDetection",
    # Assign transformations
    "train_transform": get_train_transform(),
    "val_transform": get_validation_transform()
}


# Define evaluation arguments
args_eval = {
    "model_checkpoints": [  # List of model checkpoint paths to evaluate or just the best one
       r"Path\to\best_model-epoch01-val_classification0.2652.ckpt"
    ],
    "evaluation_csv": r"Path\to\val_csv_ref_merged.csv",  # CSV file containing test dataset information
    "evaluation_root_dir": r"PAth\to\tile_images\folder",  # Root directory for evaluation data
    "predictions_save_dir": r"Path\to\evaluation\predictions\folder",  # Directory to save evaluation predictions
    "nms_thresh": 0.05,   # is used to remove duplicate or highly overlapping bounding boxes for the same object. The nms_thresh controls the IoU (Intersection over Union) threshold for suppressing overlapping boxes.
    "score_thresh": 0.2, # This sets a minimum confidence score for detected objects. Any bounding box with a confidence score below this threshold is discarded before the Non-Maximum Suppression (NMS) step.
    "run_name": "training_all_1_4",
}

# Define prediction arguments
args_predict = {
    "model_path": r"Path\to\model_saved\best_model-epoch01-val_classification0.2652.ckpt",  # Model path for prediction
    "folder_path": r"Path\to\target\folder",  # Folder containing TIFF files for prediction
    "savedir": r"Path\to\save\Shapefiles\predictions",  # Directory to save shapefiles generated by predictions
    "run_name": "predicted_all_1",
    "small_tiles": True,  # Whether to use tile-based prediction for large images
    "patch_size": 400,    # Size of each tile used during prediction
    "patch_overlap": 0.1,  # Overlap percentage between adjacent tiles reduce it when large image
    "iou_threshold": 0.1,   # Intersection over Union (IoU) threshold for remove overlapping bounding boxes (low .. remove, high .. keep)
    "thresh": 0.1,  # Confidence score threshold for filtering predictions (TP, FP)
    "output_name": "all_1"
}


# load the predict model
def run_predict(args_predict):
    """
    Run the prediction pipeline using a DeepForest model.

    Args:
        args_predict (dict): Dictionary containing prediction parameters.
    """
    logging.basicConfig(level=logging.DEBUG)
    os.makedirs(args_predict["savedir"], exist_ok=True)

    # Extract model path from args
    model_path = args_predict["model_path"]

    # Extract model version (for logging)
    model_version = extract_model_version(model_path)
    logging.info(f"Using Model Version: {model_version}")
    mlflow.set_tag("model_version", model_version)
    mlflow.set_tag("model_uri", model_path)
    mlflow.log_params(args_predict)
    # track number of files process
    num_images = len(os.listdir(args_predict["folder_path"]))
    mlflow.log_metric("num_images_predicted", num_images)

    # Load model using MLflow or direct checkpoint
    model = load_mlflow_model(model_path)
    if model is None:
        raise RuntimeError(f"❌ Failed to load model from {model_path}")

    # Set the NMS threshold dynamically
    model.config["nms_thresh"] = args_predict["iou_threshold"]  # Set correct NMS threshold
    model.nms_thresh = args_predict["iou_threshold"]  # Make sure it's applied

    logging.basicConfig(level=logging.DEBUG)
    logging.info(f"Updated nms_thresh to: {model.nms_thresh}")
    logging.info(f"Using IoU threshold for NMS: {args_predict['iou_threshold']}")

    process_all_tif_files_in_folder(
        model=model,
        file_path=args_predict["folder_path"],
        savedir=args_predict["savedir"],
        run_name=args_predict["run_name"],
        small_tiles=args_predict["small_tiles"],
        patch_size=args_predict["patch_size"],
        patch_overlap=args_predict["patch_overlap"],
        iou_threshold=args_predict["iou_threshold"],
        thresh=args_predict["thresh"],
        output_name=args_predict["output_name"]
    )


# Define tasks
split_raster = False
train = True
validate = False
predict = False

if __name__ == "__main__":
    if split_raster:
        print("\nStarting data preprocessing and split pipeline...")
        run_split.preprocess(**args_split)
        print("Data preprocessing and split completed.")
    if train:
        print("🚀 Running Training...")
        run_train(args_train)
        print("✅ Training Completed.")

    if validate:
        print("🚀 Running Evaluation...")
        run_evaluate(args_eval)
        print("✅ Evaluation Completed.")

    if predict:
        print("🚀 Running Prediction...")
        run_predict(args_predict)
        print("✅ Prediction Completed.")

    print("🎯 Script Finished.")
