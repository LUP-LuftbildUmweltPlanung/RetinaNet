import os
import logging
import pandas as pd
import geopandas as gpd
import shapely.geometry
import mlflow
from deepforest import evaluate, main
import torch
import mlflow.pytorch
from pytorch_lightning import Trainer
import re


def extract_model_version(model_path):
    """
    Extracts the MLflow model version from the given model path.

    Args:
        model_path (str): The path to the model checkpoint or MLflow model URI.

    Returns:
        str: The extracted model version (e.g., "v42") or "Unknown" if not found.
    """
    # Case 1: Model registered in MLflow (e.g., "models:/DeepForest_ObjectDetection/42")
    match_mlflow = re.search(r"/mlruns/.+?/artifacts/models$", model_path)
    if match_mlflow:
        parts = os.path.normpath(model_path).split(os.sep)
        try:
            # Extract run ID and look up MLflow model version dynamically
            run_id = parts[-3]  # Run ID is 3rd last element before "artifacts/models"
            return f"Run-{run_id[:8]}"  # Return shortened run ID (first 8 chars)
        except IndexError:
            return "Unknown"

    # Case 2: Model checkpoint file (e.g., "best_model-epoch00-iou0.7133.ckpt")
    match_ckpt = re.search(r"epoch(\d+)-iou([\d.]+)\.ckpt$", model_path)
    if match_ckpt:
        epoch = match_ckpt.group(1)
        iou = match_ckpt.group(2)
        return f"Epoch-{epoch}_IoU-{iou}"

    return "Unknown"  # Return default if no version found


def load_mlflow_model(model_uri):
    try:
        logging.info(f"🔍 Loading model from: {model_uri}")

        if model_uri.startswith("models:/") or model_uri.endswith("/artifacts/models"):
            logging.info("🔍 Fetching model from MLflow artifact store...")
            model = mlflow.pytorch.load_model(model_uri)
            trainer = Trainer(logger=False)
            model.trainer = trainer
            logging.info(f" Successfully loaded model from MLflow: {model_uri}")
            return model

        elif model_uri.endswith(".ckpt"):
            logging.info("🔍 Loading model from direct checkpoint...")
            model = main.deepforest.load_from_checkpoint(checkpoint_path=model_uri)
            trainer = Trainer(logger=False)
            model.trainer = trainer
            logging.info(f" Successfully loaded model from checkpoint: {model_uri}")
            return model

        else:
            raise ValueError(f" Invalid model path: {model_uri}")

    except Exception as e:
        logging.error(f" Failed to load model: {e}")
        return None


def run_evaluate(args_eval):
    """Execute the evaluation pipeline for DeepForest model."""
    logging.basicConfig(level=logging.DEBUG)
    os.makedirs(args_eval["predictions_save_dir"], exist_ok=True)

    # Extract model checkpoint path
    model_checkpoint_path = args_eval["model_checkpoints"][0]
    model_version = extract_model_version(model_checkpoint_path)
    logging.info(f"Using Model Version: {model_version}")

    # # Extract model ID safely
    # path_parts = os.path.normpath(model_checkpoint_path).split(os.sep)
    # if len(path_parts) < 2:
    #     raise ValueError(f"Error extracting model ID: Path too short -> {model_checkpoint_path}")

    # Ensure "Evaluation_" is always prefixed before the run_name
    run_name = f"Evaluation_{args_eval.get('run_name', model_version)}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(args_eval)  # Log parameters
        mlflow.log_param("model_version", model_version)  # Log extracted version

        # Determine evaluation CSV
        csv_file = args_eval["evaluation_csv"]
        root_dir = args_eval["evaluation_root_dir"]

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Evaluation CSV file not found: {csv_file}")

        if not any(os.scandir(root_dir)):
            raise FileNotFoundError(f"No files found in the evaluation root directory: {root_dir}")

        try:
            eval_df = pd.read_csv(csv_file)
            eval_dataset = mlflow.data.from_pandas(eval_df, source=csv_file, name="evaluation_dataset")
            mlflow.log_input(eval_dataset, context="evaluation")
            logging.info(" Evaluation dataset logged to MLflow.")
        except Exception as e:
            logging.warning(f"⚠️ Could not log evaluation dataset: {e}")

        # Retrieve the NMS threshold value from args_eval (default to 0.8 if not provided)
        nms_thresh = args_eval.get("nms_thresh", 0.05)
        score_thresh = args_eval.get("score_thresh", 0.4)

        # Initialize lists for evaluation metrics
        all_box_recalls = []
        all_box_precisions = []
        all_mean_ious = []
        results_list = []

        for model_checkpoint in args_eval["model_checkpoints"]:
            logging.info(f"Evaluating model checkpoint from MLflow: {model_checkpoint}")

            # Load the model and get run_id separately
            model = load_mlflow_model(model_checkpoint)
            if model is None:
                logging.error(f"Skipping evaluation for {model_checkpoint} due to model loading failure.")
                continue

            logging.info(f"Using CSV file for evaluation: {csv_file}")
            logging.info(f"Using root directory for evaluation: {root_dir}")

            # Ensure the model is in evaluation mode
            model.model.eval()  # <-- Insert this line here

            try:
                # Generate predictions
                predictions = model.predict_file(csv_file=csv_file, root_dir=root_dir)
                ground_truth = pd.read_csv(csv_file).dropna(subset=['label'])

                # Convert predictions and ground truth to GeoDataFrame format
                if not predictions.empty:
                    predictions["geometry"] = predictions.apply(
                        lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1
                    )
                    predictions = gpd.GeoDataFrame(predictions, geometry="geometry")

                if not ground_truth.empty:
                    ground_truth["geometry"] = ground_truth.apply(
                        lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1
                    )
                    ground_truth = gpd.GeoDataFrame(ground_truth, geometry="geometry")

                # Perform evaluation
                result = evaluate.evaluate_boxes(
                    predictions=predictions,
                    ground_df=ground_truth,
                    root_dir=root_dir,
                    iou_threshold=score_thresh,
                    savedir=args_eval.get('predictions_save_dir', None)
                )

                if result["results"].empty:
                    logging.warning("No matches found, result dataframe is empty.")
                    box_recall, box_precision, mean_iou = 0.0, 0.0, 0.0  # Avoid KeyError
                else:
                    box_recall = result["box_recall"]
                    box_precision = result["box_precision"]
                    mean_iou = result["results"][result["results"]["match"]][
                        "IoU"].mean()  # Compute IoU only for matched boxes

                    logging.info(f"Total matched predictions: {sum(result['results']['match'])}")
                    logging.info(f"Mean IoU (only matched predictions): {mean_iou:.4f}")
                    logging.info(
                        f"IoU distribution of matches:\n{result['results'][result['results']['match']]['IoU'].describe()}")

                logging.info(f"Box Recall: {box_recall:.4f}")
                logging.info(f"Box Precision: {box_precision:.4f}")
                logging.info(f"Mean IoU: {mean_iou:.4f}")

                all_box_recalls.append(box_recall)
                all_box_precisions.append(box_precision)
                all_mean_ious.append(mean_iou)

                results_list.append({
                    "model_checkpoint": model_checkpoint,
                    "box_recall": box_recall,
                    "box_precision": box_precision,
                    "mean_iou": mean_iou,
                    "iou_results": result["results"].to_dict(orient='records')
                })

            except Exception as e:
                logging.error(f"Error during evaluation for checkpoint {model_checkpoint}: {e}")
                continue

        # Compute overall metrics
        overall_box_recall = sum(all_box_recalls) / len(all_box_recalls) if all_box_recalls else 0.0
        overall_box_precision = sum(all_box_precisions) / len(all_box_precisions) if all_box_precisions else 0.0
        overall_mean_iou = sum(all_mean_ious) / len(all_mean_ious) if all_mean_ious else 0.0

        logging.info(f"Overall Box Recall: {overall_box_recall:.4f}")
        logging.info(f"Overall Box Precision: {overall_box_precision:.4f}")
        logging.info(f"Overall Mean IoU: {overall_mean_iou:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("overall_box_recall", overall_box_recall)
        mlflow.log_metric("overall_box_precision", overall_box_precision)
        mlflow.log_metric("overall_mean_iou", overall_mean_iou)

        # Save results
        results_df = pd.DataFrame(results_list)
        results_csv_path = os.path.join(args_eval["predictions_save_dir"], "evaluation_results.csv")
        results_df.to_csv(results_csv_path, index=False)

        # Drop the problematic column before logging
        results_df_cleaned = results_df.drop(columns=["iou_results"], errors="ignore")

        # Add training run name to results
        results_df_cleaned["training_reg"] = model_version

        # Log the cleaned dataframe
        mlflow.log_table(data=results_df_cleaned, artifact_file="evaluation_results.json")

        overall_metrics_path = os.path.join(args_eval["predictions_save_dir"], "overall_metrics.txt")
        with open(overall_metrics_path, "w") as f:
            f.write(f"Overall Box Recall: {overall_box_recall:.4f}\n")
            f.write(f"Overall Box Precision: {overall_box_precision:.4f}\n")
            f.write(f"Overall Mean IoU: {overall_mean_iou:.4f}\n")

        logging.info(f"Evaluation results saved to: {results_csv_path}")
        logging.info(f"Overall metrics saved to: {overall_metrics_path}")

        # Log artifacts to MLflow
        if os.path.exists(results_csv_path):
            mlflow.log_artifact(results_csv_path)
        else:
            logging.warning(f"Evaluation file {results_csv_path} not found, skipping MLflow logging.")

        if os.path.exists(overall_metrics_path):
            mlflow.log_artifact(overall_metrics_path)
        else:
            logging.warning(f"Metrics file {overall_metrics_path} not found, skipping MLflow logging.")

        logging.info("Evaluation pipeline completed.")
