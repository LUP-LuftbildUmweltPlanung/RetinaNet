import os
import pandas as pd
from deepforest import main, evaluate
import logging
import geopandas as gpd
import shapely.geometry

def run_evaluate(args_eval):
    """
    Execute the evaluation pipeline for DeepForest model.

    Args:
        args_eval (dict):
            A dictionary containing evaluation parameters:
            - "model_checkpoints" (list): List of model checkpoint paths.
            - "evaluation_csv" (str): Path to the evaluation CSV file.
            - "evaluation_root_dir" (str): Root directory containing evaluation images.
            - "predictions_save_dir" (str): Directory to save the evaluation results.
            - "evaluation_image" (str, optional): Specific image name for evaluation.

    Raises:
        FileNotFoundError: If the evaluation CSV file or root directory does not exist.
    """
    logging.basicConfig(level=logging.DEBUG)
    os.makedirs(args_eval["predictions_save_dir"], exist_ok=True)

    # Determine the evaluation CSV file based on provided image name or general CSV
    csv_file = os.path.join(args_eval["evaluation_root_dir"], args_eval["evaluation_image"] + ".csv") \
        if "evaluation_image" in args_eval else args_eval["evaluation_csv"]

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Evaluation CSV file not found: {csv_file}")

    if not any(os.scandir(args_eval["evaluation_root_dir"])):
        raise FileNotFoundError(f"No files found in the evaluation root directory: {args_eval['evaluation_root_dir']}")

    # Lists to store evaluation metrics
    all_box_recalls = []
    all_box_precisions = []
    all_mean_ious = []
    results_list = []

    for model_checkpoint in args_eval["model_checkpoints"]:
        logging.info(f"Evaluating model checkpoint: {model_checkpoint}")
        try:
            model = main.deepforest.load_from_checkpoint(checkpoint_path=model_checkpoint)
            root_dir = args_eval["evaluation_root_dir"]

            logging.info(f"Using CSV file for evaluation: {csv_file}")
            logging.info(f"Using root directory for evaluation: {root_dir}")

            # Generate predictions and load ground truth
            predictions = model.predict_file(csv_file=csv_file, root_dir=root_dir)
            ground_truth = pd.read_csv(csv_file).dropna(subset=['label'])

            # Convert predictions and ground truth to GeoDataFrame format
            predictions["geometry"] = predictions.apply(
                lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
            predictions = gpd.GeoDataFrame(predictions, geometry="geometry")

            ground_truth["geometry"] = ground_truth.apply(
                lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
            ground_truth = gpd.GeoDataFrame(ground_truth, geometry="geometry")

            # Perform evaluation
            result = evaluate.evaluate_boxes(
                predictions=predictions,
                ground_df=ground_truth,
                root_dir=root_dir,
                iou_threshold=0.4,
                savedir=args_eval.get('predictions_save_dir', None)
            )

            if result["results"].empty:
                logging.warning("No matches found, result dataframe is empty.")
                continue

            box_recall = result["box_recall"]
            box_precision = result["box_precision"]
            mean_iou = result["results"]["IoU"].mean()

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

    # Save results
    results_df = pd.DataFrame(results_list)
    results_csv_path = os.path.join(args_eval["predictions_save_dir"], "evaluation_results.csv")
    results_df.to_csv(results_csv_path, index=False)

    overall_metrics_path = os.path.join(args_eval["predictions_save_dir"], "overall_metrics.txt")
    with open(overall_metrics_path, "w") as f:
        f.write(f"Overall Box Recall: {overall_box_recall:.4f}\n")
        f.write(f"Overall Box Precision: {overall_box_precision:.4f}\n")
        f.write(f"Overall Mean IoU: {overall_mean_iou:.4f}\n")

    logging.info(f"Evaluation results saved to: {results_csv_path}")
    logging.info(f"Overall metrics saved to: {overall_metrics_path}")
    logging.info("Evaluation pipeline completed.")
