import os
import pandas as pd
from deepforest import main
from deepforest import evaluate
import logging
import geopandas as gpd  # for GeoDataFrame
import shapely.geometry  # for geometry operations


def run_evaluate(args_eval):
    """
    Main function to execute the evaluation pipeline and export results.
    Args:
        args_eval: Dictionary of evaluation arguments.
    """
    # Set up logging for detailed debugging
    logging.basicConfig(level=logging.DEBUG)

    # Ensure predictions directory exists
    os.makedirs(args_eval["predictions_save_dir"], exist_ok=True)

    # Check if the evaluation file exists
    if not os.path.exists(args_eval["evaluation_csv"]):
        raise FileNotFoundError(f"Evaluation CSV file not found: {args_eval['evaluation_csv']}")

    # Check if there are files in the evaluation root directory
    if not any(os.scandir(args_eval["evaluation_root_dir"])):
        raise FileNotFoundError(f"No files found in the evaluation root directory: {args_eval['evaluation_root_dir']}")

    # Prepare lists to accumulate results
    all_box_recalls = []
    all_box_precisions = []
    results_list = []

    # Loop through all model checkpoints provided
    for model_checkpoint in args_eval["model_checkpoints"]:
        logging.info(f"Evaluating model checkpoint: {model_checkpoint}")

        try:
            # Load the model from the checkpoint
            model = main.deepforest.load_from_checkpoint(checkpoint_path=model_checkpoint)

            # Path to evaluation annotations file and root directory
            csv_file = args_eval["evaluation_csv"]
            root_dir = args_eval["evaluation_root_dir"]

            logging.info(f"Using CSV file for evaluation: {csv_file}")
            logging.info(f"Using root directory for evaluation: {root_dir}")

            # Create the predictions dataframe from the CSV file
            predictions = model.predict_file(csv_file=csv_file, root_dir=root_dir)

            # Load the ground truth (annotations)
            ground_truth = pd.read_csv(csv_file)

            # Remove NaN labels from ground truth
            ground_truth = ground_truth.dropna(subset=['label'])

            # Ensure both predictions and ground truth are in the correct format
            predictions["geometry"] = predictions.apply(lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax),
                                                        axis=1)
            predictions = gpd.GeoDataFrame(predictions, geometry="geometry")

            ground_truth["geometry"] = ground_truth.apply(
                lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
            ground_truth = gpd.GeoDataFrame(ground_truth, geometry="geometry")

            # Evaluate predictions and ground truth using evaluate.evaluate_boxes()
            result = evaluate.evaluate_boxes(
                predictions=predictions,
                ground_df=ground_truth,
                root_dir=root_dir,
                iou_threshold=0.4,  # Set IoU threshold to 0.4 or adjust as needed
                savedir=args_eval.get('predictions_save_dir', None)
            )
            print("results", result)
            logging.info(f"Evaluation Results: {result.keys()}")

            # Extract precision and recall based on IoU threshold of 0.4
            box_recall = result["box_recall"]
            box_precision = result["box_precision"]

            logging.info(f"Box Recall: {box_recall:.4f}")
            logging.info(f"Box Precision: {box_precision:.4f}")

            # Accumulate results for overall metrics
            all_box_recalls.append(box_recall)
            all_box_precisions.append(box_precision)

            # Save results for this checkpoint
            results_list.append({
                "model_checkpoint": model_checkpoint,
                "box_recall": box_recall,
                "box_precision": box_precision,
            })

        except Exception as e:
            logging.error(f"Error during evaluation for checkpoint {model_checkpoint}: {e}")
            continue

    # Compute overall recall and precision
    overall_box_recall = sum(all_box_recalls) / len(all_box_recalls) if all_box_recalls else 0.0
    overall_box_precision = sum(all_box_precisions) / len(all_box_precisions) if all_box_precisions else 0.0

    # Log and save overall metrics
    logging.info(f"Overall Box Recall: {overall_box_recall:.4f}")
    logging.info(f"Overall Box Precision: {overall_box_precision:.4f}")

    # Convert results list to a DataFrame and save all results to CSV
    results_df = pd.DataFrame(results_list)
    results_csv_path = os.path.join(args_eval["predictions_save_dir"], "evaluation_results.csv")
    results_csv_path_all = os.path.join(args_eval["predictions_save_dir"], "results.csv")
    results_df.to_csv(results_csv_path, index=False)
    result_df_all = pd.DataFrame([result])  # Assuming `result` is a dictionary; modify if it's a list of dicts
    result_df_all.to_csv(results_csv_path_all, index=False)

    # Save overall metrics to a separate file
    overall_metrics_path = os.path.join(args_eval["predictions_save_dir"], "overall_metrics.txt")
    with open(overall_metrics_path, "w") as f:
        f.write(f"Overall Box Recall: {overall_box_recall:.4f}\n")
        f.write(f"Overall Box Precision: {overall_box_precision:.4f}\n")

    logging.info(f"Evaluation results saved to: {results_csv_path}")
    logging.info(f"Overall metrics saved to: {overall_metrics_path}")

    logging.info("Evaluation pipeline completed.")

    # Extract precision and recall based on IoU threshold of 0.4
    box_recall = result["box_recall"]
    box_precision = result["box_precision"]
    class_recall = result["class_recall"]

    logging.info(f"Box Recall: {box_recall:.4f}")
    logging.info(f"Box Precision: {box_precision:.4f}")
    #logging.info(f"Class-wise Recall and Precision:\n{class_recall}")

    # Optional: Save the evaluation results or visualizations if needed
    logging.info(f"Evaluation results saved at {args_eval['predictions_save_dir']}")


