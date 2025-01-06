# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 00:01:12 2024

@author: QuadroRTX
"""

import os
import numpy as np
import pandas as pd
import rasterio as rio
import torch
from torch import cuda
import geopandas as gpd
import rasterio
import importlib
import typing
import warnings
from shapely.geometry import box
from torchvision.ops import nms
from deepforest import main, dataset, visualize, utilities
from deepforest import predict as predict_utils




def mosiac(boxes, windows, sigma=0.5, thresh=0.001, iou_threshold=0.1):
    """
    Merge predictions from overlapping windows by transforming their coordinates
    to the original system and applying Non-Max Suppression (NMS) to reduce duplicate detections.
    
    Args:
        boxes (list of pd.DataFrame): A list of DataFrames, where each DataFrame contains bounding box 
            predictions from a single window. Each DataFrame should include columns `xmin`, `ymin`, `xmax`, `ymax`, 
            `label`, and `score`.
        windows (list of rasterio.windows.Window): A list of windows representing the coordinates of the tiles 
            relative to the original image.
        sigma (float): Variance of the Gaussian function used in Gaussian Soft NMS (if applicable). 
            Currently unused in the implementation.
        thresh (float): Confidence score threshold. Bounding boxes with scores below this threshold 
            are removed after NMS is applied.
        iou_threshold (float): Intersection-over-Union (IoU) threshold used during Non-Max Suppression 
            to filter overlapping bounding boxes. Lower values result in more aggressive suppression.
    
    Returns:
        pd.DataFrame: A DataFrame containing the filtered bounding boxes after NMS and score thresholding. 
        The DataFrame includes the following columns:
            - `xmin`: Minimum x-coordinate of the bounding box.
            - `ymin`: Minimum y-coordinate of the bounding box.
            - `xmax`: Maximum x-coordinate of the bounding box.
            - `ymax`: Maximum y-coordinate of the bounding box.
            - `label`: Class label of the detected object.
            - `score`: Confidence score of the detection.
    
    Raises:
        ValueError: If the `boxes` or `windows` arguments are empty or have mismatched lengths.
    
    Notes:
        - This function assumes that the bounding box coordinates in each tile are relative 
          to the tile's coordinate system. It adjusts these coordinates to the global image 
          coordinate system using the corresponding window offsets.
        - NMS is applied using PyTorch's torchvision.ops.nms function.
    
    Example:
        >>> boxes = [pd.DataFrame({'xmin': [10], 'ymin': [20], 'xmax': [30], 'ymax': [40], 'label': [1], 'score': [0.95]})]
        >>> windows = [rasterio.windows.Window(0, 0, 100, 100)]
        >>> result = mosaic(boxes, windows, iou_threshold=0.5, thresh=0.8)
        >>> print(result)
           xmin  ymin  xmax  ymax  label  score
        0    10    20    30    40      1   0.95
"""
    # Transform the coordinates to the original system
    for index, _ in enumerate(boxes):
        xmin, ymin, xmax, ymax = windows[index].getRect()
        boxes[index].xmin += xmin
        boxes[index].xmax += xmin
        boxes[index].ymin += ymin
        boxes[index].ymax += ymin

    predicted_boxes = pd.concat(boxes)
    print(
        f"{predicted_boxes.shape[0]} predictions in overlapping windows, applying non-max suppression"
    )

    # Move predictions to tensor
    boxes = torch.tensor(predicted_boxes[["xmin", "ymin", "xmax", "ymax"]].values,
                          dtype=torch.float32)
    scores = torch.tensor(predicted_boxes.score.values, dtype=torch.float32)
    labels = predicted_boxes.label.values

    # Perform Non-Max Suppression (NMS)
    bbox_left_idx = nms(boxes=boxes, scores=scores, iou_threshold=iou_threshold)

    bbox_left_idx = bbox_left_idx.numpy()
    new_boxes, new_labels, new_scores = boxes[bbox_left_idx].type(
        torch.int), labels[bbox_left_idx], scores[bbox_left_idx]

    # Recreate box DataFrame
    image_detections = np.concatenate([
        new_boxes,
        np.expand_dims(new_labels, axis=1),
        np.expand_dims(new_scores, axis=1)
    ],
                                      axis=1)

    mosaic_df = pd.DataFrame(image_detections,
                              columns=["xmin", "ymin", "xmax", "ymax", "label", "score"])

    print(f"{mosaic_df.shape[0]} predictions kept after non-max suppression")

    # **Apply the confidence threshold here**
    filtered_mosaic_df = mosaic_df[mosaic_df["score"] >= thresh]
    print(f"{filtered_mosaic_df.shape[0]} predictions kept after applying thresh={thresh}")

    return filtered_mosaic_df


def predict_tile(
    model_path,
    raster_path=None,
    image=None,
    patch_size=400,
    patch_overlap=0.05,
    iou_threshold=0.15,
    in_memory=True,
    mosaic=True,
    sigma=0.5,
    thresh=0.001,
    return_plot=False,
    color=None,
    thickness=1,
    crop_model=None,
    crop_transform=None,
    crop_augment=False,
):
    """
    Predict bounding boxes for large images by dividing them into overlapping tiles, processing
    each tile individually, and combining the results into a single output.

    Args:
        model_path (str): Path to the pre-trained model checkpoint. The model is loaded
            and used for prediction.
        raster_path (str, optional): Path to the raster image file on disk. Required if `image` is not provided.
        image (np.ndarray, optional): Image array in memory (BGR format). Used directly if `raster_path` is not provided.
        patch_size (int): Size of each tile (in pixels) used for dividing the image. Default is 400.
        patch_overlap (float): Overlap ratio between adjacent tiles. Default is 0.05 (5% overlap).
        iou_threshold (float): Intersection-over-Union (IoU) threshold for Non-Max Suppression. Default is 0.15.
        in_memory (bool): If True, the entire image is loaded into memory, increasing speed but requiring more RAM.
            If False, tiles are processed directly from disk.
        mosaic (bool): If True, combines all predictions into a single output DataFrame. If False, returns predictions
            for individual tiles.
        sigma (float): Variance of the Gaussian function used in Gaussian Soft NMS (if applicable). Default is 0.5.
        thresh (float): Confidence score threshold to filter predictions. Default is 0.001.
        return_plot (bool): If True, returns an image with bounding boxes overlaid. Deprecated in favor of
            `visualize.plot_results`.
        color (tuple, optional): Color of bounding boxes for visualization in BGR format. Example: (0, 255, 0) for green.
        thickness (int, optional): Thickness of bounding box lines in pixels for visualization. Default is 1.
        crop_model (object, optional): Optional model for predicting on cropped regions.
        crop_transform (object, optional): Torchvision transform to apply to image crops before prediction.
        crop_augment (bool, optional): If True, applies augmentations to image crops. Default is False.

    Returns:
        pd.DataFrame or list:
            - If `mosaic` is True: Returns a DataFrame with combined predictions, including columns:
                - `xmin`, `ymin`, `xmax`, `ymax`: Coordinates of bounding boxes.
                - `label`: Predicted class labels.
                - `score`: Confidence scores of predictions.
            - If `mosaic` is False: Returns a list of tuples, where each tuple contains a DataFrame of predictions
              and the corresponding image crop.
            - If `return_plot` is True: Returns an image with bounding boxes overlaid.

    Raises:
        ValueError: If neither `raster_path` nor `image` is provided, or if required parameters are invalid.
        FileNotFoundError: If the specified `model_path` or `raster_path` does not exist.

    Notes:
        - This function assumes the bounding box coordinates are in the tile's local coordinate system
          and transforms them to the global image coordinate system.
        - When multiple GPUs are detected, only the first GPU is used for predictions.
        - For visualization, use the `visualize.plot_results` method for more flexibility.

    Example:
        >>> predict_tile(
                model_path="model.ckpt",
                raster_path="image.tif",
                patch_size=400,
                patch_overlap=0.25,
                mosaic=True,
                thresh=0.1
            )
    """
    # Load model from the specified path
    model = model_path
    model.eval()

    # Access model configuration
    model.nms_thresh = model.config.get("nms_thresh", iou_threshold)
    if model.nms_thresh is None:
        raise ValueError("Unable to determine nms_thresh from the config.")

    print(f"Using NMS threshold: {model.nms_thresh}")

    # Check for multi-GPU and adjust config
    if cuda.device_count() > 1:
        warnings.warn(
            "More than one GPU detected. Using only the first GPU for predict_tile."
        )
        model.config["devices"] = 1
        model.create_trainer()

    # Validate input
    if raster_path is None and image is None:
        raise ValueError(
            "Either 'raster_path' or 'image' must be provided. Both cannot be None."
        )

    # Prepare the dataset
    if in_memory:
        if raster_path:
            with rio.open(raster_path) as src:
                image = src.read()
                image = np.moveaxis(image, 0, 2)  # Convert channels to last axis
        ds = dataset.TileDataset(
            tile=image, patch_overlap=patch_overlap, patch_size=patch_size
        )
    else:
        if raster_path is None:
            raise ValueError("raster_path is required if in_memory is False.")
        ds = dataset.RasterDataset(
            raster_path=raster_path,
            patch_overlap=patch_overlap,
            patch_size=patch_size,
        )

    # Predict on tiles
    batched_results = model.trainer.predict(model, model.predict_dataloader(ds))

    # Flatten predictions
    results = [box for batch in batched_results for box in batch]

    if mosaic:
        results = mosiac(
            results, ds.windows, sigma=sigma, thresh=thresh, iou_threshold=iou_threshold
        )
        results["label"] = results.label.apply(
            lambda x: model.numeric_to_label_dict[x]
        )

        # Add raster_path information
        if raster_path:
            results["image_path"] = os.path.basename(raster_path)

        # Optionally return plot
        if return_plot:
            warnings.warn(
                "return_plot is deprecated and will be removed in future versions. "
                "Use visualize.plot_results instead."
            )
            tile_image = (
                rio.open(raster_path).read() if raster_path else image[:, :, ::-1]
            )
            drawn_plot = visualize.plot_predictions(
                tile_image, results, color=color, thickness=thickness
            )
            return drawn_plot
        return results

    else:
        for df in results:
            df["label"] = df.label.apply(lambda x: model.numeric_to_label_dict[x])

        # Generate crops for non-mosaic output
        crops = []
        if raster_path:
            with rio.open(raster_path) as src:
                image = src.read()
                image = np.moveaxis(image, 0, 2)

        for window in ds.windows:
            crop = image[window.indices()]
            crops.append(crop)

        return list(zip(results, crops))







def predict_and_save_shapefile_with_transform(
    model,
    image_path,
    small_tiles=True,
    patch_size=400,
    patch_overlap=0.25,
    iou_threshold=0.1,
    thresh=0.1,
    savedir=None
):
    """
    Predict bounding boxes and save them as a shapefile with the same CRS as the raster,
    transforming pixel coordinates to geographic coordinates.
    """

    if small_tiles:
        # Predict using tiles
        predictions = predict_tile(
                        model_path=model,
                        raster_path=image_path,
                        patch_size=patch_size,           # Size of each tile
                        patch_overlap=patch_overlap,     # Overlap between tiles (25%)
                        iou_threshold=iou_threshold,    # Intersection Over Union threshold for merging boxes
                        return_plot=False,               # Set True to get an annotated image
                        mosaic=True,
                        sigma=0.5,
                        thresh=thresh,
                        color=(0, 255, 0),
                        thickness=1,
                        crop_model=None,
                        crop_transform=None,
                        crop_augment=False
                    )

        print(predictions["score"].describe())

    else:
        # Predict on the entire image
        predictions = model.predict_image(path=image_path)

    # Read the raster to get the CRS and affine transform
    with rasterio.open(image_path) as src:
        raster_crs = src.crs  # Get the CRS of the raster image
        transform = src.transform  # Get the affine transformation matrix

    # Create a list of geometries (bounding boxes)
    geometries = []
    for _, row in predictions.iterrows():
        # Convert pixel coordinates to geographic coordinates using the affine transform
        min_x, min_y = transform * (row["xmin"], row["ymin"])  # Transform from pixel to geographic
        max_x, max_y = transform * (row["xmax"], row["ymax"])

        # Create a bounding box (shapely geometry) from the geographic coordinates
        geom = box(min_x, min_y, max_x, max_y)
        geometries.append(geom)

    # Create a GeoDataFrame from the predictions and set CRS
    gdf = gpd.GeoDataFrame(
        predictions, 
        geometry=geometries, 
        crs=raster_crs  # Set CRS from the raster image
    )

    # Save the GeoDataFrame as a shapefile with the same name as the input image
    if savedir:
        # Extract filename without extension
        filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(savedir, f"{filename}_predictions.shp")
        gdf.to_file(output_path)
        print(f"Shapefile with predictions saved to {output_path}")
    else:
        print("No directory specified to save the shapefile.")

def process_all_tif_files_in_folder(model, folder_path, savedir, **kwargs):
    """
    Process all TIFF files in a folder and save predictions as shapefiles with the same name.
    """
    # List all files in the folder and filter out non-TIF files
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]

    # Process each TIFF file
    for tif_file in tif_files:
        image_path = os.path.join(folder_path, tif_file)
        print(f"Processing {image_path}...")

        # Call the prediction and saving function for each file
        predict_and_save_shapefile_with_transform(
            model=model,
            image_path=image_path,
            savedir=savedir,
            **kwargs
        )
