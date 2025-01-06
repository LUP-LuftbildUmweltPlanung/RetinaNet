# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 00:21:57 2024

@author: QuadroRTX
"""

import os
import glob
import shutil
import geopandas as gpd
from pathlib import Path
from data import annotations_split, annotations_merge  # Assuming these are implemented elsewhere
from data import preprocess_data


def get_image_name(file_path):
    """
    Extracts the file name (without extension) from a given file path.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: File name without the extension.
    """
    return os.path.splitext(os.path.basename(file_path))[0]  # Returns the file name without extension

def preprocess(
    annotations,
    image_path,
    directory_to_save_crops,
    patch_size=400,
    patch_overlap=0.0,
    merge_name=None,
    split=0.3,
    seed=None,
    label=None,
    max_empty=None,
):
    """
    Prepares training data by splitting large images into smaller patches and creating
    relevant annotation files from shapefiles.

    Args:
        annotations (str or list): Path(s) to the shapefile(s) containing annotations.
        image_path (str or list): Path(s) to the image(s) corresponding to the annotations.
        directory_to_save_crops (str): Directory where the cropped images will be saved.
        patch_size (int): Size of the image patches (default=400).
        patch_overlap (float): Overlap between patches as a fraction of the patch size (default=0.0).
        merge_name (str): Name of the merged annotation file (default=None).
        split (float): Fraction of data to use for testing (default=0.3).
        seed (int): Random seed for splitting annotations (default=None).
        label (dict or str): Label(s) to assign to annotations (default=None).
        max_empty (int): Maximum number of empty patches allowed (default=None).
    """
    def ensure_label_column(shp, label):
        """
        Ensures the 'label' column exists in the given shapefile.

        Args:
            shp (str): Path to the shapefile.
            label (dict or str): Labels to assign.

        Returns:
            str: Path to the updated shapefile.
        """
        gdf = gpd.read_file(shp)
        print(f"Processing {shp}. Initial columns: {gdf.columns}")

        # Remove extra spaces from column names
        gdf.columns = gdf.columns.str.strip()

        # Generate bounding box columns if missing
        required_columns = ['xmin', 'ymin', 'xmax', 'ymax']
        if not all(col in gdf.columns for col in required_columns):
            if 'geometry' in gdf.columns:
                bounds = gdf.geometry.bounds
                gdf['xmin'] = bounds['minx']
                gdf['ymin'] = bounds['miny']
                gdf['xmax'] = bounds['maxx']
                gdf['ymax'] = bounds['maxy']
            else:
                raise KeyError(f"Missing 'geometry' column in {shp}. Cannot generate bounding boxes.")

        # Add the 'label' column if missing
        if 'label' not in gdf.columns:
            if isinstance(label, dict):
                image_name = get_image_name(shp)
                if image_name in label:
                    gdf['label'] = label[image_name]
                else:
                    print(f"No label found for {image_name}. Assigning default value 'Tree'.")
                    gdf['label'] = label
            else:
                print(f"Adding single value label: {label}")
                gdf['label'] = label

        # Save updated shapefile
        updated_shp = shp.replace(".shp", "_updated.shp")
        gdf.to_file(updated_shp, driver="ESRI Shapefile")
        print(f"Updated shapefile saved to {updated_shp} with columns: {gdf.columns}")
        return updated_shp

    # Ensure directory exists
    os.makedirs(directory_to_save_crops, exist_ok=True)
    print(f"Directory to save crops: {directory_to_save_crops}")

    # Process images and annotations
    if isinstance(image_path, list):
        assert isinstance(annotations, list), "If 'image_path' is a list, 'annotations' must also be a list"
        assert len(image_path) == len(annotations), "Mismatch between number of images and annotations"
        for img, shp in zip(image_path, annotations):
            print(f"Processing image: {img} with annotations: {shp}")
            shp = ensure_label_column(shp, label)
            preprocess_data(directory_to_save_crops, img, shp, patch_size, patch_overlap, max_empty)
    else:
        if image_path.endswith('.tif'):
            print(f"Processing single image: {image_path} with annotations: {annotations}")
            annotations = ensure_label_column(annotations, label)
            preprocess_data(directory_to_save_crops, image_path, annotations, patch_size, patch_overlap, max_empty)
        else:
            tif_files = glob.glob(os.path.join(image_path, "*.tif"))
            if annotations.endswith('.shp'):
                for tif in tif_files:
                    print(f"Processing image: {tif} with annotations: {annotations}")
                    annotations = ensure_label_column(annotations, label)
                    preprocess_data(directory_to_save_crops, tif, annotations, patch_size, patch_overlap, max_empty)
            else:
                shp_files = glob.glob(os.path.join(annotations, "*.shp"))
                assert len(tif_files) == len(shp_files), "Number of .tif files does not match .shp files"
                for tif, shp in zip(tif_files, shp_files):
                    print(f"Processing image: {tif} with annotations: {shp}")
                    shp = ensure_label_column(shp, label)
                    preprocess_data(directory_to_save_crops, tif, shp, patch_size, patch_overlap, max_empty)

    # Merge annotations
    merge_name = merge_name or "csv_ref_merged.csv"
    print(f"Merging annotations into: {merge_name}")
    annotations_merge(directory_to_save_crops, merge_name)

    # Split annotations into train/test
    if split > 0:
        print(f"Splitting annotations into train/test with split ratio: {split}")
        annotations_split(directory_to_save_crops, merge_name, split, seed=seed)
