import os
import glob
import warnings
import rasterio

import numpy as np
import pandas as pd
from numpy.random import RandomState

from deepforest import preprocess
from deepforest.utilities import shapefile_to_annotations


def split_raster(annotations_file,
                 path_to_raster=None,
                 numpy_image=None,
                 base_dir=".",
                 patch_size=400,
                 patch_overlap=0.05,
                 allow_empty=False,
                 image_name=None,
                 max_empty=0.9):
    """Divide a large tile into smaller arrays. Each crop will be saved to
    file.

    Args:
        numpy_image: a numpy object to be used as a raster, usually opened from rasterio.open.read()
        path_to_raster: (str): Path to a tile that can be read by rasterio on disk
        annotations_file (str): Path to annotations file (with column names)
            data in the format -> image_path, xmin, ymin, xmax, ymax, label
        base_dir (str): Where to save the annotations and image
            crops relative to current working dir
        patch_size (int): Maximum dimensions of square window
        patch_overlap (float): Percent of overlap among windows 0->1
        allow_empty: If True, include images with no annotations
            to be included in the dataset
        image_name (str): If numpy_image arg is used, what name to give the raster?
        max_empty (float): How much of the image can be no data

    Returns:
        A pandas dataframe with annotations file for training.
    """

    # Load raster as image
    # Load raster as image
    if (numpy_image is None) & (path_to_raster is None):
        raise IOError(
            "supply a raster either as a path_to_raster or if ready from existing in memory numpy object, as numpy_image=")

    if path_to_raster:
        numpy_image = rasterio.open(path_to_raster).read()
        numpy_image = np.moveaxis(numpy_image, 0, 2)
    else:
        if image_name is None:
            raise (IOError(
                "If passing an numpy_image, please also specify a image_name to match the column in the annotation.csv file"))

    # Check that its 3 band
    bands = numpy_image.shape[2]
    if not bands == 3:
        warnings.warn("Input rasterio had non-3 band shape of {}, ignoring alpha channel".format(numpy_image.shape))
        try:
            numpy_image = numpy_image[:, :, :3].astype("uint8")
        except:
            raise IOError("Input file {} has {} bands. DeepForest only accepts 3 band RGB "
                          "rasters in the order (height, width, channels). Selecting the first three bands failed, please reshape manually."
                          "If the image was cropped and saved as a .jpg, "
                          "please ensure that no alpha channel was used.".format(
                path_to_raster, bands))

    # Check that patch size is greater than image size
    height = numpy_image.shape[0]
    width = numpy_image.shape[1]
    if any(np.array([height, width]) < patch_size):
        raise ValueError("Patch size of {} is larger than the image dimensions {}".format(
            patch_size, [height, width]))

    # Compute sliding window index
    windows = preprocess.compute_windows(numpy_image, patch_size, patch_overlap)

    # Get image name for indexing
    if image_name is None:
        image_name = os.path.basename(path_to_raster)

        # Load annotations file and coerce dtype
    annotations = pd.read_csv(annotations_file)

    # open annotations file
    image_annotations = annotations[annotations.image_path == image_name]

    # Sanity checks
    if image_annotations.empty:
        raise ValueError(
            "No image names match between the file:{} and the image_path: {}. "
            "Reminder that image paths should be the relative "
            "path (e.g. 'image_name.tif'), not the full path "
            "(e.g. path/to/dir/image_name.tif)".format(annotations_file, image_name))

    if not all([
        x in annotations.columns
        for x in ["image_path", "xmin", "ymin", "xmax", "ymax", "label"]
    ]):
        raise ValueError("Annotations file has {} columns, should have "
                         "format image_path, xmin, ymin, xmax, ymax, label".format(
            annotations.shape[1]))

    annotations_files = []
    for index, window in enumerate(windows):

        # Crop image
        crop = numpy_image[windows[index].indices()]

        # skip if empty crop
        if crop.size == 0:
            continue
        if np.sum(crop != 0) < np.prod(crop.shape) * (1 - max_empty):
            continue

        # Find annotations, image_name is the basename of the path
        crop_annotations = preprocess.select_annotations(image_annotations, windows, index,
                                              allow_empty)

        # If empty images not allowed, select annotations returns None
        if crop_annotations is not None:
            # save annotations
            annotations_files.append(crop_annotations)

            # save image crop
            preprocess.save_crop(base_dir, image_name, index, crop)
    if len(annotations_files) == 0:
        raise ValueError(
            "Input file has no overlapping annotations and allow_empty is {}".format(
                allow_empty))

    annotations_files = pd.concat(annotations_files)

    # Checkpoint csv files, useful for parallelization
    # Use filename of the raster path to save the annotations
    image_basename = os.path.splitext(image_name)[0]
    file_path = image_basename + ".csv"
    file_path = os.path.join(base_dir, file_path)
    annotations_files.to_csv(file_path, index=False, header=True)

    return annotations_files


def preprocess_data(dir_tiles, image_path, annotations, patch_size=400, patch_overlap=0):
    """
        Create Tiles and CSV from Shapefile and Ortho

                    Keyword arguments
                    -----------------
                    dir_tiles:     directory for results
                    image_path:    orthomosaic as raster
                    annotations:   ground truth data as shapefile
                    patch_size:    tile size of crops (default=400)
                    patch_overlap: overlap of crops (default=0)

    """
    if not os.path.isdir(dir_tiles):
        os.makedirs(dir_tiles)

    saved = dir_tiles + "\\" + "csv_ref_wholepic.csv"

    df = shapefile_to_annotations(
        shapefile=annotations,
        rgb=image_path,
        savedir=saved
    )

    df.to_csv(saved, index=False)

    preprocess.split_raster(
        path_to_raster=image_path,
        annotations_file=saved,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        base_dir=dir_tiles,
        allow_empty=False
    )


def annotations_split(path, file_name, split, seed):
    """
    Splits a specific .csv-file containing image annotations into a train- and a test-file. The split is done on
    images, not on annotations.

                Keyword arguments
                -----------------
                path:       Path to the specific .csv-file
                file_name:  Name of the specific .csv-file
                split:      Percentage of crops used for test-file (default=0.3)
                seed:       Random seed for annotation splitting (default=None -> Random)
    """

    annotations = pd.read_csv(path + "\\" + file_name)
    amount = sorted(annotations["image_path"].unique())

    assert len(amount) > 1, "Annotation file contains a single or no annotations."

    train_amount = int(len(amount) * (1 - split))
    if seed is not None:
        r = RandomState(seed)
        train_indices = r.choice(amount, size=train_amount, replace=False)
    else:
        train_indices = np.random.choice(amount, size=train_amount, replace=False)
    test_indices = [index for index in amount if index not in train_indices]

    train_annotations = annotations[annotations["image_path"].isin(train_indices)]
    test_annotations = annotations[annotations["image_path"].isin(test_indices)]

    train_annotations.to_csv(path + r"\train_" + file_name, index=False)
    test_annotations.to_csv(path + r"\test_" + file_name, index=False)


def annotations_merge(path, file_name):
    """
    Merges all .csv files in a given directory. Will ignore 'csv_ref_wholepic.csv'.

                Keyword arguments
                -----------------
                path:       Path containing the relevant .csv files
                file_name:  Name to give the resulting merged file
    """
    os.chdir(path)
    annotation_files = glob.glob('*.csv')
    all_annotations = None

    for file in annotation_files:
        if file != "csv_ref_wholepic.csv":
            annotations = pd.read_csv(path + "\\" + file)

            if all_annotations is None:
                all_annotations = annotations
            else:
                all_annotations = pd.concat([all_annotations, annotations], ignore_index=True)

    all_annotations.to_csv(path + "\\" + file_name, index=False)
