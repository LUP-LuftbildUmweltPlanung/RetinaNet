
import os
import glob
import warnings
import rasterio
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import RandomState
import geopandas as gpd
from deepforest import preprocess
from shapely import geometry
from deepforest.utilities import shapefile_to_annotations


def split_raster(annotations_file=None,
                 path_to_raster=None,
                 numpy_image=None,
                 root_dir=None,
                 base_dir=None,
                 patch_size=400,
                 patch_overlap=0.05,
                 allow_empty=False,
                 image_name=None,
                 save_dir=".",
                 max_empty=0.02):
    """Divide a large tile into smaller arrays. Each crop will be saved to
    file.

    Args:
        numpy_image: a numpy object to be used as a raster, usually opened from rasterio.open.read(), in order (height, width, channels)
        root_dir: (str): Root directory of annotations file, if not supplied, will be inferred from annotations_file
        path_to_raster: (str): Path to a tile that can be read by rasterio on disk
        annotations_file (str or pd.DataFrame): A pandas dataframe or path to annotations csv file to transform to cropped images. In the format -> image_path, xmin, ymin, xmax, ymax, label. If None, allow_empty is ignored and the function will only return the cropped images.
        save_dir (str): Directory to save images
        base_dir (str): Directory to save images
        patch_size (int): Maximum dimensions of square window
        patch_overlap (float): Percent of overlap among windows 0->1
        allow_empty: If True, include images with no annotations
            to be included in the dataset. If annotations_file is None, this is ignored.
        image_name (str): If numpy_image arg is used, what name to give the raster?
    Note:
        When allow_empty is True, the function will return 0's for coordinates, following torchvision style, the label will be ignored, so for continuity, the first label in the annotations_file will be used.
    Returns:
        If annotations_file is provided, a pandas dataframe with annotations file for training. A copy of this file is written to save_dir as a side effect.
        If not, a list of filenames of the cropped images.
    """
    # Set deprecation warning for base_dir and set to save_dir
    if base_dir:
        warnings.warn(
            "base_dir argument will be deprecated in 2.0. The naming is confusing, the rest of the API uses 'save_dir' to refer to location of images. Please use 'save_dir' argument.",
            DeprecationWarning)
        save_dir = base_dir

    # Load raster as image
    if numpy_image is None and path_to_raster is None:
        raise IOError("Supply a raster either as a path_to_raster or if ready "
                      "from existing in-memory numpy object, as numpy_image=")

    if path_to_raster:
        numpy_image = rasterio.open(path_to_raster).read()
        numpy_image = np.moveaxis(numpy_image, 0, 2)
    else:
        if image_name is None:
            raise IOError("If passing a numpy_image, please also specify an image_name"
                          " to match the column in the annotation.csv file")

    # Confirm that raster is H x W x C, if not, convert, assuming image is wider/taller than channels
    if numpy_image.shape[0] < numpy_image.shape[-1]:
        warnings.warn(
            "Input rasterio had shape {}, assuming channels first. Converting to channels last"
            .format(numpy_image.shape), UserWarning)
        numpy_image = np.moveaxis(numpy_image, 0, 2)

    # Check that it's 3 bands
    bands = numpy_image.shape[2]
    if not bands == 3:
        warnings.warn(
            "Input rasterio had non-3 band shape of {}, ignoring "
            "alpha channel".format(numpy_image.shape), UserWarning)
        try:
            numpy_image = numpy_image[:, :, :3].astype("uint8")
        except:
            raise IOError("Input file {} has {} bands. "
                          "DeepForest only accepts 3 band RGB rasters in the order "
                          "(height, width, channels). "
                          "Selecting the first three bands failed, "
                          "please reshape manually. If the image was cropped and "
                          "saved as a .jpg, please ensure that no alpha channel "
                          "was used.".format(path_to_raster, bands))

    # Check that patch size is greater than image size
    height, width = numpy_image.shape[0], numpy_image.shape[1]
    if any(np.array([height, width]) < patch_size):
        raise ValueError("Patch size of {} is larger than the image dimensions {}".format(
            patch_size, [height, width]))

    # Compute sliding window index
    windows = preprocess.compute_windows(numpy_image, patch_size, patch_overlap)

    # Get image name for indexing
    if image_name is None:
        image_name = os.path.basename(path_to_raster)

    # Load annotations file and coerce dtype
    if annotations_file is None:
        allow_empty = True
    elif type(annotations_file) == str:
        annotations = preprocess.read_file(annotations_file, root_dir=root_dir)
    elif type(annotations_file) == pd.DataFrame:
        if root_dir is None:
            raise ValueError(
                "If passing a pandas DataFrame with relative pathnames in image_path, please also specify a root_dir"
            )
        annotations = preprocess.read_file(annotations_file, root_dir=root_dir)
    elif type(annotations_file) == gpd.GeoDataFrame:
        annotations = annotations_file
    else:
        raise TypeError(
            "Annotations file must either be None, a path, Pandas Dataframe, or Geopandas GeoDataFrame, found {}"
            .format(type(annotations_file)))

    # Select matching annotations
    if annotations_file is not None:
        image_annotations = annotations[annotations.image_path == image_name]
    image_basename = os.path.splitext(image_name)[0]
    image_basename = os.path.splitext(image_name)[0]

    # Sanity checks
    if not allow_empty:
        if image_annotations.empty:
            raise ValueError(
                "No image names match between the file:{} and the image_path: {}. "
                "Reminder that image paths should be the relative "
                "path (e.g. 'image_name.tif'), not the full path "
                "(e.g. path/to/dir/image_name.tif)".format(annotations_file, image_name))

    annotations_files = []
    crop_filenames = []
    empty_crop_count = 0  # Counter for empty crops
    for index, window in enumerate(windows):
        # Crop image
        crop = numpy_image[windows[index].indices()]


        # Skip if empty crop
        if crop.size == 0:
            empty_crop_count += 1
            if empty_crop_count > max_empty:  # Check if max_empty is exceeded
                raise ValueError(f"Maximum number of empty crops ({max_empty}) exceeded.")
            continue
        if np.sum(crop != 0) < np.prod(crop.shape) * (1 - max_empty):
            continue
    



        # Find annotations, image_name is the basename of the path
        if annotations_file is not None:
            crop_annotations = preprocess.select_annotations(image_annotations,
                                                  window=windows[index])
            crop_annotations["image_path"] = "{}_{}.png".format(image_basename, index)
            if crop_annotations.empty:
                if allow_empty:
                    geom_type = preprocess.determine_geometry_type(image_annotations)
                    # The safest thing is to use the first label and it will be ignored
                    crop_annotations.loc[0, "label"] = image_annotations.label.unique()[0]
                    crop_annotations.loc[0, "image_path"] = "{}_{}.png".format(
                        image_basename, index)
                    if geom_type == "box":
                        crop_annotations.loc[0, "xmin"] = 0
                        crop_annotations.loc[0, "ymin"] = 0
                        crop_annotations.loc[0, "xmax"] = 0
                        crop_annotations.loc[0, "ymax"] = 0
                    elif geom_type == "point":
                        crop_annotations.loc[0, "geometry"] = geometry.Point(0, 0)
                        crop_annotations.loc[0, "x"] = 0
                        crop_annotations.loc[0, "y"] = 0
                    elif geom_type == "polygon":
                        crop_annotations.loc[0, "geometry"] = geometry.Polygon([(0, 0),
                                                                                (0, 0),
                                                                                (0, 0)])
                        crop_annotations.loc[0, "polygon"] = 0
                else:
                    continue

            annotations_files.append(crop_annotations)

        # Save image crop
        if allow_empty or crop_annotations is not None:
            crop_filename = preprocess.save_crop(save_dir, image_name, index, crop)
            crop_filenames.append(crop_filename)

    if annotations_file is None:
        return crop_filenames
    elif len(annotations_files) == 0:
        raise ValueError(
            "Input file has no overlapping annotations and allow_empty is {}".format(
                allow_empty))
    else:
        annotations_files = pd.concat(annotations_files)

        # Checkpoint csv files, useful for parallelization
        # use the filename of the raster path to save the annotations
        image_basename = os.path.splitext(image_name)[0]
        file_path = os.path.join(save_dir, f"{image_basename}.csv")
        annotations_files.to_csv(file_path, index=False, header=True)

        return annotations_files





def preprocess_data(dir_tiles, image_path, annotations, patch_size=400, patch_overlap=0,max_empty=0.2):
    """
    Create Tiles and CSV from Shapefile and Ortho
    
    Keyword arguments:
    -------------------
    dir_tiles: Directory for results
    image_path: Orthomosaic as raster
    annotations: Ground truth data as shapefile
    patch_size: Tile size of crops (default=400)
    patch_overlap: Overlap of crops (default=0)
    """
    
    if not os.path.isdir(dir_tiles):
        os.makedirs(dir_tiles)
    
    saved = dir_tiles + "\\" + "csv_ref_wholepic.csv"
    
    # Simulating file processing with tqdm
    df = shapefile_to_annotations(
        shapefile=annotations,
        rgb=image_path,
        save_dir=saved
    )
    
    df.to_csv(saved, index=False)

    # If you're processing a list of files or batches
    file_list = ["file1", "file2", "file3", "file4"]  # Replace with actual file list or data
    
    for file in tqdm(file_list, desc="Processing image files", unit="file"):
        # Simulate a time-consuming task for each file
        time.sleep(2)  # Simulate processing each file

        # Call the modified split_raster function with max_empty
        split_raster(
            annotations_file=saved,
            path_to_raster=image_path,
            patch_size=400,
            patch_overlap=0.05,
            max_empty=max_empty,  # Limit empty crops to 10
            save_dir=dir_tiles,

        )
    
    print("Data preprocessing and split completed.")



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
    Merges all .csv files in a given directory and saves them in the required format.
    Will ignore 'csv_ref_wholepic.csv' and include only the required columns.
    
    Keyword arguments:
    -------------------
    path:       Path containing the relevant .csv files
    file_name:  Name to give the resulting merged file
    """
    os.chdir(path)
    annotation_files = glob.glob('*.csv')

    all_annotations = None

    for file in annotation_files:
        if file != "csv_ref_wholepic.csv":  # Ignore the reference file
            annotations = pd.read_csv(file)
            
            # Remove the 'geometry' column if it exists
            if 'geometry' in annotations.columns:
                annotations = annotations.drop(columns='geometry')

            # Ensure the required columns are present
            required_columns = ['xmin', 'ymin', 'xmax', 'ymax', 'label', 'image_path']
            if not all(col in annotations.columns for col in required_columns):
                raise ValueError(f"Missing one of the required columns: {required_columns}")

            # Fill missing labels with a default label if necessary
            annotations['label'] = annotations['label'].fillna('Tree')  # Default label for missing entries

            # Round the coordinate columns to integers to remove decimals
            annotations['xmin'] = annotations['xmin'].round(0).astype(int)
            annotations['ymin'] = annotations['ymin'].round(0).astype(int)
            annotations['xmax'] = annotations['xmax'].round(0).astype(int)
            annotations['ymax'] = annotations['ymax'].round(0).astype(int)

            # Add the current annotations to the merged DataFrame
            if all_annotations is None:
                all_annotations = annotations
            else:
                all_annotations = pd.concat([all_annotations, annotations], ignore_index=True)

    # Reorder columns to ensure 'image_path' is the first column
    all_annotations = all_annotations[['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label']]

    # Save the merged annotations as a CSV
    all_annotations.to_csv(os.path.join(path, file_name), index=False)

    print(f"Merged annotations saved to: {os.path.join(path, file_name)}")



