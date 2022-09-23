# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 11:59:17 2021

@author: Benny, Manel
"""

import glob
import os
import warnings
from pathlib import Path

import pandas as pd

from data import preprocess_data, annotations_split, annotations_merge
from params import *
from predict import predict_in_tiles
from train import train_model
from validate import validate, visualize_validation


class RetinaNet:
    """Contains all functions available within the Deepforest RetinaNet framework.

                Functions
                ---------
                preprocess:         processes and prepares data for training
                annotations_merge:  merges .csv files
                annotations_split:  splits a .csv file into train and test data
                train:              trains a model on data
                validate:           validates a model on data
                validate_all:       validates all models in a folder and provides a comparison
                predict:            predicts on a given image using a provided model
    """

    def __init__(self):
        self.csv = None
        self.path = None
        self.label_dict = None

    def preprocess(self, annotations, image_path, directory_to_save_crops, patch_size=400, patch_overlap=0.0,
                   merge_name=None, split=0.3, seed=None):
        """
        Creates training images by splitting the provided image in small chunks. Further creates the relevant
        .csv-files from the provided .shp-file.

                    Keyword arguments
                    -----------------
                    annotations:                Single .shp-file or folder containing .shp-files
                    image_path:                 Single .tif-file or folder containing .tif-files
                    directory_to_save_crops:    Directory to save the resulting crops in
                    patch_size:                 Size of resulting crops (square, default=400)
                    patch_overlap:              Overlap of resulting crops (default=0.0)
                    merge_name:                 Name of the resulting annotations-file if merged
                                                (default=csv_ref_merged.csv)
                    split:                      Percentage of crops used for test-file (default=0.3)
                    seed:                       Random seed for annotation splitting (default=None -> Random)

                    Example
                    -------
                    annotations = "C:/Path/to/annotations.shp"
                    image_path = "C:/Path/to/image" or r"C:/Path/to/image/folder"
                    directory_to_save_crops = r"C:/path/to/save/folder"
                    r.preprocess(annotations, image_path, directory_to_save_crops, patch_size=200, patch_overlap=0.2,
                    annot_merge=True, annot_split=False)
        """

        self.path = directory_to_save_crops
        if image_path.endswith('.tif'):
            preprocess_data(self.path, image_path, annotations, patch_size, patch_overlap)
        else:
            owd = os.getcwd()
            os.chdir(image_path)
            tif_files = glob.glob('*.tif')
            os.chdir(owd)
            if annotations.endswith('.shp'):
                for tif in tif_files:
                    preprocess_data(self.path, image_path + "\\" + tif, annotations, patch_size,
                                    patch_overlap)
            else:
                os.chdir(annotations)
                shp_files = glob.glob('*.shp')
                assert len(tif_files) == len(shp_files), "multiple .shp- and .tif-files, but mount of .shp-files " \
                                                         "does not match amount of .tif-files."
                for tif, shp in zip(tif_files, shp_files):
                    preprocess_data(self.path, image_path + "\\" + tif, shp, patch_size,
                                    patch_overlap)

        if merge_name is None:
            merge_name = "csv_ref_merged.csv"
        self.annotations_merge(self.path, merge_name)
        self.csv = merge_name
        if split > 0:
            self.annotations_split(self.path, self.csv, split, seed=seed)

    @staticmethod
    def annotations_merge(path, file_name):
        """
        Merges all .csv files in a given directory. Will ignore 'csv_ref_wholepic.csv'.

                    Keyword arguments
                    -----------------
                    path:       Path containing the relevant .csv files
                    file_name:  Name to give the resulting merged file

                    Example
                    -------
                    r.annotations_merge(path=r"C:/Path/to/folder/with/csv_files")
        """
        annotations_merge(path, file_name)

    @staticmethod
    def annotations_split(path, file_name, split=0.3, seed=None):
        """
        Splits a specific .csv-file containing image annotations into a train- and a test-file. The split is done on
        images, not on annotations.

                    Keyword arguments
                    -----------------
                    path:       Path to the specific .csv-file
                    file_name:  Name of the specific .csv-file
                    split:      Percentage of crops used for test-file (default=0.3)
                    seed:       Random seed for annotation splitting (default=None -> Random)

                    Example
                    -------
                    r.annotations_split(path='C:/Path/to/folder/with/csv_file', file_name='file_name.csv')
        """
        annotations_split(path, file_name, split, seed)

    def train(self, end_model, annotations_file=None, start_model=None, gpu=True, epochs=10, lr=0.0001, batch_size=4,
              multi_class=False, checkpoint_frequency=None, label_dict=None):
        """
        Trains a model on data provided in the annotations file.

                    Keyword arguments
                    -----------------
                    end_model:          Name of the stored model file
                    annotations_file:   Annotations file to be used for training (default=None -> looks in self.csv)
                    start_model:        If provided, the start model will be trained further. Otherwise, a new model is
                                        created (default=None -> new model)
                    gpu:                If a GPU should be used for training (default=True)
                    epochs:             Epochs to be trained for (default=10)
                    lr:                 Learning rate to be used when the model is updates (default=0.0001)
                    batch_size:         Amount of single files to be trained on at the same time (default=4)
                    multi_class:        If the annotations file contains more than one class (default=False)

                    Example
                    -------
                    end_model = "C:/Path/to/model.pl"
                    annotations_file = "C:/path/to/annotations.csv"
                    label_dict = {'label_0': 0,'label_1': 1,'label_2': 2, ...}
                    r.train(end_model, annotations_file, epochs=120, label_dict=label_dict, multi_class=True,
                    batch_size=5, checkpoint_frequency=5, lr=0.0001)
        """
        if annotations_file is None:
            assert self.csv is not None, "No annotations file found in memory. Please provide a annotations file."
            annotations_file = self.csv

        train_model(start_model, end_model, annotations_file, gpu, epochs, lr, batch_size, multi_class,
                    checkpoint_frequency, label_dict)

    @staticmethod
    def validate(annotations_file, predict_model=None, multi=False, label_dict=None, labels_order=None, save=False,
                 gpu=True):
        """
        Validates a model on data provided in the annotations file.

                    Keyword arguments
                    -----------------
                    annotations_file:   Annotations file to be used during validation
                    predict_model:      Model to be validated
                    multi:              If the annotations file contains multiple classes (default=False)
                    label_dict:         Label dictionary as created by the model during training
                                        (default=None, only required when training with multiple classes)
                    save:               If prediction images during validation should be stored (default=False)
                    gpu:                If GPU should be used for prediction (default=True)

                    Example
                    -------
                    annotations_file_vali = "C:/path/to/annotations.csv"
                    vali_model = "C:/Path/to/model.pl"
                    label_dict = {'label_0': 0, 'label_1': 1, 'label_2': 2}
                    multi_class = True
                    r.validate(annotations_file, vali_model, multi_class, label_dict, save=False)
        """
        validate(annotations_file, predict_model, multi, label_dict, labels_order, save, gpu)

    @staticmethod
    def validate_all(annotations_file, predict_model_folder, multi=None, label_dict=None, labels_order=None, save=True,
                     gpu=True, score=0.1):
        """
        Validates all models in the given folder on data provided in the annotations file. Provides an overview of
        all compared models.

                    Keyword arguments
                    -----------------
                    annotations_file:       Annotations file to be used during validation
                    predict_model_folder:   Model to be validated
                    multi:                  If the annotations file contains multiple classes (default=False)
                    label_dict:             Label dictionary as created by the model during training
                                            (default=None, only required when training with multiple classes)
                    save:                   If prediction images during validation should be stored (default=False)
                    gpu:                    If GPU should be used for prediction (default=True)

                    Example
                    -------
                    annotations_file = "C:/path/to/annotations.csv"
                    predict_model_folder = "C:/Pathto/folder/with/models"
                    label_dict = {'label_0': 0,'label_1': 1,'label_2': 2, ...}
                    r.validate_all(annotations_file, predict_model_folder, multi=True, label_dict=label_dict,
                    save=False, score=0.1)
        """
        directory = Path(predict_model_folder)
        ori_dir = os.getcwd()
        os.chdir(directory)
        models = [directory / file for file in glob.glob('*.pl')]
        os.chdir(ori_dir)

        models_no_path = []
        precision_all = []
        recall_all = []
        fscore_all = []
        # support_all = []
        for m in models:
            precision, recall, fscore, support = validate(annotations_file, m, multi, label_dict,
                                                          labels_order, save, gpu, score)
            models_no_path.append(str(m).rsplit('\\', 1)[1])
            precision_all.append(precision)
            recall_all.append(recall)
            fscore_all.append(fscore)
            # support_all.append(support)

        scores = pd.DataFrame({'Precision': precision_all, 'Recall': recall_all, 'F1_Score': fscore_all},
                              index=models_no_path)  # 'Support': support_all

        episodes = scores.index.values.tolist()
        for idx, e in enumerate(episodes):
            value = e.rsplit('.', 1)[0].rsplit('_', 1)[-1]
            episodes[idx] = int(value)

        scores['Episode'] = episodes
        scores.sort_values(by='Episode', inplace=True)
        scores.to_csv(str(directory / 'Model_validation_overview.csv'))
        # scores=pandas.read_csv(str(directory / 'Model_validation_overview.csv'),index_col=0)
        # print(scores)
        visualize_validation(scores, path=str(directory / 'Model_validation_overview.png'))

    @staticmethod
    def predict(image_file, output_shape, predict_model, multi=False, label_dict=None, gpu=1, score_threshold=0.05,
                patch_size=400, patch_overlap=0.0):
        """
        Predicts on a single image file using the provided model.

                    Keyword arguments
                    -----------------
                    image_file:         Annotations file to be used during validation
                    output_shape:       Path+File_Name for storing the prediction
                    predict_model:      Model to be validated
                    multi:              If the annotations file contains multiple classes (default=False)
                    label_dict:         Label dictionary as created by the model during training
                                        (default=None, only required when training with multiple classes)
                    gpu:                If GPU should be used for prediction (default=True)
                    score_threshold:    Threshold for considering something a detection (default=0.05)
                    patch_size:         Size of resulting crops (square, default=400)
                    patch_overlap:      Overlap of resulting crops (default=0.0)

                    Example
                    -------
                    image_file = "C:/Path/to/image.tif"
                    output_shape = "C:/Path/to/output.shp"
                    pre_model = "C:/Path/to/model.pl"
                    label_dict = {'label_0': 0,'label_1': 1,'label_2': 2, ...}
                    r.predict(image_file, output_shape, pre_model, multi=True, label_dict=label_dict, gpu=True,
                    patch_size=400, patch_overlap=0.3)
        """
        if image_file.endswith('.tif') or image_file.endswith('.jpg') or image_file.endswith('.png'):
            predict_in_tiles(image_file, output_shape, predict_model, multi, label_dict, gpu, score_threshold,
                             patch_size, patch_overlap)
        else:
            os.chdir(image_file)
            tif_files = glob.glob('*.tif')
            for idx, tif in enumerate(tif_files):
                print('Running prediction for ' + tif)
                predict_in_tiles(image_file + '\\' + tif, output_shape[:-4] + '_' + str(idx) + '.shp', predict_model,
                                 multi, label_dict, gpu, score_threshold, patch_size, patch_overlap)


if __name__ == '__main__':
    r = RetinaNet()

    if enable_extra_parameters:
        warnings.warn("Extra parameters are enabled. Code may behave in unexpected ways. "
                      "Please disable unless experienced with the code.")
    else:
        seed = None
        save = True
        score_threshold = 0.05

    if Create_tiles:
        r.preprocess(annotations=annotations, image_path=image_path, directory_to_save_crops=directory_to_save_crops,
                     patch_size=patch_size,
                     patch_overlap=patch_overlap, split=split, seed=seed)

    if Train:
        r.train(end_model=end_model, annotations_file=annotations_file, epochs=EPOCHS, label_dict=label_dict,
                multi_class=multi_class,
                batch_size=BATCH_SIZE, checkpoint_frequency=CHECKPOINT_FREQUENCY, lr=LEARNING_RATE)

    if Validate:
        if vali_model.endswith('.pl'):
            r.validate(annotations_file=annotations_file, predict_model=vali_model, multi=multi_class,
                       label_dict=label_dict, save=save)
        else:
            r.validate_all(annotations_file=annotations_file, predict_model_folder=vali_model, multi=multi_class,
                           label_dict=label_dict, save=save)

    if Predict:
        r.predict(image_file=image_file, output_shape=output_shape, predict_model=predict_model, multi=multi_class,
                  label_dict=label_dict,
                  patch_size=patch_size, patch_overlap=patch_overlap, score_threshold=score_threshold)
