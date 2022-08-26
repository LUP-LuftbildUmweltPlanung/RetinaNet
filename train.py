import os
import json
import torch
import pandas as pd
from deepforest import main
from pathlib import Path


def train_model(start_model, end_model, annotations_file, gpu=True, epochs=30, lr=0.0001, batch_size=1,
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
    """

    if start_model is not None:
        # reload the checkpoint to model object
        if not multi_class:
            m = main.deepforest.load_from_checkpoint(start_model)
        else:
            m = main.deepforest(num_classes=len(label_dict), label_dict=label_dict)
            if gpu:
                m.to("cuda")
            else:
                m.to("cpu")
            ckpt = torch.load(start_model, map_location=torch.device("cuda" if gpu else "cpu"))
            m.load_state_dict(ckpt["state_dict"])
    else:
        if not multi_class:
            m = main.deepforest()
            m.use_release()
        else:
            if label_dict is None:
                data = pd.read_csv(annotations_file)
                classes = sorted(set(data['label']))
                num_classes = len(classes)
                assert num_classes > 1, "Annotations do not contain multiple classes. Please disable multi_class."
                label_dict = {cla: idx for idx, cla in enumerate(classes)}

            label_dir = os.path.dirname(end_model) + r"\label_dictionary"
            file_name = os.path.basename(end_model)[:-3]

            Path(label_dir).mkdir(parents=True, exist_ok=True)

            with open(label_dir + '\\' + file_name + '.txt', 'w') as file:
                file.write(json.dumps(label_dict))

            print("Automatically created label dictionary: ", label_dict)
            print("Dictionary saved as: " + label_dir + '\\' + file_name + '.txt')

            m = main.deepforest(num_classes=len(label_dict), label_dict=label_dict)

    if gpu:
        m.to("cuda")
    else:
        m.to("cpu")
        # import multiprocessing
        # m.config["workers"] = multiprocessing.cpu_count()-1
        m.config["workers"] = 1
    m.config["gpus"] = 1 if gpu else 0
    m.config["save-snapshot"] = False
    m.config["train"]["csv_file"] = annotations_file
    m.config["train"]["root_dir"] = os.path.dirname(annotations_file)
    # model.config["train"]["fast_dev_run"] = True
    m.config["train"]["epochs"] = epochs
    m.config["train"]["lr"] = lr
    m.config["train"]["batch_size"] = batch_size

    if checkpoint_frequency is None:
        m.create_trainer()
        m.trainer.fit(m)
        m.trainer.save_checkpoint(end_model)
    else:
        checkpoint = 0
        m.config["train"]["epochs"] = checkpoint_frequency
        m.create_trainer()
        model_name = end_model.split('.', 1)[0]
        while checkpoint_frequency * checkpoint < epochs:
            print(f'Started episode {checkpoint_frequency * checkpoint + 1}.')
            end_model = model_name + f'_{checkpoint_frequency * (checkpoint + 1)}.pl'
            m.create_trainer()
            m.trainer.fit(m)
            m.trainer.save_checkpoint(end_model)
            checkpoint += 1
            print(f'Finished episode {checkpoint_frequency * checkpoint}.\n')

        if checkpoint * checkpoint_frequency < epochs:
            end_model = model_name + f'_{epochs}.pl'
            m.config["train"]["epochs"] = epochs - checkpoint_frequency * checkpoint
            m.create_trainer()
            m.trainer.fit(m)
            m.trainer.save_checkpoint(end_model)
