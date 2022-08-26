import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

from deepforest import main
from deepforest import get_data
from deepforest import evaluate

import shapely
shapely.speedups.disable()


def validate(annotations_file, pre_model=None, multi=False, label_dict=None, labels_order=None, save=True, gpu=True,
             score=0.1):
    """
    Validates a model on data provided in the annotations file.

                Keyword arguments
                -----------------
                annotations_file:   Annotations file to be used during validation
                pre_model:      Model to be validated
                multi:              If the annotations file contains multiple classes (default=False)
                label_dict:         Label dictionary as created by the model during training
                                    (default=None, only required when training with multiple classes)
                save:               If prediction images during validation should be stored (default=False)
                gpu:                If GPU should be used for prediction (default=True)
    """
    # reload the checkpoint to model object
    if pre_model is not None:
        if multi:
            assert label_dict is not None, "Please set a label_dict when using multi-class models"
            m = main.deepforest(num_classes=len(label_dict), label_dict=label_dict)
            if gpu == 1:
                m.to("cuda")
            else:
                m.to("cpu")
            ckpt = torch.load(pre_model, map_location=torch.device("cuda" if gpu else "cpu"))
            m.load_state_dict(ckpt["state_dict"])
        else:
            m = main.deepforest()
            m.load_from_checkpoint(pre_model)

        directory = Path(str(pre_model).rsplit('\\', 1)[0] + r"\validation\\" +
                         str(pre_model).rsplit('\\', 1)[1].rsplit('.', 1)[0])
    else:
        m = main.deepforest()
        if gpu == 1:
            m.to("cuda")
        else:
            m.to("cpu")
        print("using standard neon model...")
        m.use_release()
        directory = Path(str(annotations_file).rsplit('\\', 1)[0] + r'\standard_validation')

    m.config["score_thresh"] = score
    print("predicting with score_thresh: "+str(m.config["score_thresh"]))
    csv_file = get_data(annotations_file)
    predictions = m.predict_file(csv_file=csv_file, root_dir=os.path.dirname(csv_file))

    ground_truth = pd.read_csv(csv_file)

    pred_num = predictions.copy()
    ground_num = ground_truth.copy()

    directory.mkdir(parents=True, exist_ok=True)

    if save:
        result = evaluate.evaluate(predictions=pred_num, ground_df=ground_num, root_dir=os.path.dirname(csv_file),
                                   savedir=directory)

    else:
        result = evaluate.evaluate(predictions=pred_num, ground_df=ground_num, root_dir=os.path.dirname(csv_file),
                                   savedir=None)

    result["results"].to_csv(directory / 'all_predictions.csv', index=False)
    result["class_recall"].to_csv(directory / 'class_recall.csv', index=False)

    y_true = result["results"]["true_label"].copy()
    y_pred = result["results"]["predicted_label"].copy()
    # y_pred[y_pred != y_pred] = 'background'

    y_p = y_pred[y_pred == y_pred]
    y_t = y_true[y_pred == y_pred]

    if label_dict is None:
        labels = ['Tree']
    elif labels_order is None:
        labels = list(label_dict.keys())
    else:
        labels = labels_order

    cm = confusion_matrix(y_t, y_p, labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(directory / 'confusion_matrix.png', dpi=200)

    with open(directory / "classification_report.txt", 'w') as f:
        f.write(classification_report(y_t, y_p, labels=labels))

    dict_items = result.items()
    df_summary = pd.DataFrame(list(dict_items)[1:3])

    df_summary.to_csv(directory / "general_box_accuracy.csv", index=False, header=False)

    return precision_recall_fscore_support(y_t, y_p, labels=labels, average='weighted')


def visualize_validation(data, path):
    metrics = data.columns.values.tolist()
    episode_stepsize = data['Episode'][0]
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    max_value = 1.1
    for m in metrics:
        if m != 'Episode' and m != 'Support':
            ax.plot(data['Episode'], data[m].tolist(), label=m)
            max_value = np.max([np.max(data[m].tolist()), max_value])
        if m == 'F1_Score':
            annot_max(data[m].tolist(), episode_stepsize)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Metric')
    ax.set_ylim(0, max_value)
    ax.set_title('Model Validation Overview')
    ax.legend()
    fig.savefig(path, dpi=200)


def annot_max(y, episode_stepsize, ax=None):
    xmax = np.argmax(y) * episode_stepsize
    ymax = np.max(y)
    text = f"Highest F1-Score={ymax:.2f}, Ep. {xmax}"
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=120")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="left", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.06, 0.96), **kw)
