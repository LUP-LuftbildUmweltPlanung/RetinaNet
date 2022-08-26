# Parameters
Create_tiles = True
Train = True
Validate = False
Predict = True

# General
patch_size = 400
patch_overlap = 0.3
label_dict = {'label_0': 0, 'label_1': 1, 'label_2': 2}
multi_class = True

# DATA CREATION
annotations = "C:/Path/to/annotations.shp"
image_path = "C:/Path/to/image" or r"C:/Path/to/image/folder"
directory_to_save_crops = r"C:/path/to/save/folder"
split = 0.3

# TRAINING
end_model = "C:/Path/to/model.pl"
annotations_file = "C:/path/to/annotations.csv"
EPOCHS = 120
BATCH_SIZE = 5
CHECKPOINT_FREQUENCY = 1
LEARNING_RATE = 0.0001

# MODEL VALIDATION
annotations_file_vali = "C:/path/to/annotations.csv"
vali_model = "C:/Path/to/model.pl"

# PREDICTION
image_file = "C:/Path/to/image.tif"
output_shape = "C:/Path/to/output.shp"
predict_model = "C:/Path/to/model.pl"

enable_extra_parameters = False
# Extra parameters
seed = None
save = True
score_threshold = 0.05
