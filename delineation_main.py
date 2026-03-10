# import library
import subprocess

# Please define the Parameters:
input_file = r"Path\to\tif.tif" #  the image which we want to predict
output_file = r"Path\to\output\TreeCrown" # the output prediction folder (the last part the name of the file)
model_path = r"Models\Unet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=2_jitted.pt" # path to the model
save_prediction = r"Path\to\output\TreeCrown_ndvi" # the name of ndvi file and whole path (the last part the name of the file)


# function to run inference file
def run_inference(input_file, output_file, model_path, save_prediction, red_channel, nir_channel, divide_by,
                  rescale_ndvi=False, additional_args=None):
    """
    Run the inference.py script with the provided arguments.

    :param input_file: Path to the input raster file
    :param output_file: Path to save the output file
    :param model_path: Path to the model file
    :param save_prediction: Path to save intermediate predictions
    :param red_channel: Red channel index (integer)
    :param nir_channel: Near-infrared channel index (integer)
    :param divide_by: Value to divide the input data
    :param rescale_ndvi: Flag to rescale NDVI values to 0...1
    :param additional_args: List of additional arguments to pass to the script
    """
    # Base command
    command = [
        "python", "TreeCrownDelineation/scripts/inference.py",
        "-i", input_file,
        "-o", output_file,
        "-m", model_path,
        "--ndvi",
        "--red", str(red_channel),
        "--nir", str(nir_channel),
        "--divide-by", str(divide_by),
        "--save-prediction", save_prediction
    ]

    # Add the rescale NDVI flag if specified
    if rescale_ndvi:
        command.append("--rescale-ndvi")

    # Add any additional arguments
    if additional_args:
        command.extend(additional_args)

    # Run the subprocess
    try:
        print("Running inference script...")
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Inference completed successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error occurred while running inference script.")
        print(e.stderr)


if __name__ == "__main__":
    # Variables
    input_file = tile_folder
    output_file = output_folder
    model_path = model_path
    save_prediction = save_prediction
    red_channel = 0
    nir_channel = 3
    divide_by = 255

    additional_args = [
        "--div", "255",
        "--ndvi",
        "--sigmoid",
        "-a",  # Test-Time Augmentation
        "-w", "512",
        "--simplify", "0.1",

        "--min-dist", "10",  # allow closer tree peaks
        "--label-threshold", "0.001",  # detect weaker crowns
        "--binary-threshold", "0.01",

        "--sigma", "1",  # less smoothing -> small trees
        "--upsample", "1.5"  # improve small crown detection
    ]

    # run the function
    run_inference(
        input_file=input_file,
        output_file=output_file,
        model_path=model_path,
        save_prediction=save_prediction,
        red_channel=red_channel,
        nir_channel=nir_channel,
        divide_by=divide_by,
        rescale_ndvi=True,
        additional_args=additional_args
    )
