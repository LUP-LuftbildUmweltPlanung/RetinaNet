import subprocess


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
    # Example usage
    input_file = r"N:\MnD\projects\2024_11_01_object_detection\TreeCrownDelineation-master\dop1.tif"
    output_file = r"N:\MnD\projects\2024_11_01_object_detection\TreeCrownDelineation-master\output_file_last_script"
    model_path = r"N:\MnD\projects\2024_11_01_object_detection\TreeCrownDelineation-master\Models\Unet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=2_jitted.pt"
    save_prediction = r"N:\MnD\projects\2024_11_01_object_detection\TreeCrownDelineation-master\ndvi_map_last_script"
    red_channel = 0
    nir_channel = 3
    divide_by = 255

    additional_args = [
        "--div", "255",         # Specify a division factor for input values (e.g., divide by 255 to normalize pixel values).
        "--ndvi",               # Flag to calculate NDVI (Normalized Difference Vegetation Index) during processing.
        "--sigmoid",            # Apply a sigmoid function to the predictions, typically used for scaling output probabilities.
        "-a",                   # Enable additional processing or features (short flag for a specific script feature).
        "-w", "512",            # Specify the output width for resampling or processing (e.g., 512 pixels wide).
        "--simplify", "0.1"     # Simplify geometries or results with a specified tolerance (e.g., 0.1 for simplification).
    ]


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
