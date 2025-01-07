import subprocess

def run_inference(input_file, output_file, model_path, save_prediction, red_channel, nir_channel, divide_by, rescale_ndvi=False):
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
    output_file = r"N:\MnD\projects\2024_11_01_object_detection\TreeCrownDelineation-master\output_file_4"
    model_path = r"N:\MnD\projects\2024_11_01_object_detection\TreeCrownDelineation-master\Models\Unet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=1_jitted.pt"
    save_prediction = r"N:\MnD\projects\2024_11_01_object_detection\TreeCrownDelineation-master\ndvi_map"
    red_channel = 0
    nir_channel = 3
    divide_by = 255

    run_inference(
        input_file=input_file,
        output_file=output_file,
        model_path=model_path,
        save_prediction=save_prediction,
        red_channel=red_channel,
        nir_channel=nir_channel,
        divide_by=divide_by,
        rescale_ndvi=True
    )
