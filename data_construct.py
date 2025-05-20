import os, yaml
import random
import argparse
from PIL import Image
from preset.data_construct import data_path
from quantization.methods import *
from criterions.methods import find_most_similar_file
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Randomly select and crop images at the same position")
    parser.add_argument("--config_file", type=str, default="config.yaml", help="Path to the configuration file")
    return parser.parse_args()

def load_images_from_directory(directory):
    """Load all image file paths from the specified directory"""
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(supported_formats)]

def random_crop(image, crop_size, crop_pos=None):
    """
    Perform random or specified cropping on an image
    :param image: Input image
    :param crop_size: Size of the crop (square)
    :param crop_pos: Crop position (x, y); if None, a random position will be generated
    :return: Cropped image and crop position
    """
    width, height = image.size

    # If no crop position is specified, generate a random one
    if crop_pos is None:
        x = random.randint(0, width - crop_size)
        y = random.randint(0, height - crop_size)
        crop_pos = (x, y)

    x, y = crop_pos
    cropped_image = image.crop((x, y, x + crop_size, y + crop_size))

    return cropped_image, crop_pos

def process_and_save_images(lr_image_path, hr_image_path, crop_size, upscale, output_dir, index):
    """
    Crop paired LR and HR images at the same position and save them
    :param lr_image_path: Path to the LR image
    :param hr_image_path: Path to the HR image
    :param crop_size: Crop size for the LR image
    :param upscale: Upscale factor for HR image
    :param output_dir: Output directory
    :param index: Index for saving the output images
    """
    lr_image = Image.open(lr_image_path).convert('RGB')
    hr_image = Image.open(hr_image_path).convert('RGB')

    # Calculate HR crop size
    hr_crop_size = crop_size * upscale

    # Randomly crop the LR image and get the crop position
    lr_cropped, crop_pos = random_crop(lr_image, crop_size)

    # Calculate the corresponding crop position for the HR image
    hr_crop_pos = (crop_pos[0] * upscale, crop_pos[1] * upscale)

    # Crop the HR image at the corresponding position
    hr_cropped, _ = random_crop(hr_image, hr_crop_size, hr_crop_pos)

    # Save the cropped LR and HR images
    lr_output_path = os.path.join(output_dir, "lr")
    hr_output_path = os.path.join(output_dir, "hr")
    os.makedirs(lr_output_path, exist_ok=True)
    os.makedirs(hr_output_path, exist_ok=True)
    lr_cropped.save(os.path.join(lr_output_path, f"lr_{index}.png"))
    hr_cropped.save(os.path.join(hr_output_path, f"hr_{index}.png"))

def prepare_data(opt: dict):
    opt["dataset_lr_list"] = [data_path[dataset]["lr"] for dataset in opt["dataset"]]
    opt["dataset_hr_list"] = [data_path[dataset]["hr"] for dataset in opt["dataset"]]
    return opt

def main():
    # Parse command line arguments
    args = parse_args()

    with open(args.config_file, 'r') as file:
        opt = yaml.safe_load(file)
    print(yaml.dump(opt, default_flow_style=False))

    opt = prepare_data(opt)

    output_dir = opt["output_dir"]
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    lr_images = []
    hr_images = []
    # Iterate through all pairs of input directories
    for lr_dir, hr_dir in zip(opt["dataset_lr_list"], opt["dataset_hr_list"]):
        lr_images_list = load_images_from_directory(lr_dir)
        hr_images_list = [os.path.join(hr_dir, find_most_similar_file(hr_dir, os.path.basename(lr_image))) for lr_image in lr_images_list]
        # hr_images = load_images_from_directory(hr_dir)
        # print(lr_images)
        # print(hr_images)
        lr_images.extend(lr_images_list)
        hr_images.extend(hr_images_list)

    # Ensure the number of images in LR and HR directories is the same
    if len(lr_images) != len(hr_images):
        print(len(lr_images), len(hr_images))
        raise ValueError(f"Number of images in {lr_dir} and {hr_dir} do not match!")

    # Randomly select the specified number of samples
    selected_indices = random.sample(range(len(lr_images)), opt["num_samples"])
    selected_lr_images = [lr_images[i] for i in selected_indices]
    selected_hr_images = [hr_images[i] for i in selected_indices]

    # Process and save the images
    # for index, (lr_image, hr_image) in tqdm(enumerate(zip(selected_lr_images, selected_hr_images)), desc="Processing", unit="picture", colour="green"):
    #     process_and_save_images(lr_image, hr_image, opt["crop_size"], opt["upscale"], output_dir, index)

    with tqdm(total=len(selected_lr_images), desc="Processing", unit="picture", colour="green") as pbar:
            for index, (lr_image, hr_image) in enumerate(zip(selected_lr_images, selected_hr_images)):
                process_and_save_images(lr_image, hr_image, opt["crop_size"], opt["upscale"], output_dir, index)
                pbar.update(1)  # Update the progress bar

    print(f"Successfully cropped and saved images to {output_dir}!")

if __name__ == "__main__":
    main()
