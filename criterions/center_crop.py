import os
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm  # 引入tqdm库

def parse_args():
    parser = argparse.ArgumentParser(description="Center cropping of LR and HR image pairs.")
    parser.add_argument('--lr_path', type=str, required=True, help='Path to the folder containing LR images.')
    parser.add_argument('--hr_path', type=str, required=True, help='Path to the folder containing HR images.')
    parser.add_argument('--output_dir', '-o', type=str, required=True, help='Output directory to save cropped images.')
    parser.add_argument('--crop_size', type=int, required=True, help='Crop size for LR images.')
    args = parser.parse_args()
    return args

def center_crop(img, crop_size):
    """Center crop the image to the specified crop size."""
    h, w = img.shape[:2]
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    return img[start_y:start_y + crop_size, start_x:start_x + crop_size]

def resize_and_crop(hr_img, lr_img, crop_size, scale_factor=4):
    h_lr, w_lr = lr_img.shape[:2]
    h_hr, w_hr = hr_img.shape[:2]

    # 计算目标 HR 图像大小
    target_h_hr = h_lr * scale_factor
    target_w_hr = w_lr * scale_factor

    # 如果 HR 图像大小与目标大小不匹配，调整大小
    if h_hr != target_h_hr or w_hr != target_w_hr:
        print(f"Resizing HR image from {h_hr}x{w_hr} to {target_h_hr}x{target_w_hr}")
        hr_img = cv2.resize(hr_img, (target_w_hr, target_h_hr), interpolation=cv2.INTER_LINEAR)

    # 执行中心裁剪
    lr_crop = center_crop(lr_img, crop_size)
    hr_crop_size = crop_size * scale_factor
    hr_crop = center_crop(hr_img, hr_crop_size)

    return lr_crop, hr_crop

def save_crops(lr_crop, hr_crop, lr_filename, output_dir):
    """Save the cropped LR and HR images."""
    lr_output_dir = os.path.join(output_dir, 'lr')
    hr_output_dir = os.path.join(output_dir, 'hr')

    os.makedirs(lr_output_dir, exist_ok=True)
    os.makedirs(hr_output_dir, exist_ok=True)

    base_name = Path(lr_filename).stem

    lr_crop_path = os.path.join(lr_output_dir, f'{base_name}_crop.png')
    hr_crop_path = os.path.join(hr_output_dir, f'{base_name}_crop.png')

    cv2.imwrite(lr_crop_path, lr_crop)
    cv2.imwrite(hr_crop_path, hr_crop)

def main():
    args = parse_args()

    lr_files = sorted(os.listdir(args.lr_path))
    hr_files = sorted(os.listdir(args.hr_path))

    # 使用tqdm显示进度条
    with tqdm(total=len(lr_files), desc="Processing images") as pbar:
        for lr_file, hr_file in zip(lr_files, hr_files):
            lr_img_path = os.path.join(args.lr_path, lr_file)
            hr_img_path = os.path.join(args.hr_path, hr_file)

            lr_img = cv2.imread(lr_img_path)
            hr_img = cv2.imread(hr_img_path)

            if lr_img is None or hr_img is None:
                print(f"Error reading {lr_img_path} or {hr_img_path}. Skipping.")
                continue

            lr_crop, hr_crop = resize_and_crop(hr_img, lr_img, args.crop_size)

            # Save the cropped images
            save_crops(lr_crop, hr_crop, lr_file, args.output_dir)

            pbar.update(1)  # 每处理完一个文件，进度条更新

if __name__ == "__main__":
    main()