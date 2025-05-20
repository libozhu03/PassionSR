import os
import cv2
import torch
import lpips
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pytorch_fid import fid_score
from torchvision import transforms
from PIL import Image
import re
import pyiqa
import torchvision.transforms.functional as F
import yaml

import difflib

def longest_common_prefix(s1, s2):
    match_len = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            match_len += 1
        else:
            break
    return match_len

def find_most_similar_file(target_dir, input_filename):
    files = os.listdir(target_dir)

    if not files:
        return None

    max_match_len = 0
    best_match_file = None

    for file in files:
        match_len = longest_common_prefix(input_filename, file)
        if match_len > max_match_len:
            max_match_len = match_len
            best_match_file = file

    return best_match_file

def calculate_mean_without_non_numeric(data):
    numeric_data = [x for x in data if isinstance(x, (int, float))]

    if len(numeric_data) == 0:
        return None

    mean_value = np.mean(numeric_data)
    
    return mean_value

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config 

def find_matching_images(input_image_name, target_folder):

    base_pattern = '_'.join(input_image_name.split('_')[:-1])

    regex_pattern = re.compile(r'^' + re.escape(base_pattern) + r'(_[^_]+)?\..+$')

    matching_images = []

    for filename in os.listdir(target_folder):
        if regex_pattern.match(filename):
            matching_images.append(filename)
    
    return matching_images

def rgb_to_ycrcb(tensor):
    tensor_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ycrcb_np = cv2.cvtColor(tensor_np, cv2.COLOR_RGB2YCrCb)
    ycrcb_tensor = torch.tensor(ycrcb_np).permute(2, 0, 1).unsqueeze(0).float()
    return ycrcb_tensor

class IQA:
    def __init__(self, device=None):
        self.device = device if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.iqa_metrics = {
            'psnr': pyiqa.create_metric('psnr', device=self.device), 
            'ssim': pyiqa.create_metric('ssim', device=self.device),
            'psnr_y': pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device), 
            'ssim_y': pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device), 
            'lpips': pyiqa.create_metric('lpips', device=self.device),
            'dists': pyiqa.create_metric('dists', device=self.device),
            'niqe': pyiqa.create_metric('niqe', device=self.device),
            'musiq': pyiqa.create_metric('musiq', device=self.device),
            'maniqa': pyiqa.create_metric('maniqa', device=self.device),
            'clipiqa': pyiqa.create_metric('clipiqa', device=self.device)
        }
    
    def calculate_values(self, output_image, target_image): 
        output_tensor = F.to_tensor(output_image).unsqueeze(0).to(self.device)
        target_tensor = F.to_tensor(target_image).unsqueeze(0).to(self.device)

        if output_tensor.shape != target_tensor.shape:
            print(f"[IQA Reshape] predicted shape: {output_tensor.shape}, target shape: {target_tensor.shape}")
            min_height = min(output_tensor.shape[2], target_tensor.shape[2])
            min_width = min(output_tensor.shape[3], target_tensor.shape[3])
            resize_transform = transforms.Resize((min_height, min_width))
            output_tensor = resize_transform(output_tensor)
            target_tensor = resize_transform(target_tensor)
        
        # img1_ycrcb = rgb_to_ycrcb(output_tensor)
        # img2_ycrcb = rgb_to_ycrcb(target_tensor)
        
        # # 只取 Y 通道
        # img1_y = img1_ycrcb[:, 0:1, :, :].to(self.device)
        # img2_y = img2_ycrcb[:, 0:1, :, :].to(self.device)

        psnr_value = self.iqa_metrics['psnr'](output_tensor, target_tensor)
        ssim_value = self.iqa_metrics['ssim'](output_tensor, target_tensor)

        psnr_value_y = self.iqa_metrics['psnr_y'](output_tensor, target_tensor)
        ssim_value_y = self.iqa_metrics['ssim_y'](output_tensor, target_tensor)

        lpips_value = self.iqa_metrics['lpips'](output_tensor, target_tensor)
        dists_value = self.iqa_metrics['dists'](output_tensor, target_tensor)

        niqe_value = self.iqa_metrics['niqe'](output_tensor)
        musiq_value = self.iqa_metrics['musiq'](output_tensor)
        maniqa_value = self.iqa_metrics['maniqa'](output_tensor)
        clipiqa_value = self.iqa_metrics['clipiqa'](output_tensor)

        return {
            'PSNR': psnr_value.item(),
            'SSIM': ssim_value.item(), 
            'PSNR_Y': psnr_value_y.item(),
            'SSIM_Y': ssim_value_y.item(),
            'LPIPS': lpips_value.item(), 
            'DISTS': dists_value.item(),
            'NIQE': niqe_value.item(),
            'MUSIQ': musiq_value.item(),
            'MANIQA': maniqa_value.item(),
            'CLIP-IQA': clipiqa_value.item()
        }


def calculate_iqa_for_partition(output_image, target_image, device):
    iqa = IQA(device=device)

    # 计算 IQA 值
    values = iqa.calculate_values(output_image, target_image)
    
    return values



# # PSNR
# def calculate_psnr(image_path, reference_image_path):
#     image = cv2.imread(image_path)
#     reference_image = cv2.imread(reference_image_path)

#     height, width = image.shape[:2]
#     reference_image = center_crop(reference_image, width, height)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)    
#     return psnr(image, reference_image)

# # SSIM 
# def calculate_ssim(image_path, reference_image_path):
#     image = cv2.imread(image_path)
#     reference_image = cv2.imread(reference_image_path)

#     height, width = image.shape[:2]
#     reference_image = center_crop(reference_image, width, height)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

#     height, width = image.shape[:2]
#     win_size = min(height, width, 7) if min(height, width) >= 7 else min(height, width) - (min(height, width) % 2 - 1)
#     if image.ndim == 3:
#         return ssim(image, reference_image, win_size=win_size, channel_axis=2)
#     else:
#         return ssim(image, reference_image, win_size=win_size)
    
# # Lpips
# def calculate_lpips(image_path, reference_image_path, model):
#     image = cv2.imread(image_path)
#     reference_image = cv2.imread(reference_image_path)

#     height, width = image.shape[:2]
#     reference_image = center_crop(reference_image, width, height)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

#     image1_tensor = preprocess_image(image)
#     image2_tensor = preprocess_image(reference_image)
#     return model(image1_tensor, image2_tensor).item()

# def resize_image(image, target_size):
#     return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

# def preprocess_image(image, device='cuda'):
#     # Convert the image to tensor and normalize
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])
#     return transform(image).unsqueeze(0).to(device)



# def center_crop(image, target_width, target_height):
#     height, width = image.shape[:2]
    
#     center_x, center_y = width // 2, height // 2
    
#     x1 = center_x - (target_width // 2)
#     y1 = center_y - (target_height // 2)
#     x2 = center_x + (target_width // 2)
#     y2 = center_y + (target_height // 2)

#     cropped_image = image[y1:y2, x1:x2, :]
    
#     return cropped_image


# def load_image(image_path):
#     image = Image.open(image_path).convert('RGB')
#     return image

# def center_crop_2(image, target_width, target_height):
#     width, height = image.size  # 获取 image2 的宽度和高度

#     left = (width - target_width) / 2
#     top = (height - target_height) / 2
#     right = (width + target_width) / 2
#     bottom = (height + target_height) / 2

#     cropped_image = image.crop((left, top, right, bottom))
#     return cropped_image

# # Dists
# def calculate_dists(image1_path, image2_path, dists_metric):
#     image1 = load_image(image1_path)
#     image2 = load_image(image2_path)
    
#     target_width, target_height = image1.size
    
#     cropped_image2 = center_crop_2(image2, target_width, target_height)
    
#     transform = transforms.Compose([
#         transforms.ToTensor(),  
#     ])
    
#     image1_tensor = transform(image1).unsqueeze(0).cuda() 
#     image2_tensor = transform(cropped_image2).unsqueeze(0).cuda()  
    
#     similarity_score = dists_metric(image1_tensor, image2_tensor)
    
#     return similarity_score.item()

