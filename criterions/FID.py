import torch
from torchvision import models, transforms
from scipy import linalg
import numpy as np
from PIL import Image
from pytorch_fid import fid_score

transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 图像预处理函数
def preprocess_image(image, image_size=299):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # 增加 batch 维度

# 中心裁剪函数，将第二张图按照第一张图的大小裁剪
def center_crop(image, target_width, target_height):
    width, height = image.size  # 获取图像的宽度和高度
    left = (width - target_width) / 2
    top = (height - target_height) / 2
    right = (width + target_width) / 2
    bottom = (height + target_height) / 2
    return image.crop((left, top, right, bottom))  # 裁剪图像

# 提取图像特征的函数
def get_inception_features(images, inception):
    features = []
    for image in images:
        with torch.no_grad():
            feature = inception(image).detach().numpy()
            features.append(feature.flatten())
    return np.array(features)

# 计算 Fréchet 距离
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    # # 确保协方差矩阵是二维的
    covmean = np.sqrt(sigma1 * sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real   
    return (mu1 - mu2)**2 + (sigma1 + sigma2 - 2 * covmean)

# 计算 FID 的函数
def calculate_fid(image1_path, image2_path, inception=None, device="cuda"):
    # # 加载图像1和图像2
    # image1 = Image.open(image1_path).convert('RGB')
    # image2 = Image.open(image2_path).convert('RGB')

    # # 获取 image1 的尺寸并中心裁剪 image2
    # target_width, target_height = image1.size
    # cropped_image2 = center_crop(image2, target_width, target_height)

    # # 预处理图像以供 InceptionV3 提取特征
    # image1_tensor = preprocess_image(image1)
    # cropped_image2_tensor = preprocess_image(cropped_image2)

    # # 提取图像特征
    # real_features = get_inception_features([image1_tensor], inception)
    # gen_features = get_inception_features([cropped_image2_tensor], inception)

    # # 计算均值和协方差
    # mu_real = np.mean(real_features, axis=0)
    # sigma_real = np.cov(real_features, rowvar=False)

    # mu_gen = np.mean(gen_features, axis=0)
    # sigma_gen = np.cov(gen_features, rowvar=False)

    # # 计算 FID 分数
    # fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

    fid_value = fid_score.calculate_fid_given_paths([image1_path, image2_path], inception, device=device, dims=2048)

    return fid_value