import os, sys
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
from criterions.methods import *
from criterions.FID import calculate_fid
import argparse
import json
import time
from tqdm import tqdm
from torchvision import models, transforms
from pytorch_fid import fid_score

def measure_image_quality(image_folder, reference_folder, device="cuda"):
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    dists_scores = []
    fid_scores = []
    niqe_scores = []
    musiq_scores = []
    maniqa_scores = []
    clip_iqa_scores = []
    results_dict = {}

    image_list = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    for filename in tqdm(image_list, desc="Processing", unit="picture", colour="green"):
        time.sleep(0.05)
        image_path = os.path.join(image_folder, filename)
        # reference_image_name = find_matching_images(filename, reference_folder)[0]
        reference_image_name = find_most_similar_file(reference_folder, filename)
        reference_image_path = os.path.join(reference_folder, reference_image_name)

        output_image = Image.open(image_path)
        target_image = Image.open(reference_image_path)

        values_Dict = calculate_iqa_for_partition(output_image, target_image, device)

        psnr_value = values_Dict['PSNR_Y']
        ssim_value = values_Dict['SSIM_Y']
        lpips_value = values_Dict['LPIPS']
        dists_value = values_Dict['DISTS']
        niqe_value = values_Dict['NIQE']
        musiq_value = values_Dict['MUSIQ']
        maniqa_value = values_Dict['MANIQA']
        clip_iqa_value = values_Dict['CLIP-IQA']

        psnr_scores.append(psnr_value)
        ssim_scores.append(ssim_value)
        lpips_scores.append(lpips_value)
        dists_scores.append(dists_value)
        # fid_scores.append(fid_value)
        niqe_scores.append(niqe_value)
        musiq_scores.append(musiq_value)
        maniqa_scores.append(maniqa_value)
        clip_iqa_scores.append(clip_iqa_value)

        # print(fid_value)
        # print(clip_iqa_value)

        results_dict[filename] = {"PSNR": psnr_value, "SSIM": ssim_value, "LPIPS": lpips_value,
                                  "DISTS": dists_value, "NIQE": niqe_value,
                                  "MUSIQ": musiq_value, "MANIQA": maniqa_value, "CLIP-IQA": clip_iqa_value}
        print(f"Image: {filename} | Reference: {reference_image_name} |PSNR: {psnr_value} | SSIM: {ssim_value} | LPIPS: {lpips_value}")
        print(f"DISTS: {dists_value} | NIQE: {niqe_value} | MUSIQ: {musiq_value} | MANIQA: {maniqa_value} | CLIP-IQA: {clip_iqa_value}")

    return psnr_scores, ssim_scores, lpips_scores, dists_scores, niqe_scores, musiq_scores, maniqa_scores, clip_iqa_scores, results_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default='results/test_dataset')
    parser.add_argument('--reference_image', '-r', type=str, default='preset/datasets/test_dataset/reference')
    parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--output', '-o', type=str, default='results/test_dataset/measure')
    args = parser.parse_args()
    args.output = os.path.join(args.input_image, 'measure')

    command = "python " + " ".join(sys.argv)
    print(command)

    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception.to(args.device)
    inception.eval()
    # fid_score = calculate_fid(args.input_image, args.reference_image, inception,  args.device)
    results = measure_image_quality(args.input_image, args.reference_image, args.device)
    print("PSNR scores:", sum(results[0])/len(results[0]))
    print("SSIM scores:", sum(results[1])/len(results[1]))
    print("LPIPS scores:", sum(results[2])/len(results[2]))
    print("DISTS scores:", sum(results[3])/len(results[3]))
    # print("FID scores:", fid_score)
    print("NIQE scores:", sum(results[4])/len(results[4]))
    print("MUSIQ scores:", sum(results[5])/len(results[5]))
    print("MANIQA scores:", sum(results[6])/len(results[6]))
    print("CLIP-IQA scores:", calculate_mean_without_non_numeric(results[7]))
    dict = results[-1]

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with open(os.path.join(args.output, 'score_each.json'), 'w') as f:
        json.dump(dict, f, indent=4)

    with open(os.path.join(args.output, 'score_average.json'), 'w') as f:
        json.dump({
            "PSNR": sum(results[0])/len(results[0]),
            "SSIM": sum(results[1])/len(results[1]),
            "LPIPS": sum(results[2])/len(results[2]),
            "DISTS": sum(results[3])/len(results[3]),
            # "FID": fid_score,
            "NIQE": sum(results[4])/len(results[4]),
            "MUSIQ": sum(results[5])/len(results[5]),
            "MANIQA": sum(results[6])/len(results[6]),
            "CLIP-IQA": calculate_mean_without_non_numeric(results[7])
            },
            f, indent=4)



