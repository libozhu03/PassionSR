# from ptq_quantize_ldm import OSEDiff_ptq
import argparse, os, sys, gc, glob, datetime, yaml
import logging
import time
import numpy as np
from tqdm import trange
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from PIL import Image
import wandb
import torch
import torch.nn as nn
import sys
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import quantization.saw as saw
from quantization.saw.saw_layer import get_loss_function
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from quantization.methods import *
from criterions.methods import *
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.optim.lr_scheduler as lr_scheduler
from datetime import datetime

def apply_quant(model, visual=False):
    quant_config = model.quant_config
    quant_method = quant_config["Unet"]["method"]
    merge = model.merge_lora
    unet_q = model.unet
    vae_q = model.vae
    if quant_method == "saw" or quant_method == "saw_sep":
        cali_data = model.callibaration()
        unet_q, vae_q = quantize_prepare(unet_q, vae_q, quant_config, device=model.device)
        unet_f = model.unet
        vae_f = model.vae
        model.unet = unet_q
        model.vae = vae_q
        model = PTQ(unet_f, vae_f, quant_config, cali_data, merge, device=model.device, whole_model=model)
        model = quantize_save(model, quant_config, merge, device=model.device)
    else:
        raise NotImplementedError(f"Quantization method {quant_method} not implemented")


def quantize_prepare(unet_q, vae_q, quant_config, device):
    quant_method = quant_config["Unet"]["method"]
    if quant_method == "saw" or quant_method == "saw_sep":
        unet_config = quant_config["Unet"]
        quant_config_C = saw.QuantizeModel_config(unet_config)
        unet_q = saw.QuantModel(unet_q, quant_config_C, device=device)
        unet_q.set_quant_state(weight_quant=True, act_quant=True)
        save_model_to_txt(unet_q, os.path.join(quant_config["output_modelpath"], "unet_q.txt"))
        if quant_config["only_Unet"]:
            vae_q = vae_q
        else:
            vae_config = quant_config["Vae"]
            quant_config_C = saw.QuantizeModel_config(vae_config)
            vae_q = saw.QuantModel(vae_q, quant_config_C, device=device)
            vae_q.set_quant_state(weight_quant=True, act_quant=True)
            save_model_to_txt(vae_q, os.path.join(quant_config["output_modelpath"], "vae_q.txt"))
        return unet_q, vae_q
    else:
        raise NotImplementedError(f"Quantization method {quant_method} not implemented")

def PTQ(unet_q, vae_q, quant_config, cali_data, merge, device, whole_model=None):
    quant_method = quant_config["Unet"]["method"]
    print(f"Quantization method: {quant_method}")
    cali_xs, cali_ls, cali_ts, cali_cs, cali_ys, cali_hs, cali_gs = cali_data
    if quant_method == "saw" or quant_method == "saw_sep":
        if quant_config["only_Unet"]:
            # whole_model = saw.saw_cali_U(whole_model, quant_config, cali_data, merge, device)
            if quant_method == "saw_sep":
                whole_model = saw.saw_cali_U_sep(whole_model, quant_config, cali_data, merge, device)
            else:
                whole_model = saw.saw_cali_U(whole_model, quant_config, cali_data, merge, device)
        else:
            if quant_method == "saw_sep":
                whole_model = saw.saw_cali_UV_sep(unet_q, vae_q, whole_model, quant_config, cali_data, merge, device)
            else:
                whole_model = saw.saw_cali_UV(unet_q, vae_q, whole_model, quant_config, cali_data, merge, device)
        return whole_model
    else:
        raise NotImplementedError(f"Quantization method {quant_method} not implemented")

def quantize_save_whole(model, quant_config, merge, device, outpath=None):
    method = quant_config["Unet"]["method"]
    logdir = os.path.join(quant_config["output_modelpath"], "PTQ")
    os.makedirs(logdir, exist_ok=True)
    if quant_config["only_Unet"]:
        if merge:
            torch.save(model.unet, os.path.join(logdir, f"unet_ckpt_merge_{method}_whole.pth"))
            # torch.save(model.vae, os.path.join(logdir, f"vae_ckpt_merge_{method}.pth"))
        else:
            torch.save(model.unet, os.path.join(logdir, f"unet_ckpt_{method}_whole.pth"))
            # torch.save(model.vae, os.path.join(logdir, f"vae_ckpt_{method}.pth"))
    else:
        if merge:
            torch.save(model.unet, os.path.join(logdir, f"unet_ckpt_merge_{method}_whole.pth"))
            torch.save(model.vae, os.path.join(logdir, f"vae_ckpt_merge_{method}_whole.pth"))
        else:
            torch.save(model.unet, os.path.join(logdir, f"unet_ckpt_{method}_whole.pth"))
            torch.save(model.vae, os.path.join(logdir, f"vae_ckpt_{method}_whole.pth"))
    print(f"Model saved to {logdir}")
    return model

def quantize_save(model, quant_config, merge, device, outpath=None):
    method = quant_config["Unet"]["method"]
    logdir = os.path.join(quant_config["output_modelpath"], "PTQ")
    os.makedirs(logdir, exist_ok=True)
    if quant_config["only_Unet"]:
        if merge:
            torch.save(model.unet.state_dict(), os.path.join(logdir, f"unet_ckpt_merge_{method}.pth"))
            # torch.save(model.vae.state_dict(), os.path.join(logdir, f"vae_ckpt_merge_{method}.pth"))
        else:
            torch.save(model.unet.state_dict(), os.path.join(logdir, f"unet_ckpt_{method}.pth"))
            # torch.save(model.vae.state_dict(), os.path.join(logdir, f"vae_ckpt_{method}.pth"))
    else:
        if merge:
            torch.save(model.unet.state_dict(), os.path.join(logdir, f"unet_ckpt_merge_{method}.pth"))
            torch.save(model.vae.state_dict(), os.path.join(logdir, f"vae_ckpt_merge_{method}.pth"))
        else:
            torch.save(model.unet.state_dict(), os.path.join(logdir, f"unet_ckpt_{method}.pth"))
            torch.save(model.vae.state_dict(), os.path.join(logdir, f"vae_ckpt_{method}.pth"))
    print(f"Model saved to {logdir}")
    return model

