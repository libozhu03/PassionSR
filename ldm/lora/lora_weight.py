import argparse
import os.path as osp
import re

import torch
from safetensors.torch import load_file, save_file

# =================#
# UNet Conversion #
# =================#

unet_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("time_embed.0", "time_embedding.linear_1"),
    ("time_embed.2", "time_embedding.linear_2"),
    ("input_blocks.0.0", "conv_in"),
    ("out.0", "conv_norm_out"),
    ("out.2", "conv_out"),
]

unet_conversion_map_resnet = [
    # (stable-diffusion, HF Diffusers)
    ("in_layers.0", "norm1"),
    ("in_layers.2", "conv1"),
    ("out_layers.0", "norm2"),
    ("out_layers.3", "conv2"),
    ("emb_layers.1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),
]

unet_conversion_map_layer = []
# hardcoded number of downblocks and resnets/attentions...
# would need smarter logic for other networks.
for i in range(4):
    # loop over downblocks/upblocks

    for j in range(2):
        # loop over resnets/attentions for downblocks
        hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
        sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
        unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

        if i < 3:
            # no attention layers in down_blocks.3
            hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
            sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
            unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

    for j in range(3):
        # loop over resnets/attentions for upblocks
        hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
        sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
        unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

        if i > 0:
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

    if i < 3:
        # no downsample in down_blocks.3
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
        sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
        unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

        # no upsample in up_blocks.3
        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"output_blocks.{3*i + 2}.{1 if i == 0 else 2}."
        unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{2*j}."
    unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

# def convert_unet_state_dict(unet_state_dict):
#     # buyer beware: this is a *brittle* function,
#     # and correct output requires that all of these pieces interact in
#     # the exact order in which I have arranged them.
#     mapping = {k: k for k in unet_state_dict.keys()}
#     for sd_name, hf_name in unet_conversion_map:
#         mapping[hf_name] = sd_name
#     for k, v in mapping.items():
#         if "resnets" in k:
#             for sd_part, hf_part in unet_conversion_map_resnet:
#                 v = v.replace(hf_part, sd_part)
#             mapping[k] = v
#     for k, v in mapping.items():
#         for sd_part, hf_part in unet_conversion_map_layer:
#             v = v.replace(hf_part, sd_part)
#         mapping[k] = v
#     new_state_dict = {v: unet_state_dict[k] for k, v in mapping.items()}
#     return new_state_dict

def convert_unet_state_dict(unet_state_dict):
    # buyer beware: this is a *brittle* function,
    # and correct output requires that all of these pieces interact in
    # the exact order in which I have arranged them.
    lora_layer = []
    lora_layer.append(("lora_A.default.weight", "lora_A.default_0.weight"))
    lora_layer.append(("lora_A.default.weight", "lora_A.weight"))
    lora_layer.append(("lora_B.default.weight", "lora_B.default_0.weight"))
    lora_layer.append(("lora_B.default.weight", "lora_B.weight"))
    lora_layer.append(("lora_A.default.weight", "lora.down.weight"))
    lora_layer.append(("lora_B.default.weight", "lora.up.weight"))
    
    unet_conversion = unet_conversion_map_layer + unet_conversion_map  + lora_layer
    # unet_conversion = unet_conversion_map_layer
    conversion_map = {hf_key : sd_key  for sd_key, hf_key in unet_conversion} 
    
    model_name = ""
    new_lora_weights = {}
    for key, value in unet_state_dict.items():
        new_key = key
        new_key = new_key.replace("unet.", model_name)
        for  hf_key , sd_key in conversion_map.items():
            if hf_key in new_key:
                new_key = new_key.replace(hf_key, sd_key)  
        new_lora_weights[new_key] = value
    
    unet_conversion = unet_conversion_map_resnet
    conversion_map = {hf_key : sd_key  for sd_key, hf_key in unet_conversion} 
    unet_state_dict = new_lora_weights
    new_lora_weights = {}
    for key, value in unet_state_dict.items():
        new_key = key
        for  hf_key , sd_key in conversion_map.items():
            if hf_key in new_key and not "transformer_blocks" in new_key:
                new_key = new_key.replace(hf_key, sd_key) 
        new_lora_weights[new_key] = value
        
    return new_lora_weights

def convert_unet_target_module(unet_target_module):
    unet_conversion = unet_conversion_map_layer + unet_conversion_map + unet_conversion_map_resnet
    # unet_conversion = unet_conversion_map_layer
    conversion_map = {hf_key : sd_key  for sd_key, hf_key in unet_conversion} 
    
    model_name = ""
    new_target_module = []
    for value in unet_target_module:
        new_value = value
        for  hf_key , sd_key in conversion_map.items():
            if hf_key in new_value:
                new_value = new_value.replace(hf_key, sd_key)        
        new_target_module.append(new_value)
    
    return new_target_module
    
# ================#
# VAE Conversion #
# ================#

vae_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("nin_shortcut", "conv_shortcut"),
    ("norm_out", "conv_norm_out"),
    ("mid.attn_1.", "mid_block.attentions.0."),
]

for i in range(4):
    # down_blocks have two resnets
    for j in range(2):
        hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
        sd_down_prefix = f"encoder.down.{i}.block.{j}."
        vae_conversion_map.append((sd_down_prefix, hf_down_prefix))

    if i < 3:
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
        sd_downsample_prefix = f"down.{i}.downsample."
        vae_conversion_map.append((sd_downsample_prefix, hf_downsample_prefix))

        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"up.{3-i}.upsample."
        vae_conversion_map.append((sd_upsample_prefix, hf_upsample_prefix))

    # up_blocks have three resnets
    # also, up blocks in hf are numbered in reverse from sd
    for j in range(3):
        hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
        sd_up_prefix = f"decoder.up.{3-i}.block.{j}."
        vae_conversion_map.append((sd_up_prefix, hf_up_prefix))

# this part accounts for mid blocks in both the encoder and the decoder
for i in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{i}."
    sd_mid_res_prefix = f"mid.block_{i+1}."
    vae_conversion_map.append((sd_mid_res_prefix, hf_mid_res_prefix))


vae_conversion_map_attn = [
    # (stable-diffusion, HF Diffusers)
    ("norm.", "group_norm."),
    ("q.", "query."),
    ("k.", "key."),
    ("v.", "value."),
    ("proj_out.", "proj_attn."),
]

# This is probably not the most ideal solution, but it does work.
vae_extra_conversion_map = [
    ("to_q", "q"),
    ("to_k", "k"),
    ("to_v", "v"),
    ("to_out.0", "proj_out"),
]

# vae_extra_conversion_map.append(("lora_A.default.weight", "lora_A.default_0.weight"))
# vae_extra_conversion_map.append(("lora_A.default.weight", "lora_A.weight"))
# vae_extra_conversion_map.append(("lora_B.default.weight", "lora_B.default_0.weight"))
# vae_extra_conversion_map.append(("lora_B.default.weight", "lora_B.weight"))
# vae_extra_conversion_map.append(("lora_A.default.weight", "lora.down.weight"))
# vae_extra_conversion_map.append(("lora_B.default.weight", "lora.up.weight"))

def reshape_weight_for_sd(w):
    # convert HF linear weights to SD conv2d weights
    if not w.ndim == 1:
        return w.reshape(*w.shape, 1, 1)
    else:
        return w


def convert_vae_state_dict(vae_state_dict):
    mapping = {k: k for k in vae_state_dict.keys()}
    for k, v in mapping.items():
        for sd_part, hf_part in vae_conversion_map:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    for k, v in mapping.items():
        if "attentions" in k:
            for sd_part, hf_part in vae_conversion_map_attn:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    # for k, v in mapping.items():
    #     for hf_part, sd_part in vae_extra_conversion_map:
    #         v = v.replace(hf_part, sd_part)
    #     mapping[k] = v
    new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
    weights_to_convert = ["q", "k", "v", "proj_out"]
    keys_to_rename = {}
    for k, v in new_state_dict.items():
        for weight_name in weights_to_convert:
            # if f"mid.attn_1.{weight_name}.weight" in k:
            if f"mid.attn_1.{weight_name}." in k:
                print(f"Reshaping {k} for SD format")
                new_state_dict[k] = reshape_weight_for_sd(v)
        for weight_name, real_weight_name in vae_extra_conversion_map:
            # if f"mid.attn_1.{weight_name}.weight" in k or f"mid.attn_1.{weight_name}.bias" in k:
            if f"mid.attn_1.{weight_name}." in k or f"mid.attn_1.{weight_name}." in k:
                keys_to_rename[k] = k.replace(weight_name, real_weight_name)
    for k, v in keys_to_rename.items():
        if k in new_state_dict:
            # print(f"Renaming {k} to {v}")
            new_state_dict[v] = reshape_weight_for_sd(new_state_dict[k])
            del new_state_dict[k]
    return new_state_dict

def convert_vae_target_module(vae_target_module):
    mapping = {k: k for k in vae_target_module}
    for k, v in mapping.items():
        for sd_part, hf_part in vae_conversion_map:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    for k, v in mapping.items():
        if "attentions" in k:
            for sd_part, hf_part in vae_conversion_map_attn:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for  hf_part , sd_part in vae_extra_conversion_map:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    new_target_module = [v for k, v in mapping.items()]
    return new_target_module