from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import load_file
from torch import nn
from ldm.lora.lora_weight import *
import torch

train_args = {
    "lora_rank":8,
    "lora_alpha":8,
    "lora_dropout":0.0
}

# lora_target_modules = [
#         "to_q", "to_k", "to_v", "to_out.0",
#         "proj_in", "proj_out",
#         "ff.net.0.proj", "ff.net.2",
#         "conv1", "conv2", "conv_shortcut",
#         "downsamplers.0.conv", "upsamplers.0.conv",
#         "time_emb_proj",
# ]

lora_target_modules = [
        "to_q", "to_k", "to_v", "to_out.0",
        "proj_in", "proj_out",
        "ff.net.0.proj", "ff.net.2",
        "in_layers.2", "out_layers.3",
        "0.op", "output_blocks.2.1.conv", "output_blocks.5.2.conv", "output_blocks.8.2.conv",
        "emb_layers.1",
]

# unet_conversion_map_resnet = [
#     # (stable-diffusion, HF Diffusers)
#     ("in_layers.2", "conv1"),
#     ("out_layers.3", "conv2"),
#     ("emb_layers.1", "time_emb_proj"),
# ]
# conversion_map = {hf_key: sd_key for sd_key, hf_key in unet_conversion_map_resnet}


def find_skip_connection_modules(model, target_modules):
    found_modules = []

    for name, module in model.named_modules():
        if "skip_connection" in name:
            if not isinstance(module, nn.Identity):
                found_modules.append(name)

    target_modules.extend(found_modules)
    return target_modules

def load_lora(model, lora_weights_path):
    lora_target = lora_target_modules + find_skip_connection_modules(model, lora_target_modules)

    lora_config = LoraConfig(
        r=train_args["lora_rank"],
        target_modules=lora_target,
        lora_alpha=train_args["lora_alpha"],
        lora_dropout=train_args["lora_dropout"]
    )
    # model_state_dict = model.state_dict()
    model = get_peft_model(model, lora_config, adapter_name="default").base_model.model
    
    # lora_weights = load_file(lora_weights_path)
    lora_weights = torch.load(lora_weights_path)
    print("-----------------")
    lora_weights = convert_unet_state_dict(lora_weights["unet_state_dict"])
    state_dict_info = model.load_state_dict(lora_weights, strict=False)
    print(state_dict_info)
        
    if len(state_dict_info.unexpected_keys) == 0:
        print("DMDSR Unet is loaded successfully!")
    else:
        print("Unexpected keys:", state_dict_info.unexpected_keys)
    
    return model

def load_lora_unet_OSE(model, lora_weights_path, merge_lora=False):
    model_ckpt = torch.load(lora_weights_path)
    rank_unet = model_ckpt["rank_unet"]
    unet_lora_encoder_modules = convert_unet_target_module(model_ckpt["unet_lora_encoder_modules"])
    unet_lora_decoder_modules = convert_unet_target_module(model_ckpt["unet_lora_decoder_modules"])
    unet_lora_others_modules = convert_unet_target_module(model_ckpt["unet_lora_others_modules"])
    
    unet_lora_encoder_modules.extend(["input_blocks.3.0.op", "input_blocks.6.0.op", "input_blocks.9.0.op"])
    # i don't understand why missing three blocks

    # load unet lora
    lora_conf_encoder = LoraConfig(r=rank_unet, init_lora_weights="gaussian", target_modules=unet_lora_encoder_modules)
    lora_conf_decoder = LoraConfig(r=rank_unet, init_lora_weights="gaussian", target_modules=unet_lora_decoder_modules)
    lora_conf_others = LoraConfig(r=rank_unet, init_lora_weights="gaussian", target_modules=unet_lora_others_modules)
    
    model = get_peft_model(model, lora_conf_encoder, adapter_name="default_encoder").base_model.model
    model = get_peft_model(model, lora_conf_decoder, adapter_name="default_decoder").base_model.model
    model = get_peft_model(model, lora_conf_others, adapter_name="default_others").base_model.model

    unet_state_dict = model_ckpt["unet_state_dict"]
    unet_state_dict = convert_unet_state_dict(unet_state_dict)
    
    # lora_state_dict = {}
    # for key, value in unet_state_dict.items():
    #     if "lora" in key:
    #         lora_state_dict[key] = value
    lora_state_dict = unet_state_dict
    
    state_dict_info = model.load_state_dict(lora_state_dict, strict=False)
    
    if len(state_dict_info.unexpected_keys) == 0:
        print("OSEDiff Unet is loaded successfully!")
    else:
        print("Unexpected keys:", state_dict_info.unexpected_keys)
        
    if merge_lora:
        empty_lora_config = LoraConfig(r=rank_unet, init_lora_weights="gaussian", target_modules=[])
        model = get_peft_model(model, empty_lora_config, adapter_name="").merge_and_unload()
    return model

def load_lora_vae_OSE(model, lora_weights_path, merge_lora=False):
    model_ckpt = torch.load(lora_weights_path)
    rank_vae = model_ckpt["rank_vae"]
    # print(model_ckpt["vae_lora_encoder_modules"])
    vae_lora_encoder_modules = convert_vae_target_module(model_ckpt["vae_lora_encoder_modules"])
    vae_lora_decoder_modules = convert_vae_target_module(model_ckpt["vae_lora_decoder_modules"])

    # print(vae_lora_encoder_modules)
    vae_lora_conf_encoder = LoraConfig(r=rank_vae, init_lora_weights="gaussian", target_modules=vae_lora_encoder_modules)
    vae_lora_conf_decoder = LoraConfig(r=rank_vae, init_lora_weights="gaussian", target_modules=vae_lora_decoder_modules)
    
    model = get_peft_model(model, vae_lora_conf_encoder, adapter_name="default_encoder").base_model.model
    # model = get_peft_model(model, vae_lora_conf_decoder, adapter_name="default_decoder").base_model.model
    vae_state_dict = model_ckpt["vae_state_dict"]
    vae_state_dict = convert_vae_state_dict(vae_state_dict)
    
    lora_state_dict = {}
    for key, value in vae_state_dict.items():
        # if "lora" or "input_blocks.0.0" in key:
        #     lora_state_dict[key] = value
        if "default_decoder" in key:
            continue
        else:
            lora_state_dict[key] = value
    
    # lora_state_dict = vae_state_dict
    state_dict_info = model.load_state_dict(lora_state_dict, strict=False)
    if len(state_dict_info.unexpected_keys) == 0:
        print("OSEDiff vae is loaded successfully!")
    else:
        print("Unexpected keys:", state_dict_info.unexpected_keys)
    
    if merge_lora:
        empty_lora_config = LoraConfig(r=rank_vae, init_lora_weights="gaussian", target_modules=[]) # change peft codabase , delete valueerror
        model = get_peft_model(model, empty_lora_config, adapter_name="").merge_and_unload()
    return model

def merge_lora_to_base_model(lora_model):
    for name, lora_module in lora_model.named_modules():
        if hasattr(lora_module, "merge_and_unload"):
            lora_module.merge_and_unload()
            print(f"Merged and removed LoRA layer: {name}")
    return lora_model