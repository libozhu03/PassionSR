import torch
from torch import nn
import yaml, os

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_model_to_txt(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write("Model Architecture:\n")
        f.write(str(model) + "\n\n")

    print(f"Model details saved to {file_path}")

def save_model_params_to_txt(model, file_path='model_params.txt'):
    with open(file_path, 'w') as f:
        for name, param in model.named_parameters():
            f.write(f"Parameter: {name}\n")
            param_flat = param.view(-1)[:10]
            f.write(f"Values: {param_flat.tolist()}\n")
            f.write("="*50 + "\n")
    print(f"Model parameters have been saved to {file_path}")

def adjust_model_params_shape(model: nn.Module, state_dict: dict):

    for name, param in model.named_parameters():
        if name in state_dict:
            if param.shape != state_dict[name].shape:
                # print(f"Parameter shape mismatch at '{name}': Model param shape: {param.shape}, "
                #       f"State_dict param shape: {state_dict[name].shape}")
                module_name = '.'.join(name.split('.')[:-1])
                param_name = name.split('.')[-1]

                target_module = model
                if module_name:
                    target_module = dict(model.named_modules())[module_name]

                with torch.no_grad():
                    new_param = torch.nn.Parameter(torch.empty(state_dict[name].shape).to(param.device))
                    setattr(target_module, param_name, new_param)
                # if "scale_factor" in name:
                #     print(f"Parameter '{name}' in the model has been reshaped to {new_param.shape}.")
        # if "scale_factor" in name:
        #     print(f"Parameter '{name}' in the model")
    return model