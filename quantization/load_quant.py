import torch

from quantization.methods import adjust_model_params_shape
import quantization.saw as saw

def load_Quantmodel(model, quant_config, device):
    quant_method = quant_config["Unet"]["method"]
    if quant_method == "saw" or quant_method == "saw_sep":
        if quant_config["only_Unet"]:
            quant_config_C = saw.QuantizeModel_config(quant_config["Unet"])
            qnn = saw.QuantWrapper(model, quant_config_C, device=device)
            qnn.set_quant_state(weight_quant=True, act_quant=True)
            qnn = adjust_model_params_shape(qnn, torch.load(quant_config["Unet"]["quant_ckpt"]))
            # set_qnn_init(qnn)
            state_dict_info = qnn.load_state_dict(torch.load(quant_config["Unet"]["quant_ckpt"]), strict=False)
            if len(state_dict_info.unexpected_keys) == 0:
                print("quantization parameters is loaded successfully!")
            else:
                print("quantization parameters Unexpected keys:", state_dict_info.unexpected_keys)
            qnn.set_running_stat(False)
            qnn.set_record(False)
            qnn.set_all_init(True, 'all')
            # qnn.set_all_init(False, 'all')
            qnn.set_all_recon(True)
            # qnn.set_all_recon(False)
            qnn.set_all_recon_init(True)
            qnn.set_quant_state(weight_quant=True, act_quant=True)
            qnn.eval()
            return qnn
        else:
            unet, vae = model

            quant_config_u = saw.QuantizeModel_config(quant_config["Unet"])
            unet_q = saw.QuantWrapper(unet, quant_config_u, device=device)
            unet_q.set_quant_state(weight_quant=True, act_quant=True)
            unet_q = adjust_model_params_shape(unet_q, torch.load(quant_config["Unet"]["quant_ckpt"]))
            state_dict_info = unet_q.load_state_dict(torch.load(quant_config["Unet"]["quant_ckpt"]), strict=False)
            if len(state_dict_info.unexpected_keys) == 0:
                print("Unet quantization parameters is loaded successfully!")
            else:
                print("Unet quantization parameters Unexpected keys:", state_dict_info.unexpected_keys)
            unet_q.set_running_stat(False)
            unet_q.set_record(False)
            unet_q.set_all_init(True, 'all')
            unet_q.set_all_recon(True)
            # unet_q.set_all_recon(False)
            unet_q.set_all_recon_init(True)
            unet_q.eval()

            quant_config_v = saw.QuantizeModel_config(quant_config["Vae"])
            vae_q = saw.QuantWrapper(vae, quant_config_v, device=device)
            vae_q.set_quant_state(weight_quant=True, act_quant=True)
            vae_q = adjust_model_params_shape(vae_q, torch.load(quant_config["Vae"]["quant_ckpt"]))
            state_dict = torch.load(quant_config["Vae"]["quant_ckpt"], map_location=lambda storage, loc: storage.cuda(device))
            state_dict_info = vae_q.load_state_dict(state_dict, strict=False)
            if len(state_dict_info.unexpected_keys) == 0:
                print("Vae quantization parameters is loaded successfully!")
            else:
                print("Vae quantization parameters Unexpected keys:", state_dict_info.unexpected_keys)
            vae_q.set_running_stat(False)
            vae_q.set_record(False)
            vae_q.set_all_init(True, 'all')
            vae_q.set_all_recon(True)
            # vae_q.set_all_recon(False)
            vae_q.set_all_recon_init(True)
            vae_q.eval()
            return unet_q, vae_q.model
    else:
        raise ValueError(f"Quantization method {quant_method} is not supported")
    

