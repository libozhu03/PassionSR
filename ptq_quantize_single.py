import argparse, os, yaml, sys
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as F
from pytorch_lightning import seed_everything
import glob

from diffusers import DDIMScheduler
from ldm.lora.lora_unet_loader import load_lora_unet_OSE, load_lora_vae_OSE
from quantization.apply_quant_ldm import *
from quantization.methods import *
from ldm.lora.load_model import load_model_from_config

tensor_transforms = transforms.Compose([transforms.ToTensor()])

ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class OSEDiff_ptq(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        config = OmegaConf.load(opt['basic_config']['config'])

        ckpt_path = opt['basic_config']['ckpt']
        self.model = load_model_from_config(config, ckpt_path)
        self.vae = self.model.first_stage_model
        self.unet = self.model.model.diffusion_model
        self.unet_config = self.model.model.diffusion_model_config["params"]
        self.encode_scaling_factor = self.model.scale_factor
        self.decode_scaling_factor = self.model.scale_factor
        del self.model
        torch.cuda.empty_cache()

        self.device = opt["device"]
        self.weight_dtype = torch.float32
        self.timesteps = torch.tensor([999], device=self.device)
        self.noise_scheduler = DDIMScheduler.from_pretrained(opt['basic_config']['pretrained_model_name_or_path'], subfolder="scheduler")
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device).to(self.weight_dtype)
        self.merge_lora = opt['basic_config']['merge_lora']
        self.load_lora_ckpt(opt['basic_config']['lora_weights_path'], opt['basic_config']['merge_lora'])
        self.load_context_embedding(opt)
        self.quantize = opt['quantize_config']['quantize']
        if self.quantize:
            self.quant_config = opt['quantize_config']

    def load_context_embedding(self, opt):
        empty_context_embedding = torch.load(opt['basic_config']["context_embedding_path"]).to(self.device)
        self.empty_context_embedding = empty_context_embedding
        # self.empty_context_embedding = torch.rand_like(empty_context_embedding, dtype=torch.float)

    def load_lora_ckpt(self, lora_weights_path, merge_lora):
        self.unet = load_lora_unet_OSE(self.unet, lora_weights_path, merge_lora)
        # save_model_to_txt(self.unet, "results/visual/Unet.txt")
        self.vae = load_lora_vae_OSE(self.vae, lora_weights_path, merge_lora)
        # save_model_to_txt(self.vae, "results/visual/Vae.txt")

    @torch.no_grad()
    def unet2image(self, lq_latent, unet_output):
        time_steps = 999
        alphas_cumprod = self.alphas_cumprod.requires_grad_(True)
        x_0 = get_x0_from_noise(
            lq_latent.to(torch.float64),
            unet_output.to(torch.float64), alphas_cumprod, time_steps
        ).to(torch.float32)
        output_image = (self.vae.decode(x_0.to(self.weight_dtype) / self.decode_scaling_factor)).clamp(-1, 1)
        output_image = output_image * 0.5 + 0.5
        return output_image

    def unet2vae(self, lq_latent, unet_output):
        time_steps = 999
        alphas_cumprod = self.alphas_cumprod.requires_grad_(True)
        x_0 = get_x0_from_noise(
            lq_latent.to(torch.float64),
            unet_output.to(torch.float64), alphas_cumprod, time_steps
        ).to(torch.float32)
        return x_0

    def forward(self, lq, context_embedding=None):
        if context_embedding is None:
            context_embedding = self.empty_context_embedding
        lq = lq * 2 - 1.0
        lq_latent = self.vae.encode(lq.to(self.weight_dtype)).sample() * self.encode_scaling_factor
        noise_pred =  self.unet.model(lq_latent, self.timesteps, context_embedding)
        time_steps = 999
        x_0 = get_x0_from_noise(
            lq_latent.to(torch.float64),
            noise_pred.to(torch.float64), self.alphas_cumprod, time_steps
            ).to(torch.float32)
        output_image = (self.vae.decode(x_0.to(self.weight_dtype) / self.decode_scaling_factor))
        output_image = output_image * 0.5 + 0.5
        return output_image

    def preprocee_image(self, image):
        ori_width, ori_height = image.size
        rscale = self.opt["basic_config"]["upscale"]
        process_size = self.opt["basic_config"]["process_size"]
        if ori_width < process_size//rscale or ori_height < process_size//rscale:
            scale = (process_size//rscale)/min(ori_width, ori_height)
            image = image.resize((int(scale*ori_width), int(scale*ori_height)))
        image = image.resize((image.size[0]*rscale, image.size[1]*rscale))
        new_width = image.width - image.width % 8
        new_height = image.height - image.height % 8
        image = image.resize((new_width, new_height), Image.LANCZOS)
        return image, new_width, new_height

    @torch.no_grad()
    def callibaration(self, valid_len=None):
        # get all input images
        cali_lr_path = os.path.join(self.opt["cali_img_path"], "lr")
        cali_hr_path = os.path.join(self.opt["cali_img_path"], "hr")
        image_names = sorted(glob.glob(f'{cali_lr_path}/*.[jpJP][pnPN]*[gG]'))
        gt_image_names = [os.path.join(cali_hr_path, os.path.basename(image_name).replace("lr", "hr")) for image_name in image_names]

        # random.shuffle(image_names)
        cali_x_list = []
        cali_l_list = []
        cali_t_list = []
        cali_c_list = []
        cali_y_list = []
        cali_h_list = []
        cali_g_list = []
        for index, image_name in enumerate(tqdm(image_names, desc="Processing", unit="picture", colour="blue")):
            input_image = Image.open(image_name).convert('RGB')
            input_image, w, h = self.preprocee_image(input_image)

            gt_image = Image.open(gt_image_names[index]).convert('RGB')
            gt_image = gt_image.resize((w, h), Image.LANCZOS)

            prompt_embeds = self.empty_context_embedding
            cali_c_list.append(prompt_embeds)

            lq = F.to_tensor(input_image).unsqueeze(0)
            gt = F.to_tensor(gt_image).unsqueeze(0)
            lq = lq.to(self.device)
            cali_x_list.append(lq)
            lq = lq * 2 - 1.0
            lq_latent = self.vae.encode(lq.to(self.weight_dtype)).sample() * self.encode_scaling_factor
            cali_l_list.append(lq_latent)
            cali_t_list.append(self.timesteps)
            cali_g_list.append(gt)
            u_out = self.unet(lq_latent, self.timesteps, prompt_embeds)
            cali_y_list.append(u_out)
            x_0 = get_x0_from_noise(
                lq_latent.to(torch.float64),
                u_out.to(torch.float64), self.alphas_cumprod, self.timesteps
                ).to(torch.float32)
            cali_h_list.append(x_0)
            # out = self.unet2image(lq_latent, u_out)
        cali_x = torch.cat(cali_x_list, dim=0)
        cali_l = torch.cat(cali_l_list, dim=0)
        cali_c = torch.cat(cali_c_list, dim=0)
        cali_t = torch.cat(cali_t_list, dim=0)
        cali_g = torch.cat(cali_g_list, dim=0)
        cali_h = torch.cat(cali_h_list, dim=0)
        cali_y = torch.cat(cali_y_list, dim=0)
        if valid_len is None:
            valid_len = len(cali_t_list)
        cali_data = [cali_x[:valid_len].to(self.device), cali_l[:valid_len].to(self.device),cali_t[:valid_len].to(self.device), 
                        cali_c[:valid_len].to(self.device), cali_y[:valid_len].to(self.device),cali_h[:valid_len].to(self.device), cali_g[:valid_len].to(self.device)]
        print(f"callibration data shape: {cali_x.shape}, {cali_l.shape}, {cali_t.shape}, {cali_c.shape}, {cali_y.shape}, {cali_h.shape}, {cali_g.shape}")
        return cali_data

def get_x0_from_noise(sample, model_output, alphas_cumprod, timestep):
    alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    return pred_original_sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config.yaml", help="Path to the YAML config file")
    args = parser.parse_args()

    opt = load_config(args.config_file)
    print(yaml.dump(opt, default_flow_style=False))

    seed_everything(opt['basic_config']['seed'])

    device = opt['device']
    weight_dtype = torch.float32
    opt['weight_dtype'] = weight_dtype

    model = OSEDiff_ptq(opt)
    model = model.to(device)

    if model.quantize:
        apply_quant(model)

if __name__ == "__main__":
    main()
