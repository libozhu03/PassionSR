import numpy as np
import argparse, os, sys, gc, glob, yaml
import quantization.saw as saw
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from quantization.methods import *
from criterions.methods import *
import wandb
import torch
import torch.nn as nn
import torchvision.utils as vutils
from datetime import datetime
from .saw_layer import get_loss_function

def get_x0_from_noise(sample, model_output, alphas_cumprod, timestep):
    alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    return pred_original_sample

def saw_cali_U_sep(whole_model, quant_config, cali_data, merge, device, ):
    quant_method = quant_config["Unet"]["method"]
    cali_xs, cali_ls, cali_ts, cali_cs, cali_ys, cali_hs, cali_gs = cali_data
    inds = np.arange(cali_xs.shape[0])
    np.random.shuffle(inds)
    print("quantization initialization")
    whole_model.unet.set_quant_state(weight_quant=True, act_quant=True)
    whole_model.unet.set_running_stat(True)
    whole_model.unet.set_all_recon(True)
    batch_size = quant_config["cali_batch_size"]
    for i in tqdm(trange(int(cali_xs.size(0) / batch_size)), desc="Processing",
                    unit="batch", colour="green", position=0):
        end = min((i + 1) * batch_size, cali_xs.size(0))
        with torch.no_grad():
            cali_l = cali_ls[inds[i * batch_size:end]]
            _ = whole_model.unet(cali_l, cali_ts[inds[i * batch_size:end]],
                                    cali_cs[inds[i * batch_size:end]])
    whole_model.unet.set_running_stat(False)
    whole_model.unet.set_all_init(True)

    print("Start saw reconstruction")
    print("Start learning scaling factor:")
    torch.cuda.empty_cache()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = os.path.join(quant_config["output_modelpath"], 'log', "wb", current_time)
    os.makedirs(logdir, exist_ok=True)
    wandb.init(project="esaw_U", name=current_time, dir=logdir, config=quant_config)

    logdir = os.path.join(quant_config["output_modelpath"], 'log', "tb", current_time)
    writer = SummaryWriter(logdir)
    wandb.tensorboard.patch(root_logdir=logdir)

    unet_scale_factor_list = whole_model.unet.get_all_scale_factor()
    train_params = unet_scale_factor_list
    optimizer, criterion, scheduler, psnr_loss = saw.configure_train_component(quant_config, train_params)
    batch_size = 1
    for param in whole_model.parameters():
        if id(param) in [id(p) for p in train_params]:
            param.requires_grad = True
        else:
            param.requires_grad = False
    name_list = ["scale_factor", "offset"]
    for name, param in whole_model.named_parameters():
        if any(key in name for key in name_list):
            def hook_fn(grad, name=name):
                if grad is None:
                    print(f"parameter {name} grad is None")
            param.register_hook(hook_fn)
    save_interval = quant_config["save_interval"]
    current_lr = optimizer.param_groups[0]['lr']
    progress_bar = tqdm(trange(quant_config["cali_epochs"]), desc="Processing", unit="batch", colour="green", position=0)
    global_idx = 0
    for idx in progress_bar:
        optimizer.zero_grad()
        select_batch = range(3,7)
        with torch.no_grad():
            image_list = []
            contrast_image_list = []
            for select_batch_id in select_batch:
                lq = cali_xs[inds[select_batch_id:select_batch_id+1]]
                output = whole_model(lq)
                contrast_image_list.append(cali_gs[inds[select_batch_id:select_batch_id+1]])
                image_list.append(output)
                wandb.log({"quantized_images/s": [wandb.Image(output.squeeze())]})
                wandb.log({"contrast_images/s": [wandb.Image(cali_gs[inds[select_batch_id:select_batch_id+1]].squeeze())]})
            image = torch.cat(image_list, dim=0)
            contrast_image = torch.cat(contrast_image_list, dim=0)

            img_grid = vutils.make_grid(image, nrow=2, normalize=False)
            writer.add_image('images/quantized_images/s', img_grid, global_step=idx)
            c_img_grid = vutils.make_grid(contrast_image, nrow=2, normalize=False)
            writer.add_image('images/contrast_images/s', c_img_grid, global_step=idx)
            for name, param in whole_model.unet.named_parameters():
                    if "scale_factor" in name:
                        writer.add_histogram(f"s_distribution/s/{name}", param.data.cpu().numpy(), bins=100, global_step=global_idx)
            global_idx +=1

        for i in range(int(cali_xs.size(0) / batch_size)):
            whole_model.unet.set_all_init(False)
            end = min((i + 1) * batch_size, cali_xs.size(0))
            lq = cali_xs[inds[i * batch_size:end]]
            lq_latent = cali_ls[inds[i * batch_size:end]]
            output =  whole_model.unet(lq_latent, cali_ts[inds[i * batch_size:end]],
                                    cali_cs[inds[i * batch_size:end]])
            cali_y = cali_ys[inds[i * batch_size:end]]
            # gt_image = cali_gs[inds[select_batch_id:select_batch_id+1]]
            y_l = whole_model.unet2vae(lq_latent, output)
            cy_l = whole_model.unet2vae(lq_latent, cali_y)
            loss = criterion(y_l, cy_l)
            loss.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({"Loss": loss.item(), "LR": current_lr})
            with torch.no_grad():
                output_image = whole_model(lq)
                gt_image =  cali_gs[inds[i * batch_size:end]]
            if writer is not None:
                writer.add_scalar(f'cali_Loss_scaling_factor/{quant_method}_{quant_config["Unet"]["method"]}/s',
                                loss.item(), i * batch_size + idx * cali_xs.size(0))
                wandb.log({f'cali_Loss_scaling_factor/{quant_method}_{quant_config["Unet"]["method"]}/s': loss.item()},
                                )
                writer.add_scalar(f'psnr/{quant_method}_{quant_config["Unet"]["method"]}/s',
                                psnr_loss(output_image, gt_image).item(), i * batch_size + idx * cali_xs.size(0))
                wandb.log({f'psnr/{quant_method}_{quant_config["Unet"]["method"]}/s': psnr_loss(output_image, gt_image).item()},
                                )
        scheduler.step()
        # if idx % save_interval == 0:
        #     method = quant_config["Unet"]["method"]
        #     logdir = os.path.join(quant_config["output_modelpath"], f"PTQ_save/iter_{idx}")
        #     os.makedirs(logdir, exist_ok=True)
        #     if merge:
        #         torch.save(whole_model.unet.state_dict(), os.path.join(logdir, f"unet_ckpt_merge_{method}.pth"))
        #     else:
        #         torch.save(whole_model.unet.state_dict(), os.path.join(logdir, f"unet_ckpt_{method}.pth"))
        #     # save_model_params_to_txt(qnn, os.path.join(logdir, "contrast_model_param.txt"))
        #     print(f"Model saved to {logdir}")

    print("Start calibrating the quantization params")
    unet_quant_param_list = whole_model.unet.get_all_quant_param()
    train_params = unet_quant_param_list
    optimizer, criterion, scheduler, psnr_loss = saw.configure_train_component(quant_config, train_params)
    for param in whole_model.parameters():
        if id(param) in [id(p) for p in train_params]:
            param.requires_grad = True
        else:
            param.requires_grad = False
    name_list = ["x_max", "x_min"]
    for name, param in whole_model.named_parameters():
        if any(key in name for key in name_list):
            def hook_fn(grad, name=name):
                if grad is None:
                    print(f"parameter {name} grad is None")
            param.register_hook(hook_fn)
    save_interval = quant_config["save_interval"]
    current_lr = optimizer.param_groups[0]['lr']
    progress_bar = tqdm(trange(quant_config["cali_epochs"]), desc="Processing", unit="batch", colour="green", position=0)
    global_idx = 0
    for idx in progress_bar:
        optimizer.zero_grad()
        select_batch = range(3,7)
        with torch.no_grad():
            image_list = []
            contrast_image_list = []
            for select_batch_id in select_batch:
                lq = cali_xs[inds[select_batch_id:select_batch_id+1]]
                output = whole_model(lq)
                contrast_image_list.append(cali_gs[inds[select_batch_id:select_batch_id+1]])
                image_list.append(output)
                wandb.log({"quantized_images/q": [wandb.Image(output.squeeze())]})
                wandb.log({"contrast_images/q": [wandb.Image(cali_gs[inds[select_batch_id:select_batch_id+1]].squeeze())]})
            image = torch.cat(image_list, dim=0)
            contrast_image = torch.cat(contrast_image_list, dim=0)

            img_grid = vutils.make_grid(image, nrow=2, normalize=False)
            writer.add_image('images/quantized_images/q', img_grid, global_step=idx)
            c_img_grid = vutils.make_grid(contrast_image, nrow=2, normalize=False)
            writer.add_image('images/contrast_images/q', c_img_grid, global_step=idx)
            # for name, param in whole_model.unet.named_parameters():
            #         if "scale_factor" in name:
            #             writer.add_histogram(f"s_distribution/{name}", param.data.cpu().numpy(), bins=100, global_step=global_idx)
            global_idx +=1

        for i in range(int(cali_xs.size(0) / batch_size)):
            # qnn.set_all_init(False)
            end = min((i + 1) * batch_size, cali_xs.size(0))
            lq = cali_xs[inds[i * batch_size:end]]
            lq_latent = cali_ls[inds[i * batch_size:end]]
            output =  whole_model.unet(lq_latent, cali_ts[inds[i * batch_size:end]],
                                    cali_cs[inds[i * batch_size:end]])
            cali_y = cali_ys[inds[i * batch_size:end]]
            # gt_image = cali_gs[inds[select_batch_id:select_batch_id+1]]
            y_l = whole_model.unet2vae(lq_latent, output)
            cy_l = whole_model.unet2vae(lq_latent, cali_y)
            loss = criterion(y_l, cy_l)
            loss.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({"Loss": loss.item(), "LR": current_lr})
            with torch.no_grad():
                output_image = whole_model(lq)
                gt_image =  cali_gs[inds[i * batch_size:end]]
            if writer is not None:
                writer.add_scalar(f'cali_Loss_scaling_factor/{quant_method}_{quant_config["Unet"]["method"]}/q',
                                loss.item(), i * batch_size + idx * cali_xs.size(0))
                wandb.log({f'cali_Loss_scaling_factor/{quant_method}_{quant_config["Unet"]["method"]}/q': loss.item()},
                                )
                writer.add_scalar(f'psnr/{quant_method}_{quant_config["Unet"]["method"]}/q',
                                psnr_loss(output_image, gt_image).item(), i * batch_size + idx * cali_xs.size(0))
                wandb.log({f'psnr/{quant_method}_{quant_config["Unet"]["method"]}/q': psnr_loss(output_image, gt_image).item()},
                                )
        scheduler.step()
        # if idx % save_interval == 0:
        #     method = quant_config["Unet"]["method"]
        #     logdir = os.path.join(quant_config["output_modelpath"], f"PTQ_save/iter_{idx}")
        #     os.makedirs(logdir, exist_ok=True)
        #     if merge:
        #         torch.save(whole_model.unet.state_dict(), os.path.join(logdir, f"unet_ckpt_merge_{method}.pth"))
        #     else:
        #         torch.save(whole_model.unet.state_dict(), os.path.join(logdir, f"unet_ckpt_{method}.pth"))
        #     # save_model_params_to_txt(qnn, os.path.join(logdir, "contrast_model_param.txt"))
        #     print(f"Model saved to {logdir}")

    wandb.finish()
    writer.close()
    return whole_model

def saw_cali_UV_sep(unet_f, vae_f, whole_model, quant_config, cali_data, merge, device, ):
    quant_method = quant_config["Unet"]["method"]
    cali_xs, cali_ls, cali_ts, cali_cs, cali_ys, cali_hs, cali_gs = cali_data
    inds = np.arange(cali_xs.shape[0])
    np.random.shuffle(inds)
    print("quantization initialization")
    whole_model.unet.set_quant_state(weight_quant=True, act_quant=True)
    whole_model.unet.set_running_stat(True)
    whole_model.unet.set_all_recon(True)
    whole_model.vae.set_quant_state(weight_quant=True, act_quant=True)
    whole_model.vae.set_running_stat(True)
    whole_model.vae.set_all_recon(True)
    batch_size = quant_config["cali_batch_size"]
    for i in tqdm(trange(int(cali_xs.size(0) / batch_size)), desc="Processing",
                    unit="batch", colour="green", position=0):
        end = min((i + 1) * batch_size, cali_xs.size(0))
        with torch.no_grad():
            cali_x = cali_xs[inds[i * batch_size:end]]
            _ = whole_model(cali_x, cali_cs[inds[i * batch_size:end]])
    whole_model.unet.set_running_stat(False)
    whole_model.unet.set_all_init(True)
    whole_model.vae.set_running_stat(False)
    whole_model.vae.set_all_init(True)

    print("Start saw reconstruction")
    print("Start learning scaling factor:")
    torch.cuda.empty_cache()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = os.path.join(quant_config["output_modelpath"], 'log', "wb", current_time)
    os.makedirs(logdir, exist_ok=True)
    wandb.init(project="esaw_UV", name=current_time, dir=logdir, config=quant_config)
    logdir = os.path.join(quant_config["output_modelpath"], 'log', "tb", current_time)
    writer = SummaryWriter(logdir)
    wandb.tensorboard.patch(root_logdir=logdir)

    # calibrate encoder
    encoder_scale_factor_list = whole_model.vae.get_all_scale_factor(pick="encoder")
    train_params = encoder_scale_factor_list
    batch_size = 1
    for param in whole_model.parameters():
        if id(param) in [id(p) for p in train_params]:
            param.requires_grad = True
        else:
            param.requires_grad = False
    name_list = ["scale_factor",  "offset"]
    for name, param in whole_model.named_parameters():
        if param.requires_grad and any(key in name for key in name_list):
            def hook_fn(grad, name=name):
                if grad is None:
                    print(f"parameter {name} grad is None")
            param.register_hook(hook_fn)
    save_interval = quant_config["encoder_train"]["save_interval"]
    optimizer, criterion, scheduler, psnr_loss = saw.configure_train_component(quant_config["encoder_train"], train_params)
    current_lr = optimizer.param_groups[0]['lr']
    progress_bar = tqdm(trange(quant_config["encoder_train"]["cali_epochs"]), desc="Processing", unit="batch", colour="green", position=0)
    for idx in progress_bar:
        optimizer.zero_grad()
        for i in range(int(cali_xs.size(0) / batch_size)):
            end = min((i + 1) * batch_size, cali_xs.size(0))
            lq = cali_xs[inds[i * batch_size:end]]
            lq = lq * 2 - 1.0
            latent_out = whole_model.vae.encode(lq).sample() * whole_model.encode_scaling_factor
            lq_latent = cali_ls[inds[i * batch_size:end]]
            loss = criterion(latent_out, lq_latent)
            loss.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({"Loss": loss.item(), "LR": current_lr})
            if writer is not None:
                    writer.add_scalar(f'cali_Loss_encoder/{quant_method}_{quant_config["Unet"]["method"]}/s',
                                    loss.item(), i * batch_size + idx * cali_xs.size(0))
                    wandb.log({f'cali_Loss_encoder/{quant_method}_{quant_config["Unet"]["method"]}/s': loss.item()},
                                    )
        if scheduler is not None:
            scheduler.step()

    encoder_quant_param_list = whole_model.vae.get_all_quant_param(pick="encoder")
    train_params = encoder_quant_param_list
    for param in whole_model.parameters():
        if id(param) in [id(p) for p in train_params]:
            param.requires_grad = True
        else:
            param.requires_grad = False
    name_list = ["x_max",  "x_min"]
    for name, param in whole_model.named_parameters():
        if param.requires_grad and any(key in name for key in name_list):
            def hook_fn(grad, name=name):
                if grad is None:
                    print(f"parameter {name} grad is None")
            param.register_hook(hook_fn)
    save_interval = quant_config["encoder_train"]["save_interval"]
    optimizer, criterion, scheduler, psnr_loss = saw.configure_train_component(quant_config["encoder_train"], train_params)
    current_lr = optimizer.param_groups[0]['lr']
    progress_bar = tqdm(trange(quant_config["encoder_train"]["cali_epochs"]), desc="Processing", unit="batch", colour="green", position=0)
    for idx in progress_bar:
        optimizer.zero_grad()
        for i in range(int(cali_xs.size(0) / batch_size)):
            end = min((i + 1) * batch_size, cali_xs.size(0))
            lq = cali_xs[inds[i * batch_size:end]]
            lq = lq * 2 - 1.0
            latent_out = whole_model.vae.encode(lq).sample() * whole_model.encode_scaling_factor
            lq_latent = cali_ls[inds[i * batch_size:end]]
            loss = criterion(latent_out, lq_latent)
            loss.backward()
            optimizer.step()
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({"Loss": loss.item(), "LR": current_lr})
            if writer is not None:
                    writer.add_scalar(f'cali_Loss_encoder/{quant_method}_{quant_config["Unet"]["method"]}/q',
                                    loss.item(), i * batch_size + idx * cali_xs.size(0))
                    wandb.log({f'cali_Loss_encoder/{quant_method}_{quant_config["Unet"]["method"]}/q': loss.item()},
                                    )
        if scheduler is not None:
            scheduler.step()
    
    batch_size = 1
    cali_ql_list = []
    print("prepare quantized latent:")
    with torch.no_grad():
        for i in tqdm(trange(int(cali_xs.size(0) / batch_size)), desc="Processing",
                        unit="batch", colour="green", position=0):
            end = min((i + 1) * batch_size, cali_xs.size(0))
            lq = cali_xs[i * batch_size:end] * 2 - 1.0
            lq_latent = whole_model.vae.encode(lq).sample() * whole_model.encode_scaling_factor
            cali_ql_list.append(lq_latent)
    cali_qls = torch.cat(cali_ql_list, dim=0)

    # calibrate Unet
    unet_scale_factor_list = whole_model.unet.get_all_scale_factor()
    train_params = unet_scale_factor_list
    # train_params = unet_scale_factor_list + unet_quant_param_list
    for param in whole_model.parameters():
        if id(param) in [id(p) for p in train_params]:
            param.requires_grad = True
        else:
            param.requires_grad = False
    name_list = ["scale_factor", "offset"]
    for name, param in whole_model.named_parameters():
        if param.requires_grad and any(key in name for key in name_list):
            def hook_fn(grad, name=name):
                if grad is None:
                    print(f"parameter {name} grad is None")
            param.register_hook(hook_fn)
    optimizer, criterion, scheduler, psnr_loss = saw.configure_train_component(quant_config["Unet_train"], train_params)
    current_lr = optimizer.param_groups[0]['lr']
    progress_bar = tqdm(trange(quant_config["Unet_train"]["cali_epochs"]), desc="Processing", unit="batch", colour="green", position=0)
    save_interval = quant_config["Unet_train"]["save_interval"]
    global_idx = 0
    for idx in progress_bar:
        optimizer.zero_grad()
        select_batch = range(3,7)
        with torch.no_grad():
            image_list = []
            contrast_image_list = []
            for select_batch_id in select_batch:
                ind = inds[select_batch_id:select_batch_id+1]
                latent_lq = cali_qls[ind]
                output = whole_model.unet(latent_lq, cali_ts[ind], cali_cs[ind])
                latent_out = get_x0_from_noise(latent_lq, output, whole_model.alphas_cumprod, 999)
                out_image = vae_f.decode(latent_out.to(whole_model.weight_dtype) / whole_model.decode_scaling_factor)
                out_image = out_image * 0.5 + 0.5
                contrast_image_list.append(cali_gs[ind])
                image_list.append(out_image)
                wandb.log({"quantized_images/s": [wandb.Image(out_image)]},)
                wandb.log({"contrast_images/s": [wandb.Image(cali_gs[ind])]},)
            image = torch.cat(image_list, dim=0)
            contrast_image = torch.cat(contrast_image_list, dim=0)

            img_grid = vutils.make_grid(image, nrow=2, normalize=False)
            writer.add_image('images_Unet/quantized_images/s', img_grid, global_step=idx)
            c_img_grid = vutils.make_grid(contrast_image, nrow=2, normalize=False)
            writer.add_image('images_Unet/contrast_images/s', c_img_grid, global_step=idx)

        for i in range(int(cali_xs.size(0) / batch_size)):
            # qnn.set_all_init(False)
            end = min((i + 1) * batch_size, cali_xs.size(0))
            ind = inds[i * batch_size:end]
            latent_lq = cali_qls[ind]
            output = whole_model.unet(latent_lq, cali_ts[ind], cali_cs[ind])
            latent_out = get_x0_from_noise(latent_lq, output, whole_model.alphas_cumprod, 999)
            latent = cali_hs[ind]
            u_gt = cali_ys[ind]
            loss = criterion(output, u_gt)
            loss.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({"Loss": loss.item(), "LR": current_lr})
            with torch.no_grad():
                output_image = vae_f.decode(latent_out.to(whole_model.weight_dtype) / whole_model.decode_scaling_factor)
                output_image  = output_image * 0.5 + 0.5
                contrast_image = cali_gs[ind]
            if writer is not None:
                writer.add_scalar(f'cali_Loss_Unet/{quant_method}_{quant_config["Unet"]["method"]}/s',
                                loss.item(), i * batch_size + idx * cali_xs.size(0))
                wandb.log({f'cali_Loss_Unet/{quant_method}_{quant_config["Unet"]["method"]}/s': loss.item()},
                                )
                writer.add_scalar(f'psnr_Unet/{quant_method}_{quant_config["Unet"]["method"]}/s',
                                psnr_loss(output_image, contrast_image).item(), i * batch_size + idx * cali_xs.size(0))
                wandb.log({f'psnr_Unet/{quant_method}_{quant_config["Unet"]["method"]}/s': psnr_loss(output_image, contrast_image).item()},
                                )
                global_idx += 1
        if scheduler is not None:
            scheduler.step()

    unet_quant_param_list = whole_model.unet.get_all_quant_param(user="all")
    train_params = unet_quant_param_list
    for param in whole_model.parameters():
        if id(param) in [id(p) for p in train_params]:
            param.requires_grad = True
        else:
            param.requires_grad = False
    name_list = ["x_max", "x_min"]
    for name, param in whole_model.named_parameters():
        if param.requires_grad and any(key in name for key in name_list):
            def hook_fn(grad, name=name):
                if grad is None:
                    print(f"parameter {name} grad is None")
            param.register_hook(hook_fn)
    optimizer, criterion, scheduler, psnr_loss = saw.configure_train_component(quant_config["Unet_train"], train_params)
    current_lr = optimizer.param_groups[0]['lr']
    progress_bar = tqdm(trange(quant_config["Unet_train"]["cali_epochs"]), desc="Processing", unit="batch", colour="green", position=0)
    save_interval = quant_config["Unet_train"]["save_interval"]
    global_idx = 0
    for idx in progress_bar:
        optimizer.zero_grad()
        select_batch = range(3,7)
        with torch.no_grad():
            image_list = []
            contrast_image_list = []
            for select_batch_id in select_batch:
                ind = inds[select_batch_id:select_batch_id+1]
                latent_lq = cali_qls[ind]
                output = whole_model.unet(latent_lq, cali_ts[ind], cali_cs[ind])
                latent_out = get_x0_from_noise(latent_lq, output, whole_model.alphas_cumprod, 999)
                out_image = vae_f.decode(latent_out.to(whole_model.weight_dtype) / whole_model.decode_scaling_factor)
                out_image = out_image * 0.5 + 0.5
                contrast_image_list.append(cali_gs[ind])
                image_list.append(out_image)
                wandb.log({"quantized_images/q": [wandb.Image(out_image)]},)
                wandb.log({"contrast_images/q": [wandb.Image(cali_gs[ind])]},)
            image = torch.cat(image_list, dim=0)
            contrast_image = torch.cat(contrast_image_list, dim=0)

            img_grid = vutils.make_grid(image, nrow=2, normalize=False)
            writer.add_image('images_Unet/quantized_images/q', img_grid, global_step=idx)
            c_img_grid = vutils.make_grid(contrast_image, nrow=2, normalize=False)
            writer.add_image('images_Unet/contrast_images/q', c_img_grid, global_step=idx)

        for i in range(int(cali_xs.size(0) / batch_size)):
            # qnn.set_all_init(False)
            end = min((i + 1) * batch_size, cali_xs.size(0))
            ind = inds[i * batch_size:end]
            latent_lq = cali_qls[ind]
            output = whole_model.unet(latent_lq, cali_ts[ind], cali_cs[ind])
            latent_out = get_x0_from_noise(latent_lq, output, whole_model.alphas_cumprod, 999)
            # label_out = cali_ys[inds[i * batch_size:end]]
            # latent = get_x0_from_noise(cali_ls[ind], label_out, whole_model.alphas_cumprod, 999)
            latent = cali_hs[ind]
            loss = criterion(latent_out, latent)
            loss.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({"Loss": loss.item(), "LR": current_lr})
            with torch.no_grad():
                output_image = vae_f.decode(latent_out.to(whole_model.weight_dtype) / whole_model.decode_scaling_factor)
                output_image  = output_image * 0.5 + 0.5
                contrast_image = cali_gs[ind]
            if writer is not None:
                writer.add_scalar(f'cali_Loss_Unet/{quant_method}_{quant_config["Unet"]["method"]}/q',
                                loss.item(), i * batch_size + idx * cali_xs.size(0))
                wandb.log({f'cali_Loss_Unet/{quant_method}_{quant_config["Unet"]["method"]}/q': loss.item()},
                                )
                writer.add_scalar(f'psnr_Unet/{quant_method}_{quant_config["Unet"]["method"]}/q',
                                psnr_loss(output_image, contrast_image).item(), i * batch_size + idx * cali_xs.size(0))
                wandb.log({f'psnr_Unet/{quant_method}_{quant_config["Unet"]["method"]}/q': psnr_loss(output_image, contrast_image).item()},
                                )
                global_idx += 1
        if scheduler is not None:
            scheduler.step()

    batch_size = 1
    cali_hl_list = []
    print("prepare quantized output latent:")
    with torch.no_grad():
        for i in tqdm(trange(int(cali_xs.size(0) / batch_size)), desc="Processing",
                        unit="batch", colour="green", position=0):
            end = min((i + 1) * batch_size, cali_xs.size(0))
            lq_latent = cali_qls[i * batch_size:end]
            unet_out = whole_model.unet(lq_latent, cali_ts[i * batch_size:end], cali_cs[i * batch_size:end])
            hq_latent = get_x0_from_noise(lq_latent, unet_out, whole_model.alphas_cumprod, 999)
            cali_hl_list.append(hq_latent)
    cali_hls = torch.cat(cali_hl_list, dim=0)

    # calibrate decoder
    decoder_scale_factor_list = whole_model.vae.get_all_scale_factor(pick="decoder")
    train_params = decoder_scale_factor_list
    for param in whole_model.parameters():
        if id(param) in [id(p) for p in train_params]:
            param.requires_grad = True
        else:
            param.requires_grad = False
    name_list = ["scale_factor", "offset"]
    for name, param in whole_model.named_parameters():
        if param.requires_grad and any(key in name for key in name_list):
            def hook_fn(grad, name=name):
                if grad is None:
                    print(f"parameter {name} grad is None")
            param.register_hook(hook_fn)
    save_interval = quant_config["decoder_train"]["save_interval"]
    optimizer, criterion, scheduler, psnr_loss = saw.configure_train_component(quant_config["decoder_train"], train_params)
    current_lr = optimizer.param_groups[0]['lr']
    progress_bar = tqdm(trange(quant_config["decoder_train"]["cali_epochs"]), desc="Processing", unit="batch", colour="green", position=0)
    for idx in progress_bar:
        optimizer.zero_grad()
        for i in range(int(cali_xs.size(0) / batch_size)):
            end = min((i + 1) * batch_size, cali_xs.size(0))
            ind = inds[i * batch_size:end]
            hl = cali_hls[ind]
            output_image =  whole_model.vae.decode(hl.to(whole_model.weight_dtype)/ whole_model.decode_scaling_factor) 
            output_image = output_image * 0.5 + 0.5
            gt_image = cali_gs[ind]
            loss = criterion(output_image, gt_image)
            loss.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({"Loss": loss.item(), "LR": current_lr})
            if writer is not None:
                    writer.add_scalar(f'cali_Loss_decoder/{quant_method}_{quant_config["Unet"]["method"]}/s',
                                    loss.item(), i * batch_size + idx * cali_xs.size(0))
                    wandb.log({f'cali_Loss_decoder/{quant_method}_{quant_config["Unet"]["method"]}/s': loss.item()},
                                    )
                    writer.add_scalar(f'psnr_decoder/{quant_method}_{quant_config["Unet"]["method"]}/s',
                                psnr_loss(output_image, gt_image).item(), i * batch_size + idx * cali_xs.size(0))
                    wandb.log({f'psnr_decoder/{quant_method}_{quant_config["Unet"]["method"]}/s': psnr_loss(output_image, gt_image).item()},
                                    )
                    writer.add_image('images_decoder/quantized_images/s', output_image,
                                    global_step=idx* int(cali_xs.size(0) / batch_size)+i, dataformats='NCHW')
                    writer.add_image('images_decoder/fp_images/s', gt_image,
                                    global_step=idx* int(cali_xs.size(0) / batch_size)+i, dataformats='NCHW')
                    wandb.log({"quantized_images_decoder/s": [wandb.Image(output_image)]},)
                    wandb.log({"fp_images_decoder/s": [wandb.Image(gt_image)]},)

        if scheduler is not None:
            scheduler.step()

    decoder_quant_param_list = whole_model.vae.get_all_quant_param(pick="decoder")
    train_params = decoder_quant_param_list
    for param in whole_model.parameters():
        if id(param) in [id(p) for p in train_params]:
            param.requires_grad = True
        else:
            param.requires_grad = False
    name_list = ["x_max", "x_min"]
    for name, param in whole_model.named_parameters():
        if param.requires_grad and any(key in name for key in name_list):
            def hook_fn(grad, name=name):
                if grad is None:
                    print(f"parameter {name} grad is None")
            param.register_hook(hook_fn)
    save_interval = quant_config["decoder_train"]["save_interval"]
    optimizer, criterion, scheduler, psnr_loss = saw.configure_train_component(quant_config["decoder_train"], train_params)
    current_lr = optimizer.param_groups[0]['lr']
    progress_bar = tqdm(trange(quant_config["decoder_train"]["cali_epochs"]), desc="Processing", unit="batch", colour="green", position=0)
    for idx in progress_bar:
        optimizer.zero_grad()
        for i in range(int(cali_xs.size(0) / batch_size)):
            end = min((i + 1) * batch_size, cali_xs.size(0))
            ind = inds[i * batch_size:end]
            hl = cali_hls[ind]
            output_image =  whole_model.vae.decode(hl.to(whole_model.weight_dtype)/ whole_model.decode_scaling_factor) 
            output_image = output_image * 0.5 + 0.5
            gt_image = cali_gs[ind]
            loss = criterion(output_image, gt_image)
            loss.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({"Loss": loss.item(), "LR": current_lr})
            if writer is not None:
                    writer.add_scalar(f'cali_Loss_decoder/{quant_method}_{quant_config["Unet"]["method"]}/q',
                                    loss.item(), i * batch_size + idx * cali_xs.size(0))
                    wandb.log({f'cali_Loss_decoder/{quant_method}_{quant_config["Unet"]["method"]}/q': loss.item()},
                                    )
                    writer.add_scalar(f'psnr_decoder/{quant_method}_{quant_config["Unet"]["method"]}/q',
                                psnr_loss(output_image, gt_image).item(), i * batch_size + idx * cali_xs.size(0))
                    wandb.log({f'psnr_decoder/{quant_method}_{quant_config["Unet"]["method"]}/q': psnr_loss(output_image, gt_image).item()},
                                    )
                    writer.add_image('images_decoder/quantized_images/q', output_image,
                                    global_step=idx* int(cali_xs.size(0) / batch_size)+i, dataformats='NCHW')
                    writer.add_image('images_decoder/fp_images/q', gt_image,
                                    global_step=idx* int(cali_xs.size(0) / batch_size)+i, dataformats='NCHW')
                    wandb.log({"quantized_images_decoder/q": [wandb.Image(output_image)]},)
                    wandb.log({"fp_images_decoder/q": [wandb.Image(gt_image)]},)

        if scheduler is not None:
            scheduler.step()

    wandb.finish()
    writer.close()
    return whole_model