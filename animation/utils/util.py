import os
import imageio
import numpy as np
from typing import Union
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torchvision
import torch.distributed as dist
import flow_vis_torch

from safetensors import safe_open
from tqdm import tqdm
from einops import rearrange
from animation.utils.convert_from_ckpt import (
    convert_ldm_unet_checkpoint,
    convert_ldm_clip_checkpoint,
    convert_ldm_vae_checkpoint,
)
from animation.utils.convert_lora_safetensor_to_diffusers import (
    convert_lora,
    convert_motion_lora_ckpt_to_diffusers,
)

from matplotlib.backends.backend_agg import FigureCanvasAgg

def save_flow_grid_pillow_arrow_optimized(flow_maps: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    """
    Save the optical flow maps as a video grid augmented with arrows depicting the flow.
    Parameters:
    - flow_maps: Tensor of shape (b, t, c, h, w) containing optical flow maps.
    - path: Path to save the video.
    - rescale: Whether to rescale the values to [0, 1].
    - n_rows: Number of rows in the grid.
    - fps: Frames per second for the video.
    """
    print('starting ops')
    flow_maps = rearrange(flow_maps, "b t c h w -> t b c h w")
    
    # Fixed grid points for arrows (compute once)
    h, w = flow_maps.shape[3], flow_maps.shape[4]
    step = 30
    y_grid, x_grid = np.meshgrid(np.arange(0, h, step), np.arange(0, w, step), indexing='ij')

    print('comps done')
    
    # Pre-create figure and axes
    fig, ax = plt.subplots(figsize=(w / 20, h / 20), dpi=72)
    ax.axis('off')
    print('subplots made')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    print('subplots adjusted')
    outputs = []
    print('processing flows')
    for flow in flow_maps:
        print('new flow in loop')
        
        # Convert flow to color for visualization
        colored_flow = flow_vis_torch.flow_to_color(flow)
        grid = torchvision.utils.make_grid(colored_flow, nrow=n_rows)
        grid = grid.permute(1, 2, 0).detach().cpu().numpy()  # Convert to (H, W, C)
        print('colors done')
        
        # Update image data instead of creating a new plot
        ax.clear()
        ax.imshow(grid / 255.0)
        print('plots done')
        
        # Extract flow at the downsampled grid points
        u = flow[0, 0, y_grid, x_grid].detach().cpu().numpy()  # Flow in x direction
        v = flow[0, 1, y_grid, x_grid].detach().cpu().numpy()  # Flow in y direction
        
        # Plot arrows using fixed grid points
        ax.quiver(x_grid, y_grid, u, v, color='black', angles='xy', scale_units='xy', scale=1)
        print('more plots done')
        
        # Convert the Matplotlib figure to a PIL image using Agg renderer
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(canvas.get_width_height()[::-1] + (3,))
        outputs.append(Image.fromarray(image))
        print('draws done')
    
    print('done with flow')
    
    # Save the video (implementation not shown)
    # save_video(outputs, path, fps)

    plt.close(fig)  # Close the figure after processing all frames

    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the frames as a GIF
    try:
        outputs[0].save(
            path,
            save_all=True,
            append_images=outputs[1:],
            duration=int(1000 / fps),
            loop=0
        )
        print(f"GIF saved successfully at {path}")
    except Exception as e:
        print(f"An error occurred while saving the GIF: {e}")


    return outputs
    



def save_flow_grid_pillow_arrow(flow_maps: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    """
    Save the optical flow maps as a video grid augmented with arrows depicting the flow.
    
    Parameters:
    - flow_maps: Tensor of shape (b, t, c, h, w) containing optical flow maps.
    - path: Path to save the video.
    - rescale: Whether to rescale the values to [0, 1].
    - n_rows: Number of rows in the grid.
    - fps: Frames per second for the video.
    """
    # sample = flow.clone()
    # w = flow.shape[3]
    # h = flow.shape[2]
    # sample = (sample * 2 - 1).clamp(-1, 1)
    # #sample = sample * (1 - brush_mask.to(sample.device))
    # sample[:, 0:1, ...] = sample[:, 0:1, ...] * w
    # sample[:, 1:2, ...] = sample[:, 1:2, ...] * h

    # flow_pre = sample.squeeze(0)
    # flow_pre = rearrange(flow_pre, "c f h w -> f c h w")
    # # flow_pre = torch.cat(
    # # [torch.zeros(1, 2, h, w).to(flow_pre.device), flow_pre], dim=0
    # # )
 
    # flow_maps = []
    # flow_maps.append(flow_pre)
    print('starting ops')
    h = flow_maps[0].shape[2]
    w = flow_maps[0].shape[3]
    flow_maps = rearrange(flow_maps, "b t c h w -> t b c h w")
    #print('flow maps shape')
    #print(flow_maps.shape)
    outputs = []
    count = 0
    # Fixed grid points for arrows
    step = 30
    y_grid, x_grid = np.meshgrid(np.arange(0, h, step), np.arange(0, w, step), indexing='ij')
    

    print('processing flows')

    for flow in flow_maps:
        print('new flow in loop')
        # print(flow.shape)
        # Convert flow to color for visualization
        colored_flow = flow_vis_torch.flow_to_color(flow)
        grid = torchvision.utils.make_grid(colored_flow, nrow=n_rows)
        grid = grid.permute(1, 2, 0).detach().cpu().numpy()  # Convert to (H, W, C)

        print('colors done')

        # Plot arrows on the flow image
        fig, ax = plt.subplots(figsize=(grid.shape[1] / 100, grid.shape[0] / 100), dpi=100)
        ax.imshow(grid / 255.0)
        
        print('plots done')


        # Extract flow at the downsampled grid points
        u = flow[0, 0, y_grid, x_grid].detach().cpu().numpy()  # Flow in x direction
        v = flow[0, 1, y_grid, x_grid].detach().cpu().numpy()  # Flow in y direction
        
        # Plot arrows using fixed grid points
        ax.quiver(x_grid, y_grid, u, v, color='black', angles='xy', scale_units='xy', scale=1)

        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        print('more plots done')


        # Convert the Matplotlib figure to a PIL image
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        img = Image.fromarray(image)
        #img.save('flow' + str(count) + '.png')
        outputs.append(Image.fromarray(image))
        count += 1
        print('draws done')

        plt.close(fig)  # Close the figure to avoid memory issues
        print('done with flow')

    # print(f"Total frames: {len(outputs)}")
    # print(f"Output frame shape: {outputs[0].size if outputs else 'N/A'}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the frames as a GIF
    try:
        outputs[0].save(
            path,
            save_all=True,
            append_images=outputs[1:],
            duration=int(1000 / fps),
            loop=0
        )
        print(f"GIF saved successfully at {path}")
    except Exception as e:
        print(f"An error occurred while saving the GIF: {e}")


def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0):
        print("### " + s)

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        # print(x.shape)
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def save_flow_videos_grid(flow_maps: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    """
    Save the optical flow maps as a video grid.
    
    Parameters:
    - flow_maps: Tensor of shape (b, t, c, h, w) containing optical flow maps.
    - path: Path to save the video.
    - rescale: Whether to rescale the values to [0, 1].
    - n_rows: Number of rows in the grid.
    - fps: Frames per second for the video.
    """
    flow_maps = rearrange(flow_maps, "b t c h w -> t b c h w")
    outputs = []
    count = 0 
    for flow_map in flow_maps:
        count += 1
        flow_map = flow_vis_torch.flow_to_color(flow_map)
        
        # Convert to grid
        x = torchvision.utils.make_grid(flow_map, nrow=n_rows)
        #x = x.permute(1, 2, 0).numpy().astype(np.uint8)  # Ensure shape is (H, W, C) and type is uint8
        # print(flow_map.shape)
        x = flow_map[0].permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
        # Debug: Check each frame shape
        # print(f"Frame {count} shape: {x.shape}")
        
        outputs.append(x[0])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)
    
def save_flow_grid_pillow_arrow_alt(flow_maps: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    """
    Save the optical flow maps as a video grid augmented with arrows depicting the flow.
    
    Parameters:
    - flow_maps: Tensor of shape (b, t, c, h, w) containing optical flow maps.
    - path: Path to save the video.
    - rescale: Whether to rescale the values to [0, 1].
    - n_rows: Number of rows in the grid.
    - fps: Frames per second for the video.
    """
    flow_maps = rearrange(flow_maps, "b t c h w -> t b c h w")
    outputs = []
    count = 0
    
    for flow in flow_maps:
        # print(flow.shape)
        # Convert flow to color for visualization
        colored_flow = flow_vis_torch.flow_to_color(flow)
        grid = torchvision.utils.make_grid(colored_flow, nrow=n_rows)
        grid = grid.permute(1, 2, 0).detach().cpu().numpy()  # Convert to (H, W, C)

        # Plot arrows on the flow image
        fig, ax = plt.subplots(figsize=(grid.shape[1] / 100, grid.shape[0] / 100), dpi=100)
        ax.imshow(grid / 255.0)
        
        # Downsample the flow for quiver to avoid clutter (optional)
        step = 30
        for i in range(0, flow.shape[2], step):
            for j in range(0, flow.shape[3], step):
                # Extract the flow vector
                u = flow[0, 0, i, j].item()
                v = flow[0, 1, i, j].item()
                
                # Draw arrow
                ax.arrow(j, i, u, v, color='black', head_width=2, head_length=3, width=0.5)

        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Convert the Matplotlib figure to a PIL image
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        img = Image.fromarray(image)
        img.save('flow' + str(count) + '.png')
        outputs.append(Image.fromarray(image))
        count += 1
        plt.close(fig)  # Close the figure to avoid memory issues

    # print(f"Total frames: {len(outputs)}")
    # print(f"Output frame shape: {outputs[0].size if outputs else 'N/A'}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the frames as a GIF
    try:
        outputs[0].save(
            path,
            save_all=True,
            append_images=outputs[1:],
            duration=int(1000 / fps),
            loop=0
        )
        print(f"GIF saved successfully at {path}")
    except Exception as e:
        print(f"An error occurred while saving the GIF: {e}")

def save_flow_grid_pillow(flow_maps: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    """
    Save the optical flow maps as a video grid.
    
    Parameters:
    - flow_maps: Tensor of shape (b, t, c, h, w) containing optical flow maps.
    - path: Path to save the video.
    - rescale: Whether to rescale the values to [0, 1].
    - n_rows: Number of rows in the grid.
    - fps: Frames per second for the video.
    """
    flow_maps = rearrange(flow_maps, "b t c h w -> t b c h w")
    outputs = []
    
    for x in flow_maps:
        # print(x.shape)
        x = flow_vis_torch.flow_to_color(x)
        grid = torchvision.utils.make_grid(x, nrow=n_rows)
        grid = grid.permute(1, 2, 0).detach().cpu().numpy()  # Convert to (H, W, C)
        
        grid = (grid).astype(np.uint8)  # Convert to [0, 255]
        outputs.append(Image.fromarray(grid))

    # print(f"Total frames: {len(outputs)}")
    # print(f"Output frame shape: {outputs[0].size if outputs else 'N/A'}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the frames as a GIF
    try:
        outputs[0].save(
            path,
            save_all=True,
            append_images=outputs[1:],
            duration=int(1000 / fps),
            loop=0
        )
        print(f"GIF saved successfully at {path}")
    except Exception as e:
        print(f"An error occurred while saving the GIF: {e}")

# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = pipeline.text_encoder(
        uncond_input.input_ids.to(pipeline.device)
    )[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(
    model_output: Union[torch.FloatTensor, np.ndarray],
    timestep: int,
    sample: Union[torch.FloatTensor, np.ndarray],
    ddim_scheduler,
):
    timestep, next_timestep = (
        min(
            timestep
            - ddim_scheduler.config.num_train_timesteps
            // ddim_scheduler.num_inference_steps,
            999,
        ),
        timestep,
    )
    alpha_prod_t = (
        ddim_scheduler.alphas_cumprod[timestep]
        if timestep >= 0
        else ddim_scheduler.final_alpha_cumprod
    )
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (
        sample - beta_prod_t**0.5 * model_output
    ) / alpha_prod_t**0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(
        pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt
    )
    return ddim_latents


def load_weights(
    animation_pipeline,
    # motion module
    motion_module_path="",
    motion_module_lora_configs=[],
    # image layers
    dreambooth_model_path="",
    lora_model_path="",
    lora_alpha=0.8,
):
    # 1.1 motion module
    unet_state_dict = {}
    if motion_module_path != "":
        print(f"load motion module from {motion_module_path}")
        motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
        motion_module_state_dict = (
            motion_module_state_dict["state_dict"]
            if "state_dict" in motion_module_state_dict
            else motion_module_state_dict
        )
        unet_state_dict.update(
            {
                name: param
                for name, param in motion_module_state_dict.items()
                if "motion_modules." in name
            }
        )

    missing, unexpected = animation_pipeline.unet.load_state_dict(
        unet_state_dict, strict=False
    )
    assert len(unexpected) == 0
    del unet_state_dict

    if dreambooth_model_path != "":
        print(f"load dreambooth model from {dreambooth_model_path}")
        if dreambooth_model_path.endswith(".safetensors"):
            dreambooth_state_dict = {}
            with safe_open(dreambooth_model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    dreambooth_state_dict[key] = f.get_tensor(key)
        elif dreambooth_model_path.endswith(".ckpt"):
            dreambooth_state_dict = torch.load(
                dreambooth_model_path, map_location="cpu"
            )

        print("before vae")
        # 1. vae
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(
            dreambooth_state_dict, animation_pipeline.vae.config
        )
        print("!!!!!!!!!")
        animation_pipeline.vae.load_state_dict(converted_vae_checkpoint)
        print("before unet")
        # 2. unet
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(
            dreambooth_state_dict, animation_pipeline.unet.config
        )
        print("!!!!!!!!!")
        animation_pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
        print("before text")
        # 3. text_model
        animation_pipeline.text_encoder = convert_ldm_clip_checkpoint(
            dreambooth_state_dict
        )
        del dreambooth_state_dict

    if lora_model_path != "":
        print(f"load lora model from {lora_model_path}")
        assert lora_model_path.endswith(".safetensors")
        lora_state_dict = {}
        with safe_open(lora_model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                lora_state_dict[key] = f.get_tensor(key)

        animation_pipeline = convert_lora(
            animation_pipeline, lora_state_dict, alpha=lora_alpha
        )
        del lora_state_dict

    for motion_module_lora_config in motion_module_lora_configs:
        path, alpha = (
            motion_module_lora_config["path"],
            motion_module_lora_config["alpha"],
        )
        print(f"load motion LoRA from {path}")

        motion_lora_state_dict = torch.load(path, map_location="cpu")
        motion_lora_state_dict = (
            motion_lora_state_dict["state_dict"]
            if "state_dict" in motion_lora_state_dict
            else motion_lora_state_dict
        )

        animation_pipeline = convert_motion_lora_ckpt_to_diffusers(
            animation_pipeline, motion_lora_state_dict, alpha
        )

    return animation_pipeline
