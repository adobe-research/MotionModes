 #!/usr/bin/env python
# coding: utf-8

# In[1]:


import gradio as gr
import gc
import numpy as np
import cv2
from PIL import Image, ImageFilter
import uuid
from scipy.interpolate import interp1d, PchipInterpolator
# import torchvision
from flowgen.models.controlnet import ControlNetModel
from scripts.utils import *

import os
from omegaconf import OmegaConf

import torch

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from flowgen.models.unet3d import UNet3DConditionModel as UNet3DConditionModelFlow
from animation.models.forward_unet import UNet3DConditionModel

from flowgen.pipelines.pipeline_flow_gen_ours import FlowGenPipeline
from animation.pipelines.pipeline_animation import AnimationPipeline

from animation.utils.util import save_videos_grid, save_flow_grid_pillow_arrow
from animation.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob
import math
from pathlib import Path

import numpy as np

import torch.nn as nn

import matplotlib.pyplot as plt
import shutil
from datetime import datetime



# output_dir = "outputs"
# ensure_dirname(output_dir)

class ForwardWarp(nn.Module):
    """docstring for WarpLayer"""

    def __init__(
        self,
    ):
        super(ForwardWarp, self).__init__()

    def forward(self, img, flo):
        """
        -img: image (N, C, H, W)
        -flo: optical flow (N, 2, H, W)
        elements of flo is in [0, H] and [0, W] for dx, dy

        """

        # (x1, y1)		(x1, y2)
        # +---------------+
        # |				  |
        # |	o(x, y) 	  |
        # |				  |
        # |				  |
        # |				  |
        # |				  |
        # +---------------+
        # (x2, y1)		(x2, y2)

        N, C, _, _ = img.size()

        # translate start-point optical flow to end-point optical flow
        y = flo[:, 0:1:, :]
        x = flo[:, 1:2, :, :]

        x = x.repeat(1, C, 1, 1)
        y = y.repeat(1, C, 1, 1)

        # Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
        x1 = torch.floor(x)
        x2 = x1 + 1
        y1 = torch.floor(y)
        y2 = y1 + 1

        # firstly, get gaussian weights
        w11, w12, w21, w22 = self.get_gaussian_weights(x, y, x1, x2, y1, y2)

        # secondly, sample each weighted corner
        img11, o11 = self.sample_one(img, x1, y1, w11)
        img12, o12 = self.sample_one(img, x1, y2, w12)
        img21, o21 = self.sample_one(img, x2, y1, w21)
        img22, o22 = self.sample_one(img, x2, y2, w22)

        imgw = img11 + img12 + img21 + img22
        o = o11 + o12 + o21 + o22

        return imgw, o

    def get_gaussian_weights(self, x, y, x1, x2, y1, y2):
        w11 = torch.exp(-((x - x1) ** 2 + (y - y1) ** 2))
        w12 = torch.exp(-((x - x1) ** 2 + (y - y2) ** 2))
        w21 = torch.exp(-((x - x2) ** 2 + (y - y1) ** 2))
        w22 = torch.exp(-((x - x2) ** 2 + (y - y2) ** 2))

        return w11, w12, w21, w22

    def sample_one(self, img, shiftx, shifty, weight):
        """
        Input:
                -img (N, C, H, W)
                -shiftx, shifty (N, c, H, W)
        """

        N, C, H, W = img.size()

        # flatten all (all restored as Tensors)
        flat_shiftx = shiftx.view(-1)
        flat_shifty = shifty.view(-1)
        flat_basex = (
            torch.arange(0, H, requires_grad=False)
            .view(-1, 1)[None, None]
            .cuda()
            .long()
            .repeat(N, C, 1, W)
            .view(-1)
        )
        flat_basey = (
            torch.arange(0, W, requires_grad=False)
            .view(1, -1)[None, None]
            .cuda()
            .long()
            .repeat(N, C, H, 1)
            .view(-1)
        )
        flat_weight = weight.view(-1)
        flat_img = img.view(-1)

        # The corresponding positions in I1
        idxn = (
            torch.arange(0, N, requires_grad=False)
            .view(N, 1, 1, 1)
            .long()
            .cuda()
            .repeat(1, C, H, W)
            .view(-1)
        )
        idxc = (
            torch.arange(0, C, requires_grad=False)
            .view(1, C, 1, 1)
            .long()
            .cuda()
            .repeat(N, 1, H, W)
            .view(-1)
        )
        # ttype = flat_basex.type()
        idxx = flat_shiftx.long() + flat_basex
        idxy = flat_shifty.long() + flat_basey

        # recording the inside part the shifted
        mask = idxx.ge(0) & idxx.lt(H) & idxy.ge(0) & idxy.lt(W)

        # Mask off points out of boundaries
        ids = idxn * C * H * W + idxc * H * W + idxx * W + idxy
        ids_mask = torch.masked_select(ids, mask).clone().cuda()

        # (zero part - gt) -> difference
        # difference back propagate -> No influence! Whether we do need mask? mask?
        # put (add) them together
        # Note here! accmulate fla must be true for proper bp
        img_warp = torch.zeros(
            [
                N * C * H * W,
            ]
        ).cuda()
        img_warp.put_(
            ids_mask, torch.masked_select(flat_img * flat_weight, mask), accumulate=True
        )

        one_warp = torch.zeros(
            [
                N * C * H * W,
            ]
        ).cuda()
        one_warp.put_(ids_mask, torch.masked_select(flat_weight, mask), accumulate=True)

        return img_warp.view(N, C, H, W), one_warp.view(N, C, H, W)


def interpolate_trajectory(points, n_points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    t = np.linspace(0, 1, len(points))

    # fx = interp1d(t, x, kind='cubic')
    # fy = interp1d(t, y, kind='cubic')
    fx = PchipInterpolator(t, x)
    fy = PchipInterpolator(t, y)

    new_t = np.linspace(0, 1, n_points)

    new_x = fx(new_t)
    new_y = fy(new_t)
    new_points = list(zip(new_x, new_y))

    return new_points


def interpolate_trajectory_linear(points, n_points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    t = np.linspace(0, 1, len(points))

    fx = interp1d(t, x, kind='linear')
    fy = interp1d(t, y, kind='linear')

    new_t = np.linspace(0, 1, n_points)

    new_x = fx(new_t)
    new_y = fy(new_t)
    new_points = list(zip(new_x, new_y))

    return new_points

def visualize_drag_v2(background_image_path, brush_mask, splited_tracks, width, height):
    trajectory_maps = []

    background_image = Image.open(background_image_path).convert("RGBA")
    background_image = background_image.resize((width, height))
    w, h = background_image.size

    # Create a half-transparent background
    transparent_background = np.array(background_image)
    transparent_background[:, :, -1] = 128

    # Create a purple overlay layer
    purple_layer = np.zeros((h, w, 4), dtype=np.uint8)
    purple_layer[:, :, :3] = [128, 0, 128]  # Purple color
    purple_alpha = np.where(brush_mask < 0.5, 64, 0)  # Alpha values based on brush_mask
    purple_layer[:, :, 3] = purple_alpha

    # Convert to PIL image for alpha_composite
    purple_layer = Image.fromarray(purple_layer)
    transparent_background = Image.fromarray(transparent_background)

    # Blend the purple layer with the background
    transparent_background = Image.alpha_composite(transparent_background, purple_layer)

    # Create a transparent layer with the same size as the background image
    transparent_layer = np.zeros((h, w, 4))
    for splited_track in splited_tracks:
        if len(splited_track) > 1:
            splited_track = interpolate_trajectory(splited_track, 16)
            splited_track = splited_track[:16]
            for i in range(len(splited_track) - 1):
                start_point = (int(splited_track[i][0]), int(splited_track[i][1]))
                end_point = (int(splited_track[i + 1][0]), int(splited_track[i + 1][1]))
                vx = end_point[0] - start_point[0]
                vy = end_point[1] - start_point[1]
                arrow_length = np.sqrt(vx**2 + vy**2)
                if i == len(splited_track) - 2:
                    cv2.arrowedLine(
                        transparent_layer,
                        start_point,
                        end_point,
                        (255, 0, 0, 192),
                        2,
                        tipLength=8 / arrow_length,
                    )
                else:
                    cv2.line(
                        transparent_layer, start_point, end_point, (255, 0, 0, 192), 2
                    )
        else:
            cv2.circle(
                transparent_layer,
                (int(splited_track[0][0]), int(splited_track[0][1])),
                5,
                (255, 0, 0, 192),
                -1,
            )

    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
    trajectory_maps.append(trajectory_map)
    return trajectory_maps, transparent_layer


# Main Class for motion stuff

# In[2]:
def accumulate_warped_mask_vectorized(flow: torch.Tensor, brush_mask: torch.Tensor):
    """
    Accumulate the brush_mask across frames based on the optical flow.
    Automatically saves mask visualizations during the warping and accumulation steps.
    
    Parameters:
    - flow: Tensor of shape (b, c, f, h, w), where c = 2 (optical flow for u, v components).
    - brush_mask: Tensor of shape (1, 1, 1, h, w), representing the initial mask.
    
    Returns:
    - Accumulated brush_mask over all frames (shape: 1, 1, 1, h, w).
    """
    b, c, f, h, w = flow.shape  # Extract dimensions
    assert c == 2, "Flow tensor must have 2 channels (u, v) for horizontal and vertical components"

    brush_mask = 1 - brush_mask
    flow_copy = flow.clone()

    flow_copy[:, 1, :, :, :] = -flow[:, 1, :, :, :]
    flow_copy[:, 0, :, :, :] = -flow[:, 0, :, :, :]
    
    # Reshape the flow for parallel processing
    flow_copy = flow_copy.permute(0, 2, 3, 4, 1).reshape(b * f, h, w, 2)  # Shape (b*f, h, w, 2)
    
    # Generate a meshgrid for pixel coordinates (this will be the same for all frames)
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1).float().to(flow.device)  # Shape (h, w, 2)
    
    # Expand the grid for all frames
    grid = grid.unsqueeze(0).expand(b * f, h, w, 2)  # Shape (b*f, h, w, 2)

    # Apply the flow vectors to the grid coordinates (i.e., warp the grid for all frames)
    warped_grid = grid + flow_copy  # Shape (b*f, h, w, 2)
    
    # Normalize the grid for sampling with F.grid_sample
    warped_grid[..., 0] = (warped_grid[..., 0] / (w - 1)) * 2 - 1  # Normalize x
    warped_grid[..., 1] = (warped_grid[..., 1] / (h - 1)) * 2 - 1  # Normalize y
    
    # Expand brush_mask to apply across all frames
    brush_mask_expanded = brush_mask[0].expand(b * f, 1, h, w).to(flow.device)  # Shape (b*f, 1, h, w)

    warped_grid = warped_grid.to(flow.device)

    print(brush_mask_expanded.squeeze(1).shape)
    
    # Warp the brush_mask using grid_sample for all frames in parallel
    warped_brush_masks = F.grid_sample(brush_mask_expanded, 
                                       warped_grid, 
                                       mode='bilinear', 
                                       padding_mode='border', 
                                       align_corners=True)  # Shape (b*f, 1, h, w)
    
    # Save warped masks for each frame (in parallel processing, consider showing a representative frame)
    for frame_idx in range(f):
        save_mask_visualization(warped_brush_masks[frame_idx], step="warping", frame_idx=frame_idx)
    
    # Reshape back to (b, f, 1, h, w)
    warped_brush_masks = warped_brush_masks.view(b, f, 1, h, w)
    
    # Accumulate the warped masks by taking the union (max operation) over all frames
    accumulated_mask = torch.max(warped_brush_masks, dim=1, keepdim=True)[0]  # Shape (b, 1, 1, h, w)
    # Save the accumulated mask
    save_mask_visualization(accumulated_mask.squeeze(0), step="accumulation")

    accumulated_mask = torch.max(accumulated_mask, brush_mask.to(accumulated_mask.device))

    save_mask_visualization(accumulated_mask.squeeze(0), step="accumulation_plus")    

    return accumulated_mask

def sample_flow_points(flow_map, mask, num_points=100, threshold=0.01):
    """
    Sample flow points from areas where the mask is active.
    
    Args:
    flow_map (np.array): Flow map with shape (C, F, H, W)
    mask (np.array): Binary mask with shape (H, W)
    num_points (int): Number of points to sample
    threshold (float): Minimum magnitude of flow to consider
    
    Returns:
    List of sampled points
    """
    C, F, H, W = flow_map.shape
    
    # Compute flow magnitude
    flow_magnitude = np.linalg.norm(flow_map, axis=0)
    
    # Find the frame with the highest total flow magnitude
    max_flow_frame = np.argmax(flow_magnitude.sum(axis=(1, 2)))
    
    # Extract the flow for the frame with maximum flow
    max_frame_flow = flow_map[:, max_flow_frame, :, :]
    max_frame_magnitude = flow_magnitude[max_flow_frame]
    
    # Apply mask and threshold
    valid_points = (max_frame_magnitude > threshold) & (mask > 0)
    
    # Find coordinates of valid points
    valid_coords = np.column_stack(np.where(valid_points))
    
    # Sample points
    if len(valid_coords) > num_points:
        sampled_indices = np.random.choice(len(valid_coords), num_points, replace=False)
        sampled_coords = valid_coords[sampled_indices]
    else:
        sampled_coords = valid_coords
    
    # Create list of sampled points with their flow values
    sampled_points = []
    for y, x in sampled_coords:
        flow = max_frame_flow[:, y, x]
        start_point = (x, y)
        end_point = (x + flow[0], y + flow[1])
        sampled_points.append((start_point, end_point))
    
    return sampled_points



def sample_flow_points_traj(flow_map, mask, num_points=100, threshold=0.01, frames_to_sample=8):
    """
    Sample flow points from specified intermediate frames where the mask is active.

    Args:
    flow_map (np.array): Flow map with shape (C, F, H, W)
    mask (np.array): Binary mask with shape (H, W)
    num_points (int): Number of points to sample
    threshold (float): Minimum magnitude of flow to consider
    frames_to_sample (int): Number of intermediate frames to sample from

    Returns:
    List of dictionaries containing sampled points and their corresponding frames
    """
    C, F, H, W = flow_map.shape

    # Compute flow magnitude for all frames
    flow_magnitude = np.linalg.norm(flow_map, axis=0)  # Shape: (F, H, W)

    # Determine frames to sample from
    if frames_to_sample >= F:
        sampled_frames = np.arange(F)
    else:
        # Evenly spaced frames excluding the first frame (since flow at frame 0 is zero)
        sampled_frames = np.linspace(1, F - 1, frames_to_sample, dtype=int)

    # Prepare a list to collect all valid points across sampled frames
    all_valid_coords = []

    for frame_idx in sampled_frames:
        # Extract flow magnitude for the current frame
        frame_magnitude = flow_magnitude[frame_idx]  # Shape: (H, W)

        # Apply mask and threshold
        valid_points = (frame_magnitude > threshold) & (mask > 0)

        # Find coordinates of valid points
        valid_coords = np.column_stack(np.where(valid_points))

        # Append frame index to valid coordinates
        for coord in valid_coords:
            all_valid_coords.append((frame_idx, coord[0], coord[1]))

    # Sample points
    if len(all_valid_coords) > num_points:
        sampled_indices = np.random.choice(len(all_valid_coords), num_points, replace=False)
        sampled_coords = [all_valid_coords[idx] for idx in sampled_indices]
    else:
        sampled_coords = all_valid_coords

    # Create list of sampled points with their flow values
    sampled_points = []
    for frame_idx, y, x in sampled_coords:
        flow = flow_map[:, frame_idx, y, x]  # Shape: (C,)
        start_point = (x, y)
        sampled_points.append({
            'frame_idx': frame_idx,
            'start_point': start_point,
            'flow': flow
        })

    return sampled_points


def generate_control_tensor_sample(dense_flow_map, mask, model_length, flow_unit_id, vae_flow, control_ratio=1.0, threshold=0.1, frames_to_sample=5):
    """
    Generate a control tensor from a dense flow map for use in generation algorithms.

    Args:
    dense_flow_map (torch.Tensor): Dense flow map with shape (1, C, F, H, W)
    mask (np.array): Binary mask with shape (H, W)
    model_length (int): Number of frames in the output sparse flow map
    flow_unit_id (int): Size of the area around each point where flow is applied
    vae_flow: VAE model for encoding flow
    control_ratio (float): Ratio to determine the number of points to sample
    threshold (float): Minimum magnitude of flow to consider
    frames_to_sample (int): Number of intermediate frames to sample from

    Returns:
    torch.Tensor: Control tensor for use in generation algorithms
    """
    num_points = int(control_ratio * np.sum(1 - mask))
    print('Number of sample points:', num_points)

    # Convert dense flow map to numpy array for processing
    dense_flow_map_np = dense_flow_map.cpu().numpy()[0]  # Shape: (C, F, H, W)
    C, F, H, W = dense_flow_map_np.shape

    # Sample flow points from intermediate frames
    sampled_points = sample_flow_points_traj(dense_flow_map_np, mask, num_points, threshold, frames_to_sample)

    # Initialize output tensors
    input_drag = torch.zeros(1, model_length - 1, H, W, 2).to(dense_flow_map.device)  # Added batch dimension
    mask_drag = torch.zeros(1, model_length - 1, H, W, 1).to(dense_flow_map.device)  # Added batch dimension

    for point in sampled_points:
        frame_idx = point['frame_idx']
        start_point = point['start_point']
        flow = point['flow']

        # Since flows are already cumulative, we can directly use the flow at frame_idx
        cumulative_flow = flow  # Shape: (2,)
        end_point = (start_point[0] + cumulative_flow[0], start_point[1] + cumulative_flow[1])

        # Interpolate trajectory from start_point to end_point over model_length frames
        trajectory_points = [start_point, end_point]
        interpolated_track = interpolate_trajectory_linear(trajectory_points, model_length)

        for i in range(model_length - 1):
            # Compute cumulative flow from start_point to the point at frame i + 1
            next_point = interpolated_track[i + 1]
            flow_vector = (next_point[0] - start_point[0], next_point[1] - start_point[1])

            # Calculate flow area
            y_start = max(int(start_point[1]) - flow_unit_id, 0)
            y_end = min(int(start_point[1]) + flow_unit_id, H)
            x_start = max(int(start_point[0]) - flow_unit_id, 0)
            x_end = min(int(start_point[0]) + flow_unit_id, W)

            # Apply cumulative flow to the area
            input_drag[0, i, y_start:y_end, x_start:x_end, 0] = flow_vector[0]
            input_drag[0, i, y_start:y_end, x_start:x_end, 1] = flow_vector[1]
            mask_drag[0, i, y_start:y_end, x_start:x_end, 0] = 1  # Added channel dimension

    input_drag[..., 0] /= W
    input_drag[..., 1] /= H
    input_drag = (input_drag + 1) / 2  # Normalize to [0, 1]

    # Process drag and mask to create control tensor
    with torch.no_grad():
        b, l, h, w, c = input_drag.size()
        drag = rearrange(input_drag, "b l h w c -> b c l h w")
        mask = rearrange(mask_drag, "b l h w c -> b c l h w")

        sparse_flow = drag.to('cuda')
        sparse_mask = mask.to('cuda')

        # Adjust sparse flow values
        sparse_flow = (sparse_flow - 1/2) + 1/2  # Scale back to [-1, 1]

        # Encode flow using VAE
        flow_mask_latent = rearrange(
            vae_flow.encode(
                rearrange(sparse_flow, "b c f h w -> (b f) c h w")
            ).latent_dist.sample(),
            "(b f) c h w -> b c f h w",
            f=l,
        )

    print('Flow mask latent shape:', flow_mask_latent.shape)
    return flow_mask_latent


# def sample_flow_points(flow_map, num_points=1000, threshold=50.0):
#     """
#     Sample a set of start and end points from the frame with the highest magnitude of flow.
    
#     Args:
#     flow_map (torch.Tensor): Flow map with shape (C, F, H, W)
#     num_points (int): Number of points to sample
#     threshold (float): Minimum magnitude of flow to consider
    
#     Returns:
#     List of tuples, where each tuple contains (start_point, end_point)
#     """
#     C, F, H, W = flow_map.shape
#     assert C == 2, "Flow map should have 2 channels (dx, dy)"
    
#     # Compute flow magnitude for each frame
#     flow_magnitude = torch.norm(flow_map, dim=0)
    
#     # Find the frame with the highest total flow magnitude
#     max_flow_frame = torch.argmax(flow_magnitude.sum(dim=(1, 2)))
    
#     # Extract the flow for the frame with maximum flow
#     max_frame_flow = flow_map[:, max_flow_frame, :, :]
#     max_frame_magnitude = flow_magnitude[max_flow_frame]
    
#     # Find points with significant flow
#     significant_flow = (max_frame_magnitude > threshold).nonzero(as_tuple=False)
    
#     # Randomly sample from significant flow points
#     if len(significant_flow) > num_points:
#         indices = np.random.choice(len(significant_flow), num_points, replace=False)
#         sampled_points = significant_flow[indices]
#     else:
#         sampled_points = significant_flow
    
#     # Generate start and end points
#     tracks = []
#     for point in sampled_points:
#         y, x = point.tolist()
#         dx = max_frame_flow[0, y, x].item()
#         dy = max_frame_flow[1, y, x].item()
        
#         start_point = (x, y)
#         end_point = (x + dx, y + dy)
        
#         tracks.append((start_point, end_point))
    
#     return tracks

def generate_control_tensor(dense_flow_map, mask, model_length, flow_unit_id, vae_flow, control_ratio=1.0, threshold=0.1):
    """
    Generate a control tensor from a dense flow map for use in generation algorithms.
    
    Args:
    dense_flow_map (torch.Tensor): Dense flow map with shape (C, F, H, W)
    model_length (int): Number of frames in the output sparse flow map
    flow_unit_id (int): Size of the area around each point where flow is applied
    vae_flow: VAE model for encoding flow
    num_points (int): Number of points to sample
    threshold (float): Minimum magnitude of flow to consider
    
    Returns:
    torch.Tensor: Control tensor for use in generation algorithms
    """

    num_points = int(control_ratio*np.sum(1 - mask))

    print('num sample points')
    print(num_points)

    dense_flow_map = dense_flow_map[0]
    C, F, H, W = dense_flow_map.shape
    
    # Sample flow points
    sampled_tracks = sample_flow_points(dense_flow_map, mask, num_points, threshold)
    
    # Initialize output tensors
    input_drag = torch.zeros(1, model_length - 1, H, W, 2).to('cuda')  # Added batch dimension
    mask_drag = torch.zeros(1, model_length - 1, H, W, 1).to('cuda')  # Added batch dimension
    
    for track in sampled_tracks:
        start_point, end_point = track
        
        # Handle stationary points
        if start_point == end_point:
            end_point = (start_point[0] + 1, start_point[1] + 1)
        
        # Interpolate the track
        interpolated_track = interpolate_trajectory([start_point, end_point], model_length)
        
        # Ensure the track has the correct length
        if len(interpolated_track) < model_length:
            interpolated_track += [interpolated_track[-1]] * (model_length - len(interpolated_track))
        
        for i in range(model_length - 1):
            start_point = interpolated_track[0]
            end_point = interpolated_track[i + 1]
            
            # Calculate flow area
            y_start = max(int(start_point[1]) - flow_unit_id, 0)
            y_end = min(int(start_point[1]) + flow_unit_id, H)
            x_start = max(int(start_point[0]) - flow_unit_id, 0)
            x_end = min(int(start_point[0]) + flow_unit_id, W)
            
            # Apply flow to the area
            input_drag[0, i, y_start:y_end, x_start:x_end, 0] = end_point[0] - start_point[0]
            input_drag[0, i, y_start:y_end, x_start:x_end, 1] = end_point[1] - start_point[1]
            mask_drag[0, i, y_start:y_end, x_start:x_end] = 1
    

    #input_drag = dense_flow_map.unsqueeze(0).permute(0, 2, 3, 4, 1).to('cuda')
    visualization_tensor = input_drag.squeeze(0).permute(0, 3, 1, 2)
    test_flow_list = []
    test_flow_list.append(visualization_tensor)
    save_flow_grid_pillow_arrow(test_flow_list, 'debug_vis/test_control_flow.gif')
    
    # Normalize flow values
    input_drag[..., 0] /= W
    input_drag[..., 1] /= H

    input_drag = (input_drag + 1) / 2

    # Process drag and mask to create control tensor
    with torch.no_grad():
        b, l, h, w, c = input_drag.size()
        drag = rearrange(input_drag, "b l h w c -> b c l h w")
        mask = rearrange(mask_drag, "b l h w c -> b c l h w")
        
        sparse_flow = drag
        sparse_mask = mask
        
        sparse_flow = (sparse_flow - 1 / 2) + 1 / 2
        
        flow_mask_latent = rearrange(
            vae_flow.encode(
                rearrange(sparse_flow, "b c f h w -> (b f) c h w")
            ).latent_dist.sample(),
            "(b f) c h w -> b c f h w",
            f=l,
        )
    print('flow mask latent shape')
    print(flow_mask_latent.shape)
        #sparse_mask = sparse_mask.to(device='cuda', dtype = torch.float32)
        # sparse_mask = torch.nn.functional.interpolate(sparse_mask, scale_factor=(1, 1/8, 1/8))
        # control = torch.cat([flow_mask_latent, sparse_mask], dim=1)
        # control = control.to(device='cuda', dtype=torch.float32)
    
    return flow_mask_latent


def save_mask_visualization(mask: torch.Tensor, step: str, frame_idx: int = None):
    """
    Automatically saves the mask visualization at each step of the process.
    
    Parameters:
    - mask: Tensor of shape (1, h, w) or (b, 1, 1, h, w), representing the mask.
    - step: String to indicate the step (e.g., 'warping', 'accumulation').
    - frame_idx: The current frame index if it's during the warping step.
    """
    path = "mask_visualizations"
    os.makedirs(path, exist_ok=True)

    # Move the mask to CPU and convert to NumPy for visualization
    mask = mask.squeeze().detach().cpu().numpy()

    # If a frame index is provided, include it in the file name
    if frame_idx is not None:
        filename = f"{step}_frame_{frame_idx}.png"
    else:
        filename = f"{step}.png"

    # Plot and save the mask
    plt.imshow(mask, cmap='gray')
    plt.colorbar()
    plt.title(f"{step} {'Frame ' + str(frame_idx) if frame_idx is not None else ''}")
    plt.savefig(os.path.join(path, filename))
    plt.close()

    print(f"Saved mask at {step} step to {filename}")

class Drag:
    def __init__(
        self,
        device,
        pretrained_model_path,
        inference_config,
        height,
        width,
        model_length,
    ):
        self.device = device
        self.num_gen = 0

        self.inference_config = OmegaConf.load(inference_config)
        inference_config = self.inference_config
        ### >>> create validation pipeline >>> ###
        print("start loading")
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_path, subfolder="text_encoder"
        )
        # unet         = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))
        unet = UNet3DConditionModelFlow.from_config_2d(
            pretrained_model_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(
                inference_config.unet_additional_kwargs
            ),
        )
        vae_img = AutoencoderKL.from_pretrained(
            "models/stage2/StableDiffusion", subfolder="vae"
        )
        import json

        with open("./models/stage1/StableDiffusion-FlowGen/vae/config.json", "r") as f:
            vae_config = json.load(f)
        vae = AutoencoderKL.from_config(vae_config)
        vae_pretrained_path = (
            "models/stage1/StableDiffusion-FlowGen/vae_flow/diffusion_pytorch_model.bin"
        )
        print("[Load vae weights from {}]".format(vae_pretrained_path))
        processed_ckpt = {}
        weight = torch.load(vae_pretrained_path, map_location="cpu")
        vae.load_state_dict(weight, strict=True)
        controlnet = ControlNetModel.from_unet(unet)
        unet.controlnet = controlnet
        unet.control_scale = 1.0

        unet_pretrained_path = (
            "models/stage1/StableDiffusion-FlowGen/unet/diffusion_pytorch_model.bin"
        )
        print("[Load unet weights from {}]".format(unet_pretrained_path))
        weight = torch.load(unet_pretrained_path, map_location="cpu")
        m, u = unet.load_state_dict(weight, strict=False)

        controlnet_pretrained_path = (
            "models/stage1/StableDiffusion-FlowGen/controlnet/controlnet.bin"
        )
        print("[Load controlnet weights from {}]".format(controlnet_pretrained_path))
        weight = torch.load(controlnet_pretrained_path, map_location="cpu")
        m, u = unet.load_state_dict(weight, strict=False)

        print("finish loading")
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            print('no xformers!!')
            assert False
        print(f"Allocated memory: {torch.cuda.memory_allocated()} bytes")
        print(f"Cached memory: {torch.cuda.memory_reserved()} bytes")

        
        pipeline = FlowGenPipeline(
            vae_img=vae_img,
            vae_flow=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=DDIMScheduler(
                **OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
            ),
        )  # .to("cuda")
        pipeline = pipeline.to("cuda")

        del tokenizer, vae_img, vae, unet

        gc.collect()

        print(f"Allocated memory: {torch.cuda.memory_allocated()} bytes")
        print(f"Cached memory: {torch.cuda.memory_reserved()} bytes")

        
        self.pipeline = pipeline
        self.height = height
        self.width = width
        self.ouput_prefix = f"flow_debug"
        self.model_length = model_length


    def setup_animate_pipeline(self):
        ### >>> create validation pipeline >>> ###
        inference_config = self.inference_config
        inference_config['inference_batch_size'] = 1
        #inference_config = OmegaConf.load(inference_config)
        pretrained_model_path = "models/stage2/StableDiffusion"
        print("start loading")
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_path, subfolder="text_encoder"
        )
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(
                inference_config.unet_additional_kwargs
            ),
        )
        # 3. text_model
        motion_module_path = "models/stage2/Motion_Module/motion_block.bin"
        print("[Loading motion module ckpt from {}]".format(motion_module_path))
        weight = torch.load(motion_module_path, map_location="cpu")
        unet.load_state_dict(weight, strict=False)

        from safetensors import safe_open

        dreambooth_state_dict = {}
        with safe_open(
            "models/stage2/DreamBooth_LoRA/realisticVisionV51_v20Novae.safetensors",
            framework="pt",
            device="cpu",
        ) as f:
            for key in f.keys():
                dreambooth_state_dict[key] = f.get_tensor(key)

        from animation.utils.convert_from_ckpt import (
            convert_ldm_unet_checkpoint,
            convert_ldm_clip_checkpoint,
            convert_ldm_vae_checkpoint,
        )

        converted_vae_checkpoint = convert_ldm_vae_checkpoint(
            dreambooth_state_dict, vae.config
        )
        vae.load_state_dict(converted_vae_checkpoint)
        personalized_unet_path = "models/stage2/DreamBooth_LoRA/realistic_unet.ckpt"
        print("[Loading personalized unet ckpt from {}]".format(personalized_unet_path))
        unet.load_state_dict(torch.load(personalized_unet_path), strict=False)

        print("finish loading")
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            assert False
        pipeline = AnimationPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=DDIMScheduler(
                **OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
            ),
        )  # .to("cuda")
        pipeline = pipeline.to("cuda")

        self.animate_pipeline = pipeline        
    
    #@torch.no_grad()
    def forward_sample(
        self,
        input_drag,
        mask_drag,
        brush_mask,
        input_first_frame,
        prompt,
        n_prompt,
        inference_steps,
        guidance_scale,
        outputs=dict(),
        output_dir = None
    ):
        device = self.device

        num_samples = 3

        with torch.no_grad():
    
            b, l, h, w, c = input_drag.size()
            # drag = torch.cat([torch.zeros_like(drag[:, 0]).unsqueeze(1), drag], dim=1)  # pad the first frame with zero flow
            drag = rearrange(input_drag, "b l h w c -> b c l h w")
            mask = rearrange(mask_drag, "b l h w c -> b c l h w")
            brush_mask = rearrange(brush_mask, "b l h w c -> b c l h w")
            #zero_mask = torch.zeros_like(brush_mask)
    
            sparse_flow = drag
            sparse_mask = mask
    
            sparse_flow = (sparse_flow - 1 / 2) * (1 - sparse_mask)  + 1 / 2 # used to be sparse_flow into sparse_mask

            print('sparse mask and flow shapes')
            print(sparse_mask.shape)
            print(sparse_flow.shape)

            sparse_flow_vis = rearrange(sparse_flow, 'b c f h w -> b f c h w')

            #print(sparse_flow.shape)

            #sparse_flow_vis = sparse_flow_vis.expand_as(torch.zeros([sparse_flow_vis.shape[0], 14*sparse_flow_vis.shape[1], sparse_flow_vis.shape[2], sparse_flow_vis.shape[3]]))

            #save_flow_grid_pillow_arrow(sparse_flow_vis, "debug_vis/test_sparse_flow.gif")            
            
            flow_mask_latent = rearrange(
                self.pipeline.vae_flow.encode(
                    rearrange(sparse_flow, "b c f h w -> (b f) c h w")
                ).latent_dist.sample(),
                "(b f) c h w -> b c f h w",
                f=l,
            )
            # flow_mask_latent = vae.encode(sparse_flow).latent_dist.sample()*0.18215
            sparse_mask = F.interpolate(sparse_mask, scale_factor=(1, 1 / 8, 1 / 8))
            control = torch.cat([flow_mask_latent, sparse_mask], dim=1)
            
            # print(drag)
            stride = list(range(8, 121, 8))

        print('control shape')
        print(control.shape)

        print('brush mask shape')
        print(brush_mask.shape)

        print(f"Allocated memory: {torch.cuda.memory_allocated()} bytes")
        print(f"Cached memory: {torch.cuda.memory_reserved()} bytes")
        torch.cuda.empty_cache()
        print(f"Allocated memory: {torch.cuda.memory_allocated()} bytes")
        print(f"Cached memory: {torch.cuda.memory_reserved()} bytes")
        sample, coarse_flow_latent = self.pipeline(
            prompt,
            first_frame=input_first_frame.squeeze(0),
            control=control,
            guidance = True,
            particle_guidance = True,
            controlnet_switch=False,
            brush_mask = brush_mask,
            stride=torch.tensor([stride]).cuda(),
            negative_prompt=n_prompt,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            width=w,
            height=h,
            video_length=len(stride),
            output_dir = output_dir
        )
        sample = sample.videos
        sample = (sample * 2 - 1).clamp(-1, 1)
        sample[:, 0:1, ...] = sample[:, 0:1, ...] * w
        sample[:, 1:2, ...] = sample[:, 1:2, ...] * h

        # coarse_flow_pre = sample.squeeze(0)

        # # flow_mask_latent = generate_control_tensor(sample, 1 - brush_mask.squeeze().detach().cpu().numpy(), self.model_length, 4, self.pipeline.vae_flow)

        # flow_mask_latent = generate_control_tensor_sample(sample, 1 - brush_mask.squeeze().detach().cpu().numpy(), self.model_length, 4, self.pipeline.vae_flow)


        # warped_brush_mask = accumulate_warped_mask_vectorized(sample, brush_mask)

        # print(sparse_mask.shape)

        # # sparse_mask = sparse_mask.expand_as(brush_mask).to('cuda')

        # warped_sparse_mask = F.interpolate(warped_brush_mask, scale_factor=(1, 1 / 8, 1 / 8))

        # warped_sparse_mask = 1 - warped_sparse_mask.expand_as(sparse_mask).to('cuda')

        # # print(sparse_mask.shape)

        # # sparse_mask = sparse_mask.expand_as(brush_mask).to('cuda')


        # refined_control = torch.cat([flow_mask_latent, sparse_mask], dim=1)
        
        # #refined_control = torch.cat([coarse_flow_latent, sparse_mask], dim=1)

        # sample, refined_flow_latent = self.pipeline(
        #     prompt,
        #     first_frame=input_first_frame.squeeze(0),
        #     control=refined_control,
        #     guidance=False,
        #     particle_guidance=False,
        #     controlnet_switch=True,
        #     brush_mask = brush_mask,
        #     stride=torch.tensor([stride]).cuda(),
        #     negative_prompt=n_prompt,
        #     num_inference_steps=inference_steps,
        #     guidance_scale=guidance_scale,
        #     width=w,
        #     height=h,
        #     video_length=len(stride),
        #     output_dir = output_dir
        # )
        # sample = sample.videos
        
        # refined_control = torch.cat([refined_flow_latent, sparse_mask], dim=1)


        # sample, refined_flow_latent = self.pipeline(
        #     prompt,
        #     first_frame=input_first_frame.squeeze(0),
        #     control=refined_control,
        #     guidance=False,
        #     particle_guidance=False,
        #     brush_mask = brush_mask,
        #     stride=torch.tensor([stride]).cuda(),
        #     negative_prompt=n_prompt,
        #     num_inference_steps=inference_steps,
        #     guidance_scale=guidance_scale,
        #     width=w,
        #     height=h,
        #     video_length=len(stride),
        # )
        # sample = sample.videos
 

        #sample typically has shape b c f h w where b is 1

        #brush mask has shape 1 1 1 h w

        #change brush mask to dilate according to the generated flow
        print('sample shape before clamps etc is')
        print(sample.shape)
        zero_mask = torch.zeros_like(brush_mask)
        # sample = (sample * 2 - 1).clamp(-1, 1)
        # sample[:, 0:1, ...] = sample[:, 0:1, ...] * w
        # sample[:, 1:2, ...] = sample[:, 1:2, ...] * h
        #sample = sample * (1 - brush_mask.to(sample.device))
        
        # brush_mask = torch.zeros_like(brush_mask) #accumulate_warped_mask_vectorized(sample, brush_mask)

        # print('shape of brush mask is')
        # print(brush_mask.shape)

        #sample = sample * (brush_mask.to(sample.device))

        #brush_mask = 1 - brush_mask
        
        # sample = (sample * 2 - 1).clamp(-1, 1)
        # sample = sample * (1 - brush_mask.to(sample.device))
        # sample[:, 0:1, ...] = sample[:, 0:1, ...] * w
        # sample[:, 1:2, ...] = sample[:, 1:2, ...] * h

        flow_pre = sample.squeeze(0)
        # print('flow_pre shape')
        # print(flow_pre.shape)        

        # test_flow = rearrange(flow_pre, "c f h w -> f c h w")
        # test_flow = torch.cat(
        #     [torch.zeros(1, 2, h, w).to(test_flow.device), test_flow], dim=0
        # )        
        # test_flow_list = []
        # test_flow_list.append(test_flow)
        # save_flow_grid_pillow_arrow(test_flow_list, "debug_vis/test_flow_pre_animate.gif")

        self.setup_animate_pipeline()
        flow_list = []
        video_list = []
        with torch.no_grad():
            if(len(flow_pre.shape) == 5):
                for flow_pred_idx in range(flow_pre.shape[0]):
                    flow_pre_curr = flow_pre[flow_pred_idx]
                    flow_pre_curr = rearrange(flow_pre_curr, "c f h w -> f c h w")
                    flow_pre_curr = torch.cat(
                    [torch.zeros(1, 2, h, w).to(flow_pre_curr.device), flow_pre_curr], dim=0
                    )
                    input_first_frame = input_first_frame[0].unsqueeze(0)
                    zero_mask = zero_mask[0].unsqueeze(0)
                    sample = self.animate_pipeline(
                        prompt,
                        first_frame=input_first_frame.squeeze(0) * 2 - 1,
                        flow_pre=flow_pre_curr,
                        brush_mask=zero_mask,
                        negative_prompt=n_prompt,
                        num_inference_steps=inference_steps,
                        guidance_scale=guidance_scale,
                        width=w,
                        height=h,
                        video_length=self.model_length,
                    ).videos
                    video_list.append(sample[0])
                    flow_list.append(flow_pre_curr)
            else:
                flow_pre = rearrange(flow_pre, "c f h w -> f c h w")
                flow_pre = torch.cat(
                    [torch.zeros(1, 2, h, w).to(flow_pre.device), flow_pre], dim=0
                )
                for samp in range(num_samples):
                    sample = self.animate_pipeline(
                        prompt,
                        first_frame=input_first_frame.squeeze(0) * 2 - 1,
                        flow_pre=flow_pre,
                        brush_mask=zero_mask,
                        negative_prompt=n_prompt,
                        num_inference_steps=inference_steps,
                        guidance_scale=guidance_scale,
                        width=w,
                        height=h,
                        video_length=self.model_length,
                    ).videos
                    video_list.append(sample[0])
                    flow_list.append(flow_pre)
                # coarse_flow_pre = rearrange(coarse_flow_pre, "c f h w -> f c h w")
                # coarse_flow_pre = torch.cat(
                #     [torch.zeros(1, 2, h, w).to(coarse_flow_pre.device), coarse_flow_pre], dim=0
                # )
                # for samp in range(num_samples):
                #     sample = self.animate_pipeline(
                #         prompt,
                #         first_frame=input_first_frame.squeeze(0) * 2 - 1,
                #         flow_pre=coarse_flow_pre,
                #         brush_mask=zero_mask,
                #         negative_prompt=n_prompt,
                #         num_inference_steps=inference_steps,
                #         guidance_scale=guidance_scale,
                #         width=w,
                #         height=h,
                #         video_length=self.model_length,
                #     ).videos
                #     video_list.append(sample[0])
                #     flow_list.append(coarse_flow_pre)

        if(len(video_list) != 0):
            self.num_gen += 1
            motion_path = os.path.join(output_dir, 'generated_motions')
            num_files = len([f for f in os.listdir(motion_path) if os.path.isfile(os.path.join(motion_path, f))])
            save_videos_grid(video_list, os.path.join(output_dir, "gradio/samples/test_out_" +str(num_files) + ".gif"))
            # save_flow_grid_pillow_arrow(flow_list, "gradio/samples/test_flow_out_" +str(self.num_gen) + ".gif")
            
        return sample

    def run(
        self,
        first_frame_path,
        brush_mask,
        tracking_points,
        inference_batch_size,
        flow_unit_id,
        prompt,
        output_dir = None
    ):
        original_width, original_height = 512, 320

        #brush_mask = image_brush["mask"]

        brush_mask = (
            cv2.resize(
                brush_mask[:, :, 0],
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)
            / 255.0
        )

        # Define the kernel (structuring element) for dilation
        # The size of the kernel defines how much dilation is applied
        kernel_size = 5  # You can adjust this size based on how much dilation you want
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # # Apply dilation to the brush mask
        dilated_brush_mask = cv2.dilate(brush_mask, kernel, iterations=1)

        

        brush_mask_bool = brush_mask > 0.5
        brush_mask[brush_mask_bool], brush_mask[~brush_mask_bool] = 0, 1

        brush_mask = torch.from_numpy(brush_mask)

        #zero_mask = torch.zeros_like(brush_mask)

        brush_mask = (
            torch.zeros_like(brush_mask) if (brush_mask == 1).all() else brush_mask
        )

        brush_mask = brush_mask.unsqueeze(0).unsqueeze(3)
        zero_mask = torch.zeros_like(brush_mask)


        dilated_brush_mask_bool = dilated_brush_mask > 0.5
        dilated_brush_mask[dilated_brush_mask_bool], dilated_brush_mask[~dilated_brush_mask_bool] = 0, 1

        dilated_brush_mask = torch.from_numpy(dilated_brush_mask)

        #zero_mask = torch.zeros_like(brush_mask)

        dilated_brush_mask = (
            torch.zeros_like(dilated_brush_mask) if (dilated_brush_mask == 1).all() else dilated_brush_mask
        )

        dilated_brush_mask = dilated_brush_mask.unsqueeze(0).unsqueeze(3)


        #when everything is included - the mask is 0 everywhere here.

        print(brush_mask)
        print('brush mask shape')
        print(brush_mask.shape)

        input_all_points = tracking_points#.constructor_args["value"]
        print('input_all_points')
        print(input_all_points)
        resized_all_points = [
            tuple(
                [
                    tuple(
                        [
                            int(e1[0] * self.width / original_width),
                            int(e1[1] * self.height / original_height),
                        ]
                    )
                    for e1 in e
                ]
            )
            for e in input_all_points
        ]

        print(resized_all_points)

        input_drag = torch.zeros(self.model_length - 1, self.height, self.width, 2)
        mask_drag = torch.zeros(self.model_length - 1, self.height, self.width, 1)
        for splited_track in resized_all_points:
            if len(splited_track) == 1:  # stationary point
                displacement_point = tuple(
                    [splited_track[0][0] + 1, splited_track[0][1] + 1]
                )
                splited_track = tuple([splited_track[0], displacement_point])
            # interpolate the track
            splited_track = interpolate_trajectory(splited_track, self.model_length)
            splited_track = splited_track[: self.model_length]
            if len(splited_track) < self.model_length:
                splited_track = splited_track + [splited_track[-1]] * (
                    self.model_length - len(splited_track)
                )
            for i in range(self.model_length - 1):
                start_point = splited_track[0]
                end_point = splited_track[i + 1]
                input_drag[
                    i,
                    max(int(start_point[1]) - flow_unit_id, 0) : int(start_point[1])
                    + flow_unit_id,
                    max(int(start_point[0]) - flow_unit_id, 0) : int(
                        start_point[0] + flow_unit_id
                    ),
                    0,
                ] = (
                    end_point[0] - start_point[0]
                )
                input_drag[
                    i,
                    max(int(start_point[1]) - flow_unit_id, 0) : int(start_point[1])
                    + flow_unit_id,
                    max(int(start_point[0]) - flow_unit_id, 0) : int(
                        start_point[0] + flow_unit_id
                    ),
                    1,
                ] = (
                    end_point[1] - start_point[1]
                )
                mask_drag[
                    i,
                    max(int(start_point[1]) - flow_unit_id, 0) : int(start_point[1])
                    + flow_unit_id,
                    max(int(start_point[0]) - flow_unit_id, 0) : int(
                        start_point[0] + flow_unit_id
                    ),
                ] = 1

        input_drag[..., 0] /= self.width
        input_drag[..., 1] /= self.height

        input_drag = input_drag * (1 - dilated_brush_mask)
        print('input_drag')
        print(input_drag.max())
        mask_drag = torch.where(dilated_brush_mask.expand_as(mask_drag) > 0, 1, mask_drag)

        # input_drag = input_drag * (1 - zero_mask)
        # mask_drag = torch.where(zero_mask.expand_as(mask_drag) > 0, 1, mask_drag)

        
        input_drag = (input_drag + 1) / 2
        dir, base, ext = split_filename(first_frame_path)
        id = base.split("_")[-1]

        image_pil = image2pil(first_frame_path)
        image_pil = image_pil.resize((self.width, self.height), Image.BILINEAR).convert(
            "RGB"
        )

        visualized_drag, _ = visualize_drag_v2(
            first_frame_path,
            zero_mask.squeeze(3).squeeze(0).cpu().numpy(),
            resized_all_points,
            self.width,
            self.height,
        )

        first_frames_transform = transforms.Compose(
            [
                lambda x: Image.fromarray(x),
                transforms.ToTensor(),
            ]
        )

        outputs = None
        ouput_video_list = []
        num_inference = 1
        for i in tqdm(range(num_inference)):
            if not outputs:

                first_frames = image2arr(first_frame_path)
                Image.fromarray(first_frames).save("./temp.png")
                first_frames = repeat(
                    first_frames_transform(first_frames),
                    "c h w -> b c h w",
                    b=inference_batch_size,
                ).to(self.device)
                print('first frames shape')
                print(first_frames.shape)
            else:
                first_frames = outputs[:, -1]


            print(input_drag.shape)
            print(mask_drag.shape)
            print(brush_mask.shape)
            print(first_frames.shape)
            

            outputs = self.forward_sample(
                repeat(
                    input_drag[
                        i * (self.model_length - 1) : (i + 1) * (self.model_length - 1)
                    ],
                    "l h w c -> b l h w c",
                    b=inference_batch_size,
                ).to(self.device),
                repeat(
                    mask_drag[
                        i * (self.model_length - 1) : (i + 1) * (self.model_length - 1)
                    ],
                    "l h w c -> b l h w c",
                    b=inference_batch_size,
                ).to(self.device),
                repeat(
                    brush_mask[
                        i * (self.model_length - 1) : (i + 1) * (self.model_length - 1)
                    ],
                    "l h w c -> b l h w c",
                    b=inference_batch_size,
                ).to(self.device),
                first_frames,
                prompt,
                "(blur, haze, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation",
                25,
                7,
                output_dir = output_dir
            )
            ouput_video_list.append(outputs)

        outputs_path = f"gradio/samples/output_{id}.gif"
        save_videos_grid(outputs, outputs_path)
        del self.animate_pipeline

        return visualized_drag[0], outputs_path

def preprocess_image(image_pil):
    raw_w, raw_h = image_pil.size
    resize_ratio = max(512 / raw_w, 320 / raw_h)
    image_pil = image_pil.resize(
        (int(raw_w * resize_ratio), int(raw_h * resize_ratio)), Image.BILINEAR
    )
    image_pil = transforms.CenterCrop((320, 512))(image_pil.convert('RGB'))
    return image_pil



def process_images(frame_path, mask_path, prompt, output_dir):
    first_frame = Image.open(frame_path)
    mask = Image.open(mask_path)

    first_frame = preprocess_image(first_frame)
    mask = preprocess_image(mask)

    print(f'Images processed for: {frame_path} with prompt: "{prompt}"')

    drag_nuwa = Drag(
        "cuda:0",
        "models/stage1/StableDiffusion-FlowGen",
        "configs/configs_flowgen/inference/inference.yaml",
        320,
        512,
        16,
    )

    print('Model initialized, running...')

    outputs = drag_nuwa.run(
        frame_path,
        np.array(mask),
        [],
        1,
        4,
        prompt,
        output_dir = output_dir
    )
    
    # Save the outputs to the specified directory
    # output_path = os.path.join(output_dir, f"{os.path.basename(frame_path).split('.')[0]}_output.png")
    # outputs.save(output_path)  # Assuming the outputs object has a save method or adapt as necessary
    # print(f"Output saved at: {output_path}")

    return outputs

def run_for_multiple_images(json_file, output_base_dir, num_generations = 1):
    with open(json_file, 'r') as f:
        image_data = json.load(f)
    
    # # Create output directory if it doesn't exist
    # os.makedirs(output_base_dir, exist_ok=True)
    
    # Create a unique output directory based on timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for item in image_data:
        frame_path = item['frame_path']
        mask_path = item['mask_path']
        prompt = item['prompt']

        unique_output_dir = os.path.join(output_base_dir, os.path.basename(frame_path).split('.')[0])
        os.makedirs(unique_output_dir, exist_ok=True)


        # Create a unique output sub-directory for each entry if needed
        output_dir = os.path.join(unique_output_dir, f"run_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        for gen in range(num_generations):
            outputs = process_images(frame_path, mask_path, prompt, output_dir)
        
        print(f"Processing completed for: {frame_path}")


print('starting main')

num_gens = 8

run_for_multiple_images('frame_data.json', 'results/motion_modes/outputs/', num_gens)

# frame_path = 'first_frames/frame_0001_processed.jpg'
# mask_path = 'first_frames/frame_0001_mask.png'

# frame_path = 'first_frames/room_scene_cropped_processed.jpg'
# mask_path = 'first_frames/room_scene_cropped_mask.png'

# frame_path = 'first_frames/room_scene_processed.jpg'
# mask_path = 'first_frames/room_scene_mask.png'

# frame_path = 'first_frames/butterfly_processed.jpg'
# mask_path = 'first_frames/butterfly_mask.png'

# frame_path = 'first_frames/river2_processed.jpg'
# mask_path = 'first_frames/river2_mask.png'

# frame_path = 'first_frames/ball_air_processed.jpg'
# mask_path = 'first_frames/ball_air_mask.png'

# frame_path = 'first_frames/table_scene_2_processed.jpg'
# mask_path = 'first_frames/table_scene_2_mask2.png'

# frame_path = 'first_frames/boat_river_processed.jpg'
# mask_path = 'first_frames/boat_river_mask.png'

# frame_path = 'first_frames/basketball_processed.jpg'
# mask_path = 'first_frames/basketball_mask.png'

# frame_path = 'first_frames/basketball_basket_processed.jpg'
# mask_path = 'first_frames/basketball_basket_mask.png'

# frame_path = 'first_frames/flag_processed.jpg'
# mask_path = 'first_frames/flag_mask.png'

# frame_path = 'first_frames/table_laptop_processed.jpg'
# mask_path = 'first_frames/table_laptop_mask.png'


# first_frame = Image.open(frame_path)
# mask = Image.open(mask_path)

# first_frame = preprocess_image(first_frame)
# # Resize the image to the desired width and height
# mask = preprocess_image(mask)

# print('images processed starting i2v')


# drag_nuwa = Drag(
#     "cuda:0",
#     "models/stage1/StableDiffusion-FlowGen",
#     "configs/configs_flowgen/inference/inference.yaml",
#     320,
#     512,
#     16,
# )

# print('init done - running')

# outputs = drag_nuwa.run(
#             frame_path,
#             np.array(mask),
#             [],
#             1,
#             4,
#             "video of a butterfly",
#         )

# outputs[0].save('drag.png')
#"a boat slowly moves in a video shot with an extremely stable and fixed camera, no camera motion stable shot"
#"a butterfly slowly fans its wings in a video shot with an extremely stable and fixed camera, no camera motion stable shot"
#"a laptop lid slowly closes down in a video shot with an extremely stable and fixed camera, no camera motion stable shot"
#"a drawer is opened in a video shot with an extremely stable and fixed camera, stable shot no camera movement"

