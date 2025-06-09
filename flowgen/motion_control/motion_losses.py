import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
import flow_vis_torch
from PIL import Image
import os
from einops import rearrange, repeat
import torchvision
import math

import os 
import imageio
import numpy as np
from typing import Union
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Agg')

import torchvision
import torch.distributed as dist
import flow_vis_torch

from safetensors import safe_open
from tqdm import tqdm

def estimate_focal_length(h, w, fov_deg=60):
    """
    Estimate the focal length (in pixels) given the image height, width, and assumed FOV.
    
    :param h: Height of the image in pixels.
    :param w: Width of the image in pixels.
    :param fov_deg: Assumed Field of View in degrees (common values: 60°, 90°).
    :return: Estimated focal length in pixels.
    """
    fov_rad = math.radians(fov_deg)
    f = w / (2 * math.tan(fov_rad / 2))
    return f

class RobustRotationEstimatorTorch(nn.Module):
    def __init__(self, h: int, w: int, f: float, bin_size: float, max_angle: float, spatial_step: int):
        super(RobustRotationEstimatorTorch, self).__init__()
        
        self.h = h
        self.w = w
        self.f = f
        self.bin_size = bin_size
        self.max_angle = max_angle
        self.spatial_step = spatial_step

        self.center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device = 'cuda')
        self.shift = 0
        self.n_points = 2

        self.middle_bins = (torch.arange(-self.max_angle, self.max_angle + self.bin_size, self.bin_size) + self.shift).to('cuda')
        self.start_bins = self.middle_bins - self.bin_size / 2
        self.n_bins_per_dim = len(self.middle_bins)

        self.flow_locations = self._get_flow_vector_locations().to('cuda')

        # Precompute lines for rotation estimation
        self.lines = self._precompute_lines_vectorized().to('cuda')

    def _get_flow_vector_locations(self):
        r = torch.arange(-(self.h - 1) // 2, (self.h - 1) // 2 + 1, dtype=torch.float32)
        c = torch.arange(-(self.w - 1) // 2, (self.w - 1) // 2 + 1, dtype=torch.float32)
        out = torch.zeros((2, self.h, self.w), dtype=torch.float32)
        # Broadcast r across the width dimension
        out[0, :, :] = r.view(-1, 1).expand(-1, self.w)  # Reshape r to (h, 1) and expand to (h, w)
        out[1, :, :] = c.view(1, -1).expand(self.h, -1)  # Reshape c to (1, w) and expand to (h, w)
        out = out[:, :: self.spatial_step, :: self.spatial_step]
        return out + 0.5

    def _precompute_lines_vectorized(self):
        n_rotations = self.n_bins_per_dim * self.n_points

        eu = torch.stack(
            (
                (self.flow_locations[0] * self.flow_locations[1]) / self.f,
                -(self.f + self.flow_locations[0] ** 2 / self.f),
                self.flow_locations[1],
            ),
            dim=-1,
        )

        ev = torch.stack(
            (
                (self.f + self.flow_locations[1] ** 2 / self.f),
                -(self.flow_locations[0] * self.flow_locations[1]) / self.f,
                -self.flow_locations[0],
            ),
            dim=-1,
        )

        v_s = torch.cross(eu, ev)

        v_s = v_s / torch.norm(v_s, dim=-1, keepdim=True)

        points = torch.linspace(
            0, 1, n_rotations, device=v_s.device
        ).view(1, 1, -1, 1) * (self.middle_bins[-1] - self.middle_bins[0]) + self.middle_bins[0]

        points = v_s[..., None, :] / v_s[..., 2:3, None] * points

        return points.permute(1, 2, 0, 3)

    def _rad_to_idx(self, x):
        return torch.floor((x + self.max_angle + self.bin_size / 2) / self.bin_size - self.shift).long()

    def _idx_to_rad(self, x):
        return self.middle_bins[x]

    def estimate(self, flow):
        # Ensure the flow tensor is on the correct device
        #flow = flow.to(self.device)
    
        # Extract the sampled flow locations
        indices_x, indices_y = self.flow_locations
    
        # Ensure that the sampled flow has the correct dimensions after spatial sampling
        u = flow[0, ::self.spatial_step, ::self.spatial_step]
        v = flow[1, ::self.spatial_step, ::self.spatial_step]
    
        # Compute a and b (both of shape: (h // spatial_step, w // spatial_step))
        a = (self.f**2 * v - u * indices_x * indices_y + v * indices_x**2) / (
            self.f**3 + self.f * indices_x**2 + self.f * indices_y**2
        )
        b = -(self.f**2 * u + u * indices_y**2 - v * indices_x * indices_y) / (
            self.f**3 + self.f * indices_x**2 + self.f * indices_y**2
        )
    
        # Point should now have the same shape as the first two dimensions of self.lines
        point = torch.stack([a, b, torch.zeros_like(a)], dim=-1)  # (h // spatial_step, w // spatial_step, 3)
        point = point[:, :, None, :]  # (h // spatial_step, w // spatial_step, 1, 3)


        # Ensure self.lines and point have compatible shapes for addition
        # lines_c should have the shape: (h // spatial_step, w // spatial_step, n_rot, 3)
        # Permute point to match the shape of self.lines
        point = point.permute(1, 2, 0, 3)  # Changing the order to [35, 22, 1, 3]

        print(self.lines.shape)
        print(point.shape)
        
        
        
        lines_c = self.lines + point  # Adding the points to the precomputed lines
    
        # Flatten spatial dimensions
        votes = lines_c.reshape(-1, 3)  # (N, 3)
    
        # Convert radians to indices
        votes = self._rad_to_idx(votes)
    
        # Mask for valid indices
        mask = (votes >= 0) & (votes < self.n_bins_per_dim)
        mask = mask.all(dim=1)
        votes = votes[mask]
    
        # Flatten the valid indices
        votes_flat = votes[:, 0] * self.n_bins_per_dim**2 + votes[:, 1] * self.n_bins_per_dim + votes[:, 2]
    
        # Create histogram and find the most frequent bin
        hist = torch.histc(votes_flat.float(), bins=self.n_bins_per_dim**3, min=0, max=self.n_bins_per_dim**3 - 1)
    
        max_idx = torch.argmax(hist)
    
        # Convert the bin index back to radians
        indexes = torch.stack([
            max_idx // (self.n_bins_per_dim**2),
            (max_idx % (self.n_bins_per_dim**2)) // self.n_bins_per_dim,
            max_idx % self.n_bins_per_dim
        ]).long()
    
        # Convert the index to the corresponding rotation
        pred_rot = self._idx_to_rad(indexes)
    
        return pred_rot



def save_flow_grid_pillow_arrow(flow: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    """
    Save the optical flow maps as a video grid augmented with arrows depicting the flow.
    
    Parameters:
    - flow: Tensor of shape (b, c, f, h, w) containing optical flow maps.
    - path: Path to save the video.
    - rescale: Whether to rescale the values to [0, 1].
    - n_rows: Number of rows in the grid.
    - fps: Frames per second for the video.
    """
    sample = flow.clone()
    w = flow.shape[3]
    h = flow.shape[2]
    #print('flow sample shape before clamps')
    #print(sample.shape)
    sample = sample.unsqueeze(0)
    sample = (sample * 2 - 1).clamp(-1, 1)
    #sample = sample * (1 - brush_mask.to(sample.device))
    sample[:, 0:1, ...] = sample[:, 0:1, ...] * w
    sample[:, 1:2, ...] = sample[:, 1:2, ...] * h

    flow_pre = sample.squeeze(0)
    flow_pre = rearrange(flow_pre, "c f h w -> f c h w")
    flow_pre = torch.cat(
    [torch.zeros(1, 2, h, w).to(flow_pre.device), flow_pre], dim=0
    )
 
    flow_maps = []
    flow_maps.append(flow_pre)
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
    
    for flow in flow_maps:
        # print(flow.shape)
        # Convert flow to color for visualization
        colored_flow = flow_vis_torch.flow_to_color(flow)
        grid = torchvision.utils.make_grid(colored_flow, nrow=n_rows)
        grid = grid.permute(1, 2, 0).detach().cpu().numpy()  # Convert to (H, W, C)

        # Plot arrows on the flow image
        fig, ax = plt.subplots(figsize=(grid.shape[1] / 100, grid.shape[0] / 100), dpi=100)
        ax.imshow(grid / 255.0)
        
        # Extract flow at the downsampled grid points
        u = flow[0, 0, y_grid, x_grid].detach().cpu().numpy()  # Flow in x direction
        v = flow[0, 1, y_grid, x_grid].detach().cpu().numpy()  # Flow in y direction
        
        # Plot arrows using fixed grid points
        ax.quiver(x_grid, y_grid, u, v, color='black', angles='xy', scale_units='xy', scale=1)

        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Convert the Matplotlib figure to a PIL image
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        img = Image.fromarray(image)
        #img.save('debug_vis/flow' + str(count) + '.png')
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

def save_magnitude_gif(magnitude: torch.Tensor, path: str, rescale=True, fps=8):
    """
    Save the magnitude tensor as a grayscale GIF.
    
    Parameters:
    - magnitude: Tensor of shape (f, h, w) containing the magnitude values.
    - path: Path to save the GIF.
    - rescale: Whether to rescale the values to [0, 255] for visualization.
    - fps: Frames per second for the GIF.
    """
    # Clone and optionally rescale the magnitude tensor
    sample = magnitude.clone().cpu()  # Ensure it's on CPU for visualization
    if rescale:
        sample = (sample - sample.min()) / (sample.max() - sample.min())  # Normalize to [0, 1]
        sample = (sample * 255).byte()  # Scale to [0, 255]
    
    # Convert to numpy for visualization
    sample_np = sample.numpy()

    # Create a list to hold the frames
    frames = []

    # Iterate through each frame and convert to a grayscale image
    for i in range(sample_np.shape[0]):  # Loop over frames
        frame = sample_np[i, :, :]  # Get the ith frame (shape: h, w)
        
        # Convert the frame to a PIL image (grayscale)
        img = Image.fromarray(frame, mode='L')  # 'L' for grayscale
        frames.append(img)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the frames as a GIF
    try:
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),  # Duration between frames in milliseconds
            loop=0  # Loop indefinitely
        )
        print(f"GIF saved successfully at {path}")
    except Exception as e:
        print(f"An error occurred while saving the GIF: {e}")


def visualize_mask(mask: torch.Tensor, save_path: str = None):
    """
    Visualize a mask tensor of shape (1, 1, h, w) as a grayscale image.
    
    Parameters:
    - mask: Tensor of shape (1, 1, h, w), where 0 typically represents absence and 1 represents presence.
    - save_path: Optional path to save the image as a PNG. If None, the image is only displayed.
    """
    # Squeeze the mask to remove dimensions (1, 1, h, w) -> (h, w)
    mask_squeezed = mask.squeeze(0).squeeze(0).detach().cpu().numpy()  # Convert to numpy for visualization
    
    # Optionally normalize the mask to [0, 255] if it's not binary
    if mask_squeezed.max() > 1 or mask_squeezed.min() < 0:
        mask_squeezed = (mask_squeezed - mask_squeezed.min()) / (mask_squeezed.max() - mask_squeezed.min())  # Normalize to [0, 1]
    mask_squeezed = (mask_squeezed * 255).astype(np.uint8)  # Scale to [0, 255]
    
    # Create the image from the mask
    img = Image.fromarray(mask_squeezed, mode='L')  # 'L' mode for grayscale
    
    # Display the image
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # No axis for visualization
    plt.show()
    
    # Optionally save the image
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)
        print(f"Mask saved successfully at {save_path}")

# def save_flow_grid_pillow_arrow(flow: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
#     """
#     Save the optical flow maps as a video grid augmented with arrows depicting the flow.
    
#     Parameters:
#     - flow_maps: Tensor of shape (b, t, c, h, w) containing optical flow maps.
#     - path: Path to save the video.
#     - rescale: Whether to rescale the values to [0, 1].
#     - n_rows: Number of rows in the grid.
#     - fps: Frames per second for the video.
#     """
#     sample = flow.clone()
#     w = flow.shape[3]
#     h = flow.shape[2]
#     sample = (sample * 2 - 1).clamp(-1, 1)
#     #sample = sample * (1 - brush_mask.to(sample.device))
#     sample[:, 0:1, ...] = sample[:, 0:1, ...] * w
#     sample[:, 1:2, ...] = sample[:, 1:2, ...] * h

#     flow_pre = sample.squeeze(0)
#     flow_pre = rearrange(flow_pre, "c f h w -> f c h w")
#     # flow_pre = torch.cat(
#     # [torch.zeros(1, 2, h, w).to(flow_pre.device), flow_pre], dim=0
#     # )
 
#     flow_maps = []
#     flow_maps.append(flow_pre)
    
#     flow_maps = rearrange(flow_maps, "b t c h w -> t b c h w")
#     outputs = []
#     count = 0
    
#     for flow in flow_maps:
#         print(flow.shape)
#         # Convert flow to color for visualization
#         colored_flow = flow_vis_torch.flow_to_color(flow)
#         grid = torchvision.utils.make_grid(colored_flow, nrow=n_rows)
#         grid = grid.permute(1, 2, 0).detach().cpu().numpy()  # Convert to (H, W, C)

#         # Plot arrows on the flow image
#         fig, ax = plt.subplots(figsize=(grid.shape[1] / 100, grid.shape[0] / 100), dpi=100)
#         ax.imshow(grid / 255.0)
        
#         # Downsample the flow for quiver to avoid clutter (optional)
#         step = 30
#         for i in range(0, flow.shape[2], step):
#             for j in range(0, flow.shape[3], step):
#                 # Extract the flow vector
#                 u = flow[0, 0, i, j].item()
#                 v = flow[0, 1, i, j].item()
                
#                 # Draw arrow
#                 ax.arrow(j, i, u, v, color='black', head_width=2, head_length=3, width=0.5)

#         ax.axis('off')
#         plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
#         # Convert the Matplotlib figure to a PIL image
#         fig.canvas.draw()
#         image = np.array(fig.canvas.renderer.buffer_rgba())
#         img = Image.fromarray(image)
#         img.save('flow' + str(count) + '.png')
#         outputs.append(Image.fromarray(image))
#         count += 1
#         plt.close(fig)  # Close the figure to avoid memory issues

#     print(f"Total frames: {len(outputs)}")
#     print(f"Output frame shape: {outputs[0].size if outputs else 'N/A'}")

#     os.makedirs(os.path.dirname(path), exist_ok=True)
    
#     # Save the frames as a GIF
#     try:
#         outputs[0].save(
#             path,
#             save_all=True,
#             append_images=outputs[1:],
#             duration=int(1000 / fps),
#             loop=0
#         )
#         print(f"GIF saved successfully at {path}")
#     except Exception as e:
#         print(f"An error occurred while saving the GIF: {e}")


def compute_exact_distance_map(mask, falloff_strength=10):
    """
    Computes an exact Euclidean distance map using SciPy's distance transform.
    
    mask: Binary mask of shape (1, 1, h, w) indicating allowed regions.
    falloff_strength: Controls the sharpness of the fall-off for the penalty.
    
    Returns:
    - Exact Euclidean distance map with fall-off applied.
    """
    # Convert mask to numpy (since we are working with torch)
    mask_np = mask.squeeze().cpu().numpy()  # Shape (h, w)

    # Compute the Euclidean distance transform (distance to the nearest mask pixel)
    distance_map = distance_transform_edt(mask_np)  

    # Apply an exponential fall-off based on the distance
    distance_map = 1 - np.exp(-falloff_strength * distance_map /  distance_map.max())

    # Convert back to torch tensor
    return torch.from_numpy(distance_map).unsqueeze(0).to(mask.device)  

def mask_restriction_loss_falloff(flow, mask, epsilon=1e-8, falloff_strength=10):
    """
    Computes a loss that restricts optical flow to operate only within a mask across all frames,
    with a fall-off that decreases the penalty as the distance from the mask increases.
    
    flow: Tensor of shape (c, f, h, w), where c is usually 2 (u, v components).
    mask: Binary mask of shape (1, 1, h, w) indicating allowed regions for the flow.
    epsilon: Small value added to avoid division by zero.
    falloff_strength: Controls the sharpness of the fall-off for the penalty.
    
    Returns:
    - Loss that penalizes flow outside the masked region across all frames, with a fall-off.
    """
    # Ensure mask and flow are on the same device
    device = flow.device
    mask = mask.to(device)



    #save_flow_grid_pillow_arrow(flow_list, 'debug_vis/flow.gif')

    # Calculate the magnitude of the flow across all frames
    magnitude = torch.sqrt(torch.sum(flow ** 2, dim=0, keepdim=True) + epsilon)  # Shape (f, h, w)

     # Expand the mask to match the flow dimensions (across frames)
    mask_expanded = mask[0].expand_as(magnitude)  # Shape (f, h, w)
   

    print(magnitude.mean())
    print(magnitude.max())
    print(magnitude.min())

    # Create a sharp fall-off distance map (based on the first frame)
    distance_map = compute_exact_distance_map(mask) 
    # F.conv2d(mask.float(), weight=torch.ones(1, 1, 3, 3).to(device), padding=1)  # Shape (1, 1, h, w)
    # distance_map = torch.where(mask > 0, torch.tensor(0.0).to(device), distance_map)  # 0 inside mask
    # distance_map = distance_map + (distance_map == 0).float() * 1e8  # Large value to ignore non-mask pixels
    # distance_map = torch.sqrt(distance_map)  # To get an approximate distance measure
    # distance_map = torch.exp(-falloff_strength * distance_map)  # Sharp fall-off with distance
    # Expand the distance map to all frames
    distance_map_expanded = distance_map.expand_as(magnitude)  # Shape (f, h, w)

    print(distance_map.max())
    print(distance_map.min())
    print(distance_map.mean())

    dist_vis = torch.sum(distance_map_expanded.squeeze(), dim=0, keepdim=True).squeeze().detach().cpu().numpy() 
    print(magnitude.shape)
    flow_mag_vis = torch.sum(magnitude.squeeze(), dim=0, keepdim=True).squeeze().detach().cpu().numpy()

    print(mask_expanded.shape)
    mask_vis = torch.sum(mask_expanded.squeeze(), dim=0, keepdim=True).squeeze().detach().cpu().numpy()
    
    

    plt.imshow(mask_vis, cmap='viridis')
    plt.colorbar()
    plt.title('Mask')
    plt.savefig('mask_map.png')
    plt.close()
        
    plt.imshow(dist_vis, cmap='viridis')
    plt.colorbar()
    plt.title('Distance Map with Fall-Off')
    plt.savefig('dist_map.png')
    plt.close()

    plt.imshow(flow_mag_vis, cmap='viridis')
    plt.colorbar()
    plt.title('Flow magnitudes')
    plt.savefig('flow_mag_map.png')
    plt.close()

    plt.imshow(flow_mag_vis, cmap='viridis')
    plt.colorbar()
    plt.title('Flow magnitudes')
    plt.savefig('flow_mag_map.png')
    plt.close()
    

    print(distance_map_expanded.shape)

    # Apply the fall-off distance map to penalize flow outside the mask across all frames
    loss = torch.mean(magnitude * (mask_expanded) * distance_map_expanded)  # Shape (f, h, w)

    return loss

def mask_restriction_loss(flow, mask, epsilon=1e-8):
    """
    Computes a loss that restricts optical flow to operate only within a mask on the first frame.
    flow: Tensor of shape (c, f, h, w), where c is usually 2 (u, v components).
    mask: Binary mask of shape (1, 1, h, w) indicating allowed regions for the flow.
    epsilon: Small value added to avoid division by zero.
    
    Returns:
    - Loss that penalizes flow outside the masked region in the first frame.
    """
    # Extract the first frame of the flow
    flow_first_frame = flow[:, 0, :, :]  # Shape (c, h, w)
    
    # Calculate the magnitude of the flow for the first frame
    magnitude = torch.sqrt(torch.sum(flow_first_frame ** 2, dim=0, keepdim=True) + epsilon)  # Shape (1, h, w)
    
    # Calculate the loss: Penalize flow outside the mask
    loss = torch.mean(magnitude * (1 - mask)) #- torch.mean(magnitude * mask)  # Apply the mask directly without expanding
    
    return loss

def mask_restriction_loss_indexing(flow, mask, epsilon=1e-8):
    """
    Computes a loss that restricts optical flow to operate only within a mask on the first frame.
    flow: Tensor of shape (c, f, h, w), where c is usually 2 (u, v components).
    mask: Binary mask of shape (1, 1, h, w) indicating allowed regions for the flow.
    epsilon: Small value added to avoid division by zero.
    
    Returns:
    - Loss that penalizes flow outside the masked region in the first frame.
    """
    flow = (flow * 2 - 1).clamp(-1, 1)
    
    # Extract the first frame of the flow
    flow_first_frame = flow[:, :, :, :]  # Shape (c, h, w)
    
    # Calculate the magnitude of the flow for the first frame
    magnitude = torch.sqrt(torch.sum(flow_first_frame ** 2, dim=0) + epsilon)  # Shape (h, w)

    # Ensure mask is squeezed to (h, w) shape
    mask_squeezed = mask.squeeze(0).squeeze(0)  # Shape (h, w)
    
    # Invert the mask (1 means outside the mask)
    inverted_mask = (mask_squeezed == 0)  # Shape (h, w)

    # Select the flow magnitudes outside the mask
    outside_mask_magnitude = magnitude[inverted_mask]

    # Calculate the loss by averaging the flow magnitudes outside the mask
    if outside_mask_magnitude.numel() > 0:
        loss = torch.mean(outside_mask_magnitude)
    else:
        # Handle case where no pixels are outside the mask
        loss = torch.tensor(0.0, device=flow.device)
    
    return loss


def smoothness_loss(flow):
    """
    Computes the spatial smoothness loss for optical flow.
    flow: Tensor of shape (c, f, h, w), where c is usually 2 (u, v components).
    """
    dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])  # Differences along width (w dimension)
    dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])  # Differences along height (h dimension)
    
    # Smoothness loss is the sum of these gradients
    loss = torch.mean(dx) + torch.mean(dy)
    return loss

def smoothness_loss(flow, mask):
    """
    Computes the spatial smoothness loss for optical flow.
    flow: Tensor of shape (c, f, h, w), where c is usually 2 (u, v components).
    mask: Binary mask of shape (1, 1, h, w) indicating allowed regions for the flow.
    """
    mask_expanded = mask.expand_as(flow)  # Shape (c. f, h, w)
    dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])  # Differences along width (w dimension)
    dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])  # Differences along height (h dimension)
    # Crop the mask to match the size of dx and dy
    mask_x = mask[:, :, :, :-1]  # For width (w dimension)
    mask_y = mask[:, :, :-1, :]  # For height (h dimension)
    
    # Apply the mask to the flow differences
    dx = dx * mask_x
    dy = dy * mask_y
    
    # Smoothness loss is the sum of these gradients, normalized by the sum of the mask
    loss = (torch.sum(dx) + torch.sum(dy)) / (torch.sum(mask_x) + torch.sum(mask_y))    
    return loss


def directional_coherence_loss_all_frames(flow, mask, epsilon=1e-8):
    """
    Computes a loss that encourages flow vectors within the masked region to have similar directions across all frames.
    
    flow: Tensor of shape (c, f, h, w), where c is usually 2 (u, v components).
    mask: Binary mask of shape (1, 1, h, w) indicating the region of interest for coherence.
    epsilon: Small value added to avoid division by zero to prevent NaNs.
    
    Returns:
    - Directional coherence loss.
    """
    # Ensure mask and flow are on the same device
    device = flow.device
    mask = mask.to(device)

    # Expand the mask to match the flow dimensions
    mask_expanded = mask.expand_as(flow)  # Shape (c, f, h, w)

    # Apply the mask to the flow to get the region of interest for all frames
    masked_flow = flow * mask_expanded  # Shape (c, f, h, w)

    # Normalize the flow vectors to compute directional alignment
    flow_magnitude = torch.sqrt(torch.sum(masked_flow ** 2, dim=0, keepdim=True) + epsilon)  # Shape (f, h, w)
    normalized_flow = masked_flow / flow_magnitude  # Normalized flow vectors, shape (c, f, h, w)

    # Compute pairwise dot products to assess directional coherence for each frame
    dot_product = torch.einsum('cfhw,cfhw->fhw', normalized_flow, normalized_flow)  # Shape (f, h, w)

    # Since we only need to penalize deviations from coherence, subtract from 1
    directional_loss = 1 - dot_product.mean()

    return directional_loss

def directional_coherence_loss(flow, mask, epsilon=1e-8):
    """
    Computes a loss that encourages flow vectors within the masked region to have similar directions.
    
    flow: Tensor of shape (c, f, h, w), where c is usually 2 (u, v components).
    mask: Binary mask of shape (1, 1, h, w) indicating the region of interest for coherence.
    epsilon: Small value added to avoid division by zero to prevent NaNs.
    
    Returns:
    - Directional coherence loss.
    """
    # Extract the first frame of the flow and mask
    flow_first_frame = flow[:, 0, :, :]  # Shape (c, h, w)
    mask_first_frame = mask[:, 0, :, :]  # Shape (1, h, w)

    # Apply the mask to the flow to get the region of interest
    masked_flow = flow_first_frame * mask_first_frame  # Shape (c, h, w)

    # Normalize the flow vectors to compute directional alignment
    flow_magnitude = torch.sqrt(torch.sum(masked_flow ** 2, dim=0, keepdim=True) + epsilon)  # Shape (1, h, w)
    normalized_flow = masked_flow / flow_magnitude  # Normalized flow vectors, shape (c, h, w)

    print('normalized flow shape')
    print(normalized_flow.shape)

    # Compute pairwise dot products to assess directional coherence
    dot_product = torch.einsum('chw,chw->hw', normalized_flow, normalized_flow)  # Shape (h, w)

    # Since we only need to penalize deviations from coherence, subtract from 1
    directional_loss = 1 - dot_product.mean()

    return directional_loss

def magnitude_coherence_loss(flow, mask, epsilon=1e-8):
    """
    Computes a loss that encourages flow vectors within the masked region to have similar magnitudes.
    
    flow: Tensor of shape (c, f, h, w), where c is usually 2 (u, v components).
    mask: Binary mask of shape (1, 1, h, w) indicating the region of interest for coherence.
    epsilon: Small value added to avoid division by zero to prevent NaNs.
    
    Returns:
    - Magnitude coherence loss.
    """
    # Extract the first frame of the flow and mask
    flow_first_frame = flow[:, 0, :, :]  # Shape (c, h, w)
    mask_first_frame = mask[:, 0, :, :]  # Shape (1, h, w)

    # Apply the mask to the flow to get the region of interest
    masked_flow = flow_first_frame * mask_first_frame  # Shape (c, h, w)

    # Compute the magnitude of the flow
    flow_magnitude = torch.sqrt(torch.sum(masked_flow ** 2, dim=0, keepdim=True) + epsilon)  # Shape (1, h, w)

    # Calculate the mean magnitude within the mask
    mean_magnitude = flow_magnitude.mean()

    # Penalize deviations from the mean magnitude
    magnitude_loss = torch.mean((flow_magnitude - mean_magnitude) ** 2)

    return magnitude_loss

def temporal_consistency_loss_0_to_i(flow, mask, thresh):
    """
    Computes the temporal consistency loss for 0-to-i flow.
    flow: Tensor of shape (c, f, h, w).
    The loss penalizes differences in incremental flow over time.
    """
    flow = (flow * 2 - 1).clamp(-1, 1)    

    c, f, h, w = flow.shape

    mask = mask.squeeze(0).expand(f-1, h, w)

    # Compute the incremental differences in flow between consecutive frames
    temporal_diff = flow[:, 1:, :, :] - flow[:, :-1, :, :]  # Shape: (c, f-1, h, w)

    temporal_diff_magnitude = torch.sqrt(torch.sum(temporal_diff ** 2, dim=0))[mask == 0]

    loss = torch.nn.functional.softplus(temporal_diff_magnitude.mean() - thresh)

    # Temporal consistency loss is the mean of these incremental flow differences
    #loss = torch.mean(incremental_flow_diff)

    return loss

def directional_consistency_loss(flow, mask, thresh=0.5, penalty_type='softplus'):
    """
    Compute a directional consistency loss to penalize sudden changes in flow direction.

    Parameters:
    - flow: torch tensor of shape (C, F, H, W) on the GPU
    - thresh: Threshold value to control the sensitivity of the penalty
    - penalty_type: Type of penalty function to use ('softplus' or 'relu')
    
    Returns:
    - loss: Computed directional consistency loss
    """
    c, f, h, w = flow.shape

    flow = (flow * 2 - 1).clamp(-1, 1)
    flow[:, 0:1, ...] = flow[:, 0:1, ...] * w
    flow[:, 1:2, ...] = flow[:, 1:2, ...] * h

    mask = mask.squeeze(0).expand(f-1, h, w)   
    
    # Extract the flow components
    flow_x = flow[0]  # Shape: (F, H, W)
    flow_y = flow[1]  # Shape: (F, H, W)
    
    # Compute the flow angle (direction) for each frame
    flow_angle = torch.atan2(flow_y, flow_x)  # Shape: (F, H, W)
    
    # Compute temporal angle differences (angle changes between consecutive frames)
    temporal_angle_diff = flow_angle[1:, :, :] - flow_angle[:-1, :, :]
    
    # Wrap angle differences to [-pi, pi] to handle angular wraparounds correctly
    temporal_angle_diff = (temporal_angle_diff + torch.pi) % (2 * torch.pi) - torch.pi
    
    # Take absolute value of angle differences to measure magnitude of change
    abs_angle_diff = torch.abs(temporal_angle_diff)[mask == 0]

    mean_angle_diff = abs_angle_diff.mean()
    
    if penalty_type == 'softplus':
        # Apply softplus penalty for large directional changes
        loss = torch.nn.functional.softplus(mean_angle_diff - thresh)
    elif penalty_type == 'relu':
        # Apply ReLU penalty for large directional changes
        loss = torch.nn.functional.relu(mean_angle_diff - thresh)
    else:
        raise ValueError("Unsupported penalty type. Choose 'softplus' or 'relu'.")
    
    return loss


def temporal_consistency_loss(flow):
    """
    Computes the temporal consistency loss for optical flow.
    flow: Tensor of shape (c, f, h, w).
    """
    flow = (flow * 2 - 1).clamp(-1, 1)

    dt = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :])  # Differences along the time dimension (f dimension)
    
    # Temporal consistency loss is the sum of these differences
    loss = torch.mean(dt)
    return loss


def estimate_camera_translation(flow):
    """
    Estimate the camera's translational motion based on the dominant flow in the scene.
    - flow: Tensor of shape (2, h, w) representing the optical flow field (u, v components).
    Returns:
    - camera_translation: Tensor of shape (2,) representing the estimated global camera translation.
    """
    u = flow[0, :, :]  # u component of flow
    v = flow[1, :, :]  # v component of flow
    
    median_u = torch.median(u)
    median_v = torch.median(v)
    
    camera_translation = torch.tensor([median_u, median_v], device=flow.device)
    
    return camera_translation

def remove_camera_translation(flow):
    """
    Remove the estimated camera translation from the flow field.
    - flow: Tensor of shape (2, h, w) representing the optical flow field.
    Returns:
    - flow_corrected: Tensor of shape (2, h, w) with camera translation removed.
    """
    camera_translation = estimate_camera_translation(flow)
    
    # Subtract the camera translation from each flow vector
    flow_corrected = flow - camera_translation.view(2, 1, 1)
    
    return flow_corrected

def process_flow_sequence(flow_sequence, mask, epsilon=1e-6):
    """
    Process the flow sequence to apply divergence, curl, and translation-invariant losses.
    - flow_sequence: Tensor of shape (2, f, h, w), where 2 are the flow components (u, v), f is the number of frames.
    - mask: Binary mask of shape (h, w) indicating the foreground region.
    """
    c, f, h, w = flow_sequence.shape
    
    # Initialize total losses
    total_div_loss = 0.0
    total_curl_loss = 0.0
    total_translation_loss = 0.0

    total_bg_div = 0.0
    total_bg_curl = 0.0
    total_bg_translation = 0.0

    mask = mask.squeeze(0).squeeze(0)

    for i in range(f):
        # Extract the flow map for the current frame
        flow = flow_sequence[:, i, :, :]  # shape (2, h, w)
        
        # Compute divergence and curl
        divergence, curl = compute_divergence_and_curl(flow)
        
        # Estimate and remove camera translation
        flow_corrected = remove_camera_translation(flow)
        
        # Compute background divergence and curl
        bg_divergence = divergence[mask > 0]
        bg_curl = curl[mask > 0]
        bg_translation = torch.mean(flow[0][mask > 0]**2 + flow[1][mask > 0]**2)

        total_bg_div = total_bg_div + torch.mean(bg_divergence)
        total_bg_curl = total_bg_curl + torch.mean(bg_curl)
        total_bg_translation = total_bg_translation + bg_translation
        
        # Compute foreground divergence and curl
        fg_divergence = divergence[mask == 0]
        fg_curl = curl[mask == 0]
        
        # Compute losses
        div_loss = torch.mean(torch.abs(fg_divergence - torch.mean(bg_divergence)))
        curl_loss = torch.mean(torch.abs(fg_curl - torch.mean(bg_curl)))
        
        # Compute translation-invariant loss
        fg_flow_corrected = flow_corrected[:, mask == 0]
        bg_flow_corrected = flow_corrected[:, mask > 0]
        bg_flow_corrected_mean = torch.mean(flow_corrected[:, mask > 0], dim=1, keepdim=True)  # shape (2, 1)
        translation_loss = torch.mean(torch.abs(fg_flow_corrected) - bg_flow_corrected_mean)
        
        # Accumulate losses
        total_div_loss += div_loss
        total_curl_loss += curl_loss
        total_translation_loss += translation_loss
    
    # Average the losses over frames
    avg_div_loss = total_div_loss / f
    avg_curl_loss = total_curl_loss / f
    avg_translation_loss = total_translation_loss / f

    total_bg_cam = total_bg_translation

    total_bg_cam = total_bg_cam / f
    
    return avg_div_loss, avg_curl_loss, avg_translation_loss, total_bg_cam

def camera_based_loss(flow, mask, timestep, object_motion = True):
    flow =  (flow * 2 - 1).clamp(-1, 1)
    div_loss, curl_loss, translation_loss, cam_loss = process_flow_sequence(flow, mask)
    object_motion_loss = 1 / (1e-2 + translation_loss)
    print('motion loss')
    print(object_motion_loss)
    print('cam loss')
    print(cam_loss)
    total_loss = object_motion_loss #+ 100*cam_loss
    total_loss = torch.nn.functional.softplus(total_loss - 60.0) #+ torch.nn.functional.softplus(10 - total_loss)
    if object_motion:
        return total_loss #+ cam_loss
    else:
        return 1000*cam_loss
    # if(timestep < 10):
    #     total_loss = object_motion_loss + 50*cam_loss
    # else:
    #     total_loss = object_motion_loss        
    return total_loss


#def cam_minimization_loss(flow,mask):
    




def difference_loss(flow, flow_list):
    """Encourages the current flow to be different from a list of optical flows."""
    total_diff = 0.0
    for ref_flow in flow_list:
        diff = torch.mean((flow - ref_flow) ** 2)
        total_diff += diff
    avg_diff = total_diff / len(flow_list)
    return -avg_diff  # Negative to maximize difference

def pairwise_difference_loss(flow, flow_list, mask, thresh = 20.0, epsilon=1e-8):
    """
    Computes a pairwise difference loss for optical flows.
    flow: Tensor of shape (c, f, h, w) representing the current optical flow.
    flow_list: List of tensors, each of shape (c, f, h, w), representing the reference optical flows.
    epsilon: Small value added to avoid division by zero.
    
    Returns:
    - Average pairwise difference loss, considering the u and v components separately.
    """
    # Stack the flow list to create a tensor of shape (N, c, f, h, w)
    stacked_flows = torch.stack(flow_list)  # (N, c, f, h, w)

    mask = mask.expand_as(stacked_flows)
    
    # Compute pairwise differences for each component (u and v separately)
    diff = torch.abs(flow.unsqueeze(0) - stacked_flows)  # (N, c, f, h, w)
    
    # Normalize difference by the flow magnitude (optional, but often useful)
    magnitude = torch.sqrt(flow.unsqueeze(0) ** 2 + stacked_flows ** 2 + epsilon)  # (N, c, f, h, w)
    normalized_diff = diff / (magnitude + epsilon)  # (N, c, f, h, w)
    
    # Sum the normalized differences and average across all flows
    avg_diff = normalized_diff[mask == 0].mean()

    diff_loss = 1 / (1e-2 + avg_diff)

    print('diff loss')
    print(diff_loss)

    diff_loss = torch.nn.functional.softplus(diff_loss - thresh) #+ torch.nn.functional.softplus(50.0 - diff_loss) 
    
    return diff_loss  # Negative to maximize difference

def pairwise_difference_loss_angle_mag(flow, flow_list, mask, thresh=20.0, epsilon=1e-8):
    """
    Computes a pairwise difference loss for optical flows, taking into account both magnitude and angle differences.
    
    Parameters:
    - flow: Tensor of shape (2, f, h, w) representing the current optical flow.
    - flow_list: List of tensors, each of shape (2, f, h, w), representing the reference optical flows.
    - mask: Binary mask tensor indicating foreground (0) and background (1).
    - thresh: Threshold for the softplus function.
    - epsilon: Small value added to avoid division by zero.
    
    Returns:
    - Loss penalizing low diversity in both magnitude and angle.
    """
    # Normalize flow to [-1, 1] range
    flow = (flow * 2 - 1).clamp(-1, 1)
    
    # Stack the flow list to create a tensor of shape (N, 2, f, h, w)
    stacked_flows = torch.stack(flow_list)  # (N, 2, f, h, w)
    
    # Compute the magnitude of the current flow and each flow in the flow_list
    flow_magnitude = torch.norm(flow, dim=0)
    stacked_flow_magnitude = torch.norm(stacked_flows, dim=1)
    
    # Compute pairwise differences
    diff = torch.abs(flow.unsqueeze(0) - stacked_flows)  # (N, 2, f, h, w)
    
    # Normalize difference by the flow magnitude
    magnitude = torch.sqrt(flow.unsqueeze(0)**2 + stacked_flows**2 + epsilon)  # (N, 2, f, h, w)
    normalized_diff = diff / (magnitude + epsilon)  # (N, 2, f, h, w)
    
    # Compute angle difference using dot product
    flow_normalized = F.normalize(flow, dim=0)
    stacked_flows_normalized = F.normalize(stacked_flows, dim=1)
    cosine_similarity = torch.sum(flow_normalized.unsqueeze(0) * stacked_flows_normalized, dim=1)
    angle_diff = torch.acos(torch.clamp(cosine_similarity, -1.0 + epsilon, 1.0 - epsilon))
    
    # Combine normalized magnitude difference and angle difference
    combined_diff = normalized_diff.mean(dim=1) + angle_diff  # (N, f, h, w)
    
    # Expand mask to match combined_diff dimensions
    expanded_mask = mask[0].expand_as(combined_diff)
    
    # Apply the mask to exclude the background (mask == 1 means background)
    avg_combined_diff = combined_diff[expanded_mask == 0].mean()
    
    # Calculate loss as inverse of average combined difference to encourage diversity
    diff_loss = 1 / (1e-2 + avg_combined_diff)
    
    # Apply softplus to the loss
    diff_loss = F.softplus(diff_loss - thresh)
    
    return diff_loss
    


def pairwise_dot_product_loss(flow, flow_list, mask, thresh=0.5, epsilon=1e-8):
    """
    Computes a pairwise loss for optical flows based on dot products.
    
    Parameters:
    - flow: Tensor of shape (2, f, h, w) representing the current optical flow.
    - flow_list: List of tensors, each of shape (2, f, h, w), representing the reference optical flows.
    - mask: Binary mask tensor of shape (1, f, h, w) indicating foreground (0) and background (1).
    - thresh: Threshold for the softplus function.
    - epsilon: Small value added to avoid division by zero and numerical instability.
    
    Returns:
    - Loss based on the dot product between the current flow and the flow list.
    """
    # Ensure flow is in the range [-1, 1]
    flow = (flow * 2 - 1).clamp(-1, 1)
    
    # Stack the flow list to create a tensor of shape (N, 2, f, h, w)
    stacked_flows = torch.stack([((f * 2 - 1).clamp(-1, 1)) for f in flow_list])
    
    # Compute flow magnitudes
    flow_mag = torch.norm(flow, dim=0, keepdim=True)
    stacked_flows_mag = torch.norm(stacked_flows, dim=1)


    # Define a magnitude threshold
    mag_threshold = 1e-2  # Adjust as needed

    # Create masks for valid magnitudes
    valid_flow_mag = (flow_mag.squeeze(0) >= mag_threshold)  # Shape: (f, h, w)
    valid_stacked_flows_mag = (stacked_flows_mag >= mag_threshold)  # Shape: (N, f, h, w)

    # Combine masks to find pixels where both flows have sufficient magnitude
    valid_magnitude_mask = valid_flow_mag.unsqueeze(0) & valid_stacked_flows_mag

    # Compute dot products
    dot_products = torch.sum(flow.unsqueeze(0) * stacked_flows, dim=1)
    
    # Normalize dot products by magnitudes
    normalized_dot_products = dot_products / (flow_mag.squeeze(0) * stacked_flows_mag + epsilon)
    
    # Expand mask to match normalized_dot_products dimensions
    expanded_mask = mask.expand_as(normalized_dot_products)

    # Combine with the existing mask (foreground mask)
    combined_mask = expanded_mask == 0  # Foreground pixels


    # Identify pixels where one flow has sufficient magnitude and the other doesn't
    one_sufficient = valid_flow_mag.unsqueeze(0) ^ valid_stacked_flows_mag  # Shape: (N, f, h, w)

    # Identify pixels where both flows have sufficient magnitude
    both_sufficient = valid_flow_mag.unsqueeze(0) & valid_stacked_flows_mag  # Shape: (N, f, h, w)

    # Create a mask for valid pixels (include pixels where one flow has magnitude and the other doesn't)
    valid_magnitude_mask = one_sufficient | both_sufficient  # Include pixels where one or both flows have sufficient magnitude

    # Adjust the normalized dot products
    adjusted_dot_products = normalized_dot_products.clone()

    # Set dot product to -1 where one flow has magnitude and the other doesn't
    adjusted_dot_products[one_sufficient] = -2.0

    # Optionally, exclude pixels where both flows have insufficient magnitude by setting them to zero
    adjusted_dot_products[~valid_magnitude_mask] = 0.0

    # Final mask includes only valid pixels in the foreground
    final_mask = combined_mask & valid_magnitude_mask


    # # Final mask includes only valid pixels
    # final_mask = combined_mask & valid_magnitude_mask

    # # Apply the mask to exclude the background (mask == 1 means background)
    # masked_dot_products = normalized_dot_products[expanded_mask == 0]

    # Apply the mask
    # masked_dot_products = normalized_dot_products[final_mask]

    masked_dot_products = adjusted_dot_products[final_mask]


    if masked_dot_products.numel() == 0:
        return torch.tensor(0.0, device=flow.device)  # Return 0 if no foreground pixels
    
    # Compute the average dot product
    avg_dot_product = masked_dot_products.mean()

    print('diffs avg dot product')
    print(avg_dot_product)
    
    # Calculate loss to encourage diversity (lower dot product means more diverse flows)
    # dot_product_loss = torch.nn.functional.relu(avg_dot_product - thresh)

    print(avg_dot_product)
    
    # Apply softplus to the loss
    final_loss = 10*avg_dot_product
    
    return final_loss


# def pairwise_difference_loss_mag_dir(flow, flow_list, mask, epsilon=1e-8, alpha=0.5, threshold=1e-8):
#     """
#     Computes a pairwise difference loss for optical flows, balancing magnitude and angle differences,
#     with handling for zero flows.

#     Parameters:
#     - flow: Tensor of shape (c, f, h, w) representing the current optical flow.
#     - flow_list: List of tensors, each of shape (c, f, h, w), representing the reference optical flows.
#     - mask: Binary mask tensor indicating foreground (0) and background (1).
#     - epsilon: Small value added to avoid division by zero.
#     - alpha: Weight for magnitude difference (between 0 and 1).
#     - threshold: Threshold value used in the softplus function.

#     Returns:
#     - Loss penalizing low diversity in both magnitude and angle.
#     """

#     # Ensure flow is in the range [-1, 1]
#     flow = (flow * 2 - 1).clamp(-1, 1)
    
#     # Stack the flow list to create a tensor of shape (N, 2, f, h, w)
#     stacked_flows = torch.stack([((f * 2 - 1).clamp(-1, 1)) for f in flow_list])
    
#     # Extract and compute the components of the current flow
#     flow_u = flow[0]  # Shape: (f, h, w)
#     flow_v = flow[1]  # Shape: (f, h, w)
#     flow_magnitude = torch.sqrt(flow_u ** 2 + flow_v ** 2 + epsilon)  # Shape: (f, h, w)
#     flow_u_norm = flow_u / flow_magnitude
#     flow_v_norm = flow_v / flow_magnitude

#     # Extract and compute the components of the stacked flows
#     stacked_flow_u = stacked_flows[:, 0, :, :, :]  # Shape: (N, f, h, w)
#     stacked_flow_v = stacked_flows[:, 1, :, :, :]  # Shape: (N, f, h, w)
#     stacked_flow_magnitude = torch.sqrt(stacked_flow_u ** 2 + stacked_flow_v ** 2 + epsilon)  # (N, f, h, w)
#     stacked_flow_u_norm = stacked_flow_u / stacked_flow_magnitude
#     stacked_flow_v_norm = stacked_flow_v / stacked_flow_magnitude

#     # Compute magnitude differences
#     magnitude_diff = torch.abs(flow_magnitude.unsqueeze(0) - stacked_flow_magnitude)  # (N, f, h, w)
#     # Compute denominator and clamp to avoid division by small numbers
#     denominator = flow_magnitude.unsqueeze(0) + stacked_flow_magnitude + epsilon
#     denominator = torch.clamp(denominator, min=2 * epsilon)
#     normalized_magnitude_diff = magnitude_diff / denominator  # (N, f, h, w)

#     # Compute angular differences using dot product
#     dot_product = flow_u_norm.unsqueeze(0) * stacked_flow_u_norm + flow_v_norm.unsqueeze(0) * stacked_flow_v_norm  # (N, f, h, w)
#     dot_product = torch.clamp(dot_product, -1.0, 1.0)
#     angular_diff = 1 - dot_product  # (N, f, h, w)

#     # Handle zero flows
#     zero_flow_mask_current = (flow_magnitude < epsilon)  # Shape: (f, h, w)
#     zero_flow_mask_stacked = (stacked_flow_magnitude < epsilon)  # Shape: (N, f, h, w)
#     both_zero_mask = zero_flow_mask_current.unsqueeze(0) & zero_flow_mask_stacked  # (N, f, h, w)
#     one_zero_mask = zero_flow_mask_current.unsqueeze(0) ^ zero_flow_mask_stacked  # (N, f, h, w)

#     # Set angular differences appropriately
#     angular_diff[both_zero_mask] = 0.0  # No difference if both are zero
#     angular_diff[one_zero_mask] = 1.0  # Maximum difference if one is zero

#     # Normalize angular difference to [0, 1]
#     angular_diff_normalized = angular_diff / 2.0  # Since max angular difference is 2

#     # Combine differences with weighting factors
#     beta = 1.0 - alpha
#     combined_diff = alpha * normalized_magnitude_diff + beta * angular_diff_normalized

#     # Apply the mask to exclude the background
#     mask_expanded = mask[0].unsqueeze(0).expand_as(combined_diff)  # Shape: (N, f, h, w)
#     combined_diff_masked = combined_diff[mask_expanded == 0]

#     # Calculate mean of the combined differences over the foreground pixels
#     avg_combined_diff = combined_diff_masked.mean()
#     avg_combined_diff = avg_combined_diff + epsilon  # Add epsilon to avoid division by zero

#     # Calculate loss as inverse of average combined difference to encourage diversity
#     diff_loss = 1 / (1e-1 + avg_combined_diff)
#     print('diff loss:', diff_loss.item())

#     # Apply softplus to the loss with adjustable threshold
#     diff_loss = torch.nn.functional.softplus(diff_loss - threshold)
#     return diff_loss



def pairwise_difference_loss_mag_dir(flow, flow_list, mask, epsilon=1e-8, alpha=0.01, beta=0.99, max_magnitude=1.0, threshold=1.0):
    """
    Computes a pairwise difference loss for optical flows, balancing magnitude and angle differences,
    and encouraging non-zero magnitudes without explicit magnitude-increasing terms.

    Parameters:
    - flow: Tensor of shape (2, f, h, w) representing the current optical flow.
    - flow_list: List of tensors, each of shape (2, f, h, w), representing the reference optical flows.
    - mask: Binary mask tensor of shape (1, f, h, w) indicating foreground (0) and background (1).
    - epsilon: Small value added to avoid division by zero.
    - alpha: Weight for magnitude difference (between 0 and 1).
    - beta: Weight for angular difference (between 0 and 1).
    - max_magnitude: Maximum expected magnitude for normalization.
    - threshold: Threshold value used in the softplus function.

    Returns:
    - Loss penalizing low diversity in both magnitude and angle, and discouraging zero flows.
    """
    # Ensure flow is in the range [-1, 1]
    flow = (flow * 2 - 1).clamp(-1, 1)

    # print('flow')
    # print(flow)
    
    # Stack the flow list to create a tensor of shape (N, 2, f, h, w)
    stacked_flows = torch.stack([((f * 2 - 1).clamp(-1, 1)) for f in flow_list])

    
    # # Stack the flow list to create a tensor of shape (N, 2, f, h, w)
    #stacked_flows = torch.stack(flow_list)  # (N, 2, f, h, w)

    # Extract the u and v components of the current flow
    flow_u = flow[0]  # Shape: (f, h, w)
    flow_v = flow[1]  # Shape: (f, h, w)
    # Compute the magnitude and normalized components
    flow_magnitude = torch.sqrt(flow_u ** 2 + flow_v ** 2 + epsilon)  # Shape: (f, h, w)
    masked_flow_magnitude = flow_magnitude * (1 - mask[0])  # (f, h, w)
    #print(masked_flow_magnitude)
    total_flow_magnitude = masked_flow_magnitude.sum() + epsilon  # Scalar    
    flow_u_norm = flow_u / flow_magnitude
    flow_v_norm = flow_v / flow_magnitude
    # Apply the mask to the flow magnitude
    

    # Extract and compute the components of the stacked flows
    stacked_flow_u = stacked_flows[:, 0, :, :, :]  # Shape: (N, f, h, w)
    stacked_flow_v = stacked_flows[:, 1, :, :, :]  # Shape: (N, f, h, w)
    stacked_flow_magnitude = torch.sqrt(stacked_flow_u ** 2 + stacked_flow_v ** 2 + epsilon)  # (N, f, h, w)
    stacked_flow_u_norm = stacked_flow_u / stacked_flow_magnitude
    stacked_flow_v_norm = stacked_flow_v / stacked_flow_magnitude

    masked_stacked_flow_magnitude = stacked_flow_magnitude * (1 - mask[0])  # (N, f, h, w)
    total_stacked_flow_magnitude = masked_stacked_flow_magnitude.sum(dim=[1, 2, 3]) + epsilon  # (N,)
    #print(masked_stacked_flow_magnitude)

    # # Compute magnitude differences
    # print('mags')
    # print(total_flow_magnitude)
    # print(total_stacked_flow_magnitude)
    # magnitude_diff = torch.abs(total_flow_magnitude - total_stacked_flow_magnitude)  # (N,)

    # # Normalize magnitude differences
    # num_masked_pixels = (1 - mask[0]).sum() + epsilon  # Scalar
    # normalized_magnitude_diff = magnitude_diff / (max_magnitude * num_masked_pixels)  # (N,)
    

    # Compute unnormalized magnitude differences
    magnitude_diff = torch.abs(flow_magnitude.unsqueeze(0) - stacked_flow_magnitude)  # (N, f, h, w)

    # Normalize magnitude differences by maximum expected magnitude
    normalized_magnitude_diff = magnitude_diff / (max_magnitude + epsilon)  # (N, f, h, w)

    # Compute angular differences using dot product
    dot_product = flow_u_norm.unsqueeze(0) * stacked_flow_u_norm + flow_v_norm.unsqueeze(0) * stacked_flow_v_norm  # (N, f, h, w)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Ensure values are within [-1, 1]

    angular_diff = 1 - dot_product  # (N, f, h, w)
    angular_diff_normalized = angular_diff / 2.0  # Normalize to [0, 1]

    # Handle zero flows
    zero_flow_mask_current = (flow_magnitude < epsilon)  # Shape: (f, h, w)
    zero_flow_mask_stacked = (stacked_flow_magnitude < epsilon)  # Shape: (N, f, h, w)
    both_zero_mask = zero_flow_mask_current.unsqueeze(0) & zero_flow_mask_stacked  # (N, f, h, w)
    one_zero_mask = zero_flow_mask_current.unsqueeze(0) ^ zero_flow_mask_stacked  # (N, f, h, w)

    # Set angular differences appropriately
    angular_diff_normalized[both_zero_mask] = 0.0  # No difference if both flows are zero
    angular_diff_normalized[one_zero_mask] = 1.0  # Maximum difference if one flow is zero

    #angular_diff_normalized 

    # print('mag diff')
    # print(normalized_magnitude_diff)

    # print('ang diff')
    # print(angular_diff_normalized)

    #maybe instead of mean you want per pixel stuffs here

    # # Combine differences with weighting factors
    combined_diff = alpha * normalized_magnitude_diff + beta * angular_diff_normalized  # (N, f, h, w)

    # Apply the mask to exclude the background (mask == 1 means background)
    mask_expanded = mask[0].unsqueeze(0).expand_as(combined_diff)  # Shape: (N, f, h, w)
    combined_diff_masked = combined_diff[mask_expanded == 0]

    # Calculate mean of the combined differences over the foreground pixels
    avg_combined_diff = combined_diff_masked.mean() + epsilon  # Add epsilon to avoid division by zero

    # print('avg diff')
    # print(avg_combined_diff)

    # Calculate loss as inverse of average combined difference to encourage diversity
    # diff_loss = torch.exp(-(avg_combined_diff))
    # print('diff loss:', diff_loss.item())

    # Apply softplus to the loss with adjustable threshold
    #diff_loss = torch.nn.functional.softplus(diff_loss - 18500)
    print('comb diff')
    print(avg_combined_diff)
    print(1.0/avg_combined_diff)
    diff_loss = torch.nn.functional.softplus(1.0 / (0.1 + avg_combined_diff) - 1.0)
    return diff_loss


def sum_magnitude_loss(flow, mask, threshold = 0.1, scale = 1.0, sharpness = 1.0):
    # Ensure flow is in the range [-1, 1]
    flow = (flow * 2 - 1).clamp(-1, 1)
    # Compute flow magnitude
    flow_magnitude = torch.norm(flow, dim=0)
    
    # Ensure mask has the same number of frames as flow
    if mask.shape[1] == 1:
        mask = mask.expand(-1, flow.shape[1], -1, -1)

    print(flow_magnitude.shape)
    print(mask.shape)
    
    # Apply mask
    masked_magnitude = flow_magnitude[mask.squeeze(0) == 0]

    average_masked_magnitude = masked_magnitude.mean()

    # average_masked_magnitude = average_masked_magnitude / flow.shape[1]

    print('avg masked mag')
    print(average_masked_magnitude)

    # max_magnitude = (1.0 / (threshold + 1.414))

    # magnitude_loss = 1.0 / (threshold + average_masked_magnitude) - max_magnitude

    # magnitude_loss = magnitude_loss * threshold

    # print(magnitude_loss)
    
    # magnitude_loss = torch.nn.functional.softplus(magnitude_loss)

    # print(max_magnitude)
    # print(1.0 / (threshold + average_masked_magnitude))
    # print(magnitude_loss)

    # magnitude_loss = epsilon / (epsilon + masked_magnitude)        

    # magnitude_loss = scale*torch.nn.functional.softplus(sharpness*(threshold - average_masked_magnitude))

    magnitude_loss = scale*torch.nn.functional.softplus(sharpness*(1 / (threshold + average_masked_magnitude)))

    # magnitude_loss = torch.nn.functional.softplus(threshold - average_masked_magnitude)

    return 1*(magnitude_loss)

def magnitude_smoothness_loss(flow, mask, fg = True):
    flow = (flow * 2 - 1).clamp(-1, 1)
    magnitude = torch.sqrt(torch.sum(flow ** 2, dim=0))
    mask = mask.squeeze(0).squeeze(0)   
    # Initialize a list to hold the mean magnitude for each frame
    frame_magnitudes = []

    full_mask = mask.unsqueeze(0).expand_as(magnitude)

    # Apply mask to magnitude
    if fg:
        masked_magnitude = magnitude[:, mask == 0]  # shape: (f, num_pixels)
        masked_flow = flow[:, :, mask == 0]  # shape: (2, f, num_pixels)
    else:
        masked_magnitude = magnitude[:, mask > 0]  # shape: (f, num_pixels)
        masked_flow = flow[:, :, mask > 0]  # shape: (2, f, num_pixels)


    print('masked mag shape')
    print(masked_magnitude.shape)
    # Compute difference between consecutive frames
    # Include a zero frame to represent zero initial motion
    zero_frame = torch.zeros_like(masked_magnitude[0:1])  # shape: (1, num_pixels)
    padded_magnitude = torch.cat([zero_frame, masked_magnitude], dim=0)  # shape: (f+1, num_pixels)
    

    diff = padded_magnitude[1:] - padded_magnitude[:-1]  # shape: (f-1, num_pixels)

    # Compute loss as the mean absolute difference
    loss = torch.mean(torch.nn.functional.relu(torch.abs(diff)**2 - 0.01))

    print('masked mag')
    print(masked_magnitude)

    # # Compute normalized flow vectors (unit vectors)
    # epsilon = 1e-4  # Small value to avoid division by zero
    # masked_magnitude_with_eps = masked_magnitude + epsilon

    # print('masked mag with eps')
    # print(masked_magnitude_with_eps)

    # normalized_flow = masked_flow / masked_magnitude_with_eps.unsqueeze(0)  # shape: (2, f, num_pixels)

    # # Transpose to shape (f, 2, num_pixels) for easier computation
    # normalized_flow = normalized_flow.permute(1, 0, 2)  # shape: (f, 2, num_pixels)

    # # Include a zero vector for initial flow direction (since there's no flow at the start)
    # zero_flow_frame = torch.zeros_like(normalized_flow[0:1])  # shape: (1, 2, num_pixels)
    # padded_normalized_flow = torch.cat([zero_flow_frame, normalized_flow], dim=0)  # shape: (f+1, 2, num_pixels)

    # # Compute temporal differences in flow direction
    # delta_direction = padded_normalized_flow[1:] - padded_normalized_flow[:-1]  # shape: (f, 2, num_pixels)

    # # Compute the magnitude of the difference in direction
    # direction_change = torch.sqrt(torch.sum(delta_direction ** 2, dim=1))  # shape: (f, num_pixels)

    # # Compute direction smoothness loss as the mean of the squared differences
    # direction_loss = torch.mean(torch.nn.functional.relu(direction_change ** 2 - 0.1))
    zero_flow_frame = torch.zeros((2, 1, masked_flow.shape[2]), device=masked_flow.device)  # shape: (2, 1, num_pixels)
    padded_flow = masked_flow #torch.cat([zero_flow_frame, masked_flow], dim=1)  # shape: (2, f+1, num_pixels)


    # Compute dot product between flow vectors at consecutive frames
    flow_t = padded_flow[:, :-1, :]  # shape: (2, f, num_pixels)
    flow_t_plus_1 = padded_flow[:, 1:, :]  # shape: (2, f, num_pixels)

    # Compute magnitudes
    mag_t = torch.sqrt(torch.sum(flow_t ** 2, dim=0))  # shape: (f, num_pixels)
    mag_t_plus_1 = torch.sqrt(torch.sum(flow_t_plus_1 ** 2, dim=0))  # shape: (f, num_pixels)


    magnitude_threshold = 1e-4


    # Mask out positions where magnitudes are below the threshold
    valid_mask = (mag_t > magnitude_threshold) & (mag_t_plus_1 > magnitude_threshold)

    # Apply valid_mask to angle_difference
    # valid_angles = angle_diff_relu[valid_mask]



    # Compute dot product between flow vectors
    dot_product = torch.sum(flow_t * flow_t_plus_1, dim=0)  # shape: (f, num_pixels)

    masked_dot_product = dot_product[valid_mask]
    masked_mag = mag_t * mag_t_plus_1
    masked_mag = masked_mag[valid_mask]

    cos_sim = masked_dot_product #/ masked_mag

    # cos_sim = torch.clamp(cosine_similarity, -1.0, 1.0)

    # # Compute magnitudes product with epsilon to prevent division by zero
    # epsilon = 1e-8
    # magnitude_product = mag_t * mag_t_plus_1 + epsilon  # shape: (f, num_pixels)

    # # Compute cosine similarity
    # cosine_similarity = dot_product / magnitude_product  # shape: (f, num_pixels)

    # # Clamp cosine_similarity to [-1, 1] to prevent numerical issues
    # cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)

    # # Compute angle difference in radians
    # angle_difference = torch.acos(cosine_similarity)  # shape: (f, num_pixels)

    # # Convert threshold angle from degrees to radians (30 degrees)
    # threshold_angle = math.pi / 6  # 30 degrees in radians

    # # Apply ReLU to the angle difference
    # angle_diff_relu = torch.relu(angle_difference - threshold_angle)  # shape: (f, num_pixels)



    # Compute direction smoothness loss
    if masked_mag.numel() > 0:
        direction_loss = 0.01*torch.mean(1 / (1 + cos_sim ** 2))
    else:
        direction_loss = torch.tensor(0.0, device=flow.device)

    print('mag reg loss')
    print(loss)

    print('dir change loss')
    print(direction_loss)

    return 100*loss + 10*direction_loss

def generate_motion_loss(flow, mask, thresh, fg = True):
    c, f, h, w = flow.shape
    flow = (flow * 2 - 1).clamp(-1, 1)
    # Initialize total losses
    total_mag_loss = 0.0
    magnitude = torch.sqrt(torch.sum(flow ** 2, dim=0))
    mask = mask.squeeze(0).squeeze(0)   
    # Initialize a list to hold the mean magnitude for each frame
    frame_magnitudes = []

    full_mask = mask.unsqueeze(0).expand_as(magnitude)

    for i in range(f):
        # Extract the flow map for the current frame
        curr_magnitude = magnitude[i]  # shape (h, w)

        if fg:
            fg_magnitude = curr_magnitude[mask == 0]
        else:
            fg_magnitude = curr_magnitude[mask > 0]

        # Compute the mean magnitude for this frame and add to the list
        frame_magnitudes.append(torch.mean(fg_magnitude))

    # Convert list to tensor for easier manipulation
    frame_magnitudes = torch.stack(frame_magnitudes)
    print(frame_magnitudes.shape)

    if fg:
        # Find the frame with the minimum mean magnitude
        fg_mag = torch.mean(magnitude[full_mask == 0])
        bg_mag = torch.mean(magnitude[full_mask == 1])
        # if(fg_mag > bg_mag):
        mean_magnitude = fg_mag - bg_mag
        # else:
        #     mean_magnitude = fg_mag
        #mean_magnitude = mean_magnitude #+ torch.mean(magnitude[full_mask == 0]) - torch.max(magnitude[full_mask == 1])
        # print(mean_mag)
        # mean_magnitude = torch.mean(mean_mag)
        print(mean_magnitude)
        print('object motion mag inv')
        print(1 / (1e-2 + mean_magnitude))


        if(fg_mag > bg_mag):
        # Calculate loss based on the frame with the minimum magnitude
            loss = torch.nn.functional.softplus(1 / (1e-2 + mean_magnitude) - thresh)
        else:
            loss = 10*torch.nn.functional.softplus(100 / (1 + mean_magnitude) - thresh)
        # else:
        #     loss = torch.nn.functional.softplus(10.0 - mean_magnitude)  
        # loss = mean_magnitude - torch.mean(magnitude[full_mask == 1]) #+ torch.nn.functional.softplus(15 - 1 / (1e-2 + mean_magnitude))
        print('object motion loss')
        print(loss)
    else:
        #magnitude[full_mask == 1]
        # Find the frame with the maximum mean magnitude
        mean_magnitude = torch.mean(frame_magnitudes) #+ torch.max(frame_magnitudes)
        mean_magnitude = torch.mean(mean_magnitude)
        print('bg motion mag')
        print(mean_magnitude)
        # Calculate loss based on the frame with the maximum magnitude
        loss = 100*(mean_magnitude)

    return loss

def limit_motion_loss(flow, mask, thresh, fg = True):
    c, f, h, w = flow.shape
    flow = (flow * 2 - 1).clamp(-1, 1)
    # Initialize total losses
    total_mag_loss = 0.0
    magnitude = torch.sqrt(torch.sum(flow ** 2, dim=0))
    mask = mask.squeeze(0).squeeze(0)   
    # Initialize a list to hold the mean magnitude for each frame
    frame_magnitudes = []

    full_mask = mask.unsqueeze(0).expand_as(magnitude)

    for i in range(f):
        # Extract the flow map for the current frame
        curr_magnitude = magnitude[i]  # shape (h, w)

        if fg:
            fg_magnitude = curr_magnitude[mask == 0]
        else:
            fg_magnitude = curr_magnitude[mask > 0]

        # Compute the mean magnitude for this frame and add to the list
        frame_magnitudes.append(torch.mean(fg_magnitude))

    # Convert list to tensor for easier manipulation
    frame_magnitudes = torch.stack(frame_magnitudes)

    if fg:
        # Find the frame with the minimum mean magnitude
        mean_magnitude = torch.mean(torch.nn.functional.relu(magnitude[full_mask==0] - thresh))

        # mean_magnitude = torch.mean(magnitude[full_mask == 0]) - torch.mean(magnitude[full_mask == 1])
        # print(mean_mag)
        # mean_magnitude = torch.mean(mean_mag)
        # print(mean_magnitude)
        print('object motion mag')
        print(mean_magnitude)

        # Calculate loss based on the frame with the minimum magnitude
        loss = mean_magnitude #torch.nn.functional.relu(mean_magnitude - thresh)
        # loss = mean_magnitude - torch.mean(magnitude[full_mask == 1]) #+ torch.nn.functional.softplus(15 - 1 / (1e-2 + mean_magnitude))
        print('limit motion loss')
        print(loss)
    else:
        # Find the frame with the maximum mean magnitude
        mean_magnitude = torch.mean(frame_magnitudes)
        mean_magnitude = torch.mean(mean_magnitude)
        print('bg motion mag')
        print(mean_magnitude)
        # Calculate loss based on the frame with the maximum magnitude
        loss = 100*(mean_magnitude)

    return loss


def combined_optical_flow_loss(flow, flow_list, mask, timestep = 0, lambda_smooth=1.0, lambda_temporal=1.0, lambda_magnitude=1.0, lambda_diff=1.0, lambda_mask=1.0, min_flow_threshold=0.1, max_flow_threshold=1.0, noisy = False, score_loss = False, sample_loss = False, object_motion = True):
    """
    Combined loss for optical flow that encourages dissimilarity from a list of reference flows
    while also ensuring smoothness, temporal consistency, and magnitude regularization.
    
    flow: Tensor of shape (c, f, h, w) representing the current optical flow.
    flow_list: List of tensors, each of shape (c, f, h, w), representing the reference optical flows.
    lambda_smooth: Weight for the spatial smoothness loss.
    lambda_temporal: Weight for the temporal consistency loss.
    lambda_magnitude: Weight for the magnitude regularization loss.
    lambda_diff: Weight for the difference loss (encourages dissimilarity).
    min_flow_threshold: Minimum allowed flow magnitude.
    max_flow_threshold: Maximum allowed flow magnitude.
    """
    # directional_coherence = directional_coherence_loss_all_frames(flow, mask[0])
    #magnitude_coherence = magnitude_coherence_loss(flow, mask[0])
    if not score_loss:
        save_flow_grid_pillow_arrow(flow, 'debug_vis/flow_' + str(timestep) + '.gif')

    #     save_flow_grid_pillow_arrow(flow, 'debug_vis/flow_noisy_' + str(timestep) + '.gif')
    # else:    

    # np.save('debug_vis/flow_' + str(timestep) + '.npy', flow.detach().cpu().numpy())

    # if(timestep > 5 and timestep < 20):
    if(len(flow_list) > 0):
        diff_loss = pairwise_difference_loss_mag_dir(flow, flow_list, mask[0], 0) #pairwise_dot_product_loss(flow, flow_list, mask[0])
    else:
        diff_loss = 0.0

    print('diff loss')
    print(diff_loss)

    # diff_loss = 0.0

    mag_reg = 10.0*magnitude_smoothness_loss(flow, mask[0])


    # if(object_motion):
    motion_reg = 1.0*generate_motion_loss(flow, mask[0], 40, True)  #+ 0.01*limit_motion_loss(flow, mask[0], 0.15, True) #+ generate_motion_loss(flow, mask[0], 20, False)#

    # return motion_reg + 0.01*diff_loss + mag_reg
        # cam_loss = 0.0
    # else:
        # diff_loss = 0
        # motion_reg = 0
    # else:

    cam_loss = generate_motion_loss(flow, mask[0], 20, False)

    print('reg loss')
    print(mag_reg)

    print('cam loss')
    print(cam_loss)

    print('motion loss')
    print(motion_reg)
    # return 5*cam_loss + diff_loss + mag_reg
    cam_thresh = 1.0
    mag_reg_thresh = 1e-1
    motion_reg_thresh = 1.0
    diff_loss_thresh = 8


    if(cam_loss < 0.0):
        cam_loss = torch.nn.functional.relu((cam_loss - cam_thresh))

    if(mag_reg < 1):
        mag_reg = torch.nn.functional.relu((mag_reg - mag_reg_thresh))

    if(motion_reg < 0.1):
        motion_reg = torch.nn.functional.relu(motion_reg - motion_reg_thresh)
    
    # if(diff_loss < 8 and diff_loss != 0):
    #     diff_loss = torch.nn.functional.relu(diff_loss - diff_loss_thresh)


    # print('reg loss')
    # print(mag_reg)

    print('cam loss')
    print(cam_loss)

    print('motion loss')
    print(motion_reg)

    print('diff loss')
    print(diff_loss)

    # mag_reg = torch.tensor(0).to('cuda')

    loss = 3.0*diff_loss + 2.0*cam_loss + 2.5*motion_reg + mag_reg #+ motion_reg #+ mag_reg

    if(sample_loss):
        return diff_loss + cam_loss

    if(score_loss):
        return cam_loss + motion_reg

    # if(cam_loss > 1.0):
    # loss = loss + cam_loss 
    # # if(motion_reg > 1.0):
    # loss = loss + motion_reg

    return loss


    # # diff_loss = pairwise_difference_loss_mag_dir(flow, flow_list, mask[0]) #pairwise_dot_product_loss(flow, flow_list, mask[0])
    # # cam_loss = generate_motion_loss(flow, mask[0], 20, False)
    # print('diff loss') 
    # print(diff_loss)
    # print('motion reg') 
    # print(motion_reg)
    # print('cam loss')
    # print(cam_loss)


    # if(object_motion):
    #     final_loss = motion_reg
    # else:
    #     final_loss = cam_loss
    
    # return final_loss #diff_loss + motion_reg + cam_loss

    #return diff_loss
    
    #return camera_based_loss(flow, mask[0], timestep, object_motion) + camera_based_loss(flow, mask[0], timestep, False) + 1*diff_loss #+ temporal_consistency_loss_0_to_i(flow, mask[0], 3.0) + directional_consistency_loss(flow, mask[0])


        
    # if(object_motion):
    #    return correlation_loss(flow, mask[0])
    

    # return cam_loss + diff_loss #+ generate_motion_loss(flow, mask[0], 20, False)
    # #get_rotation(flow)

    
    # smooth_loss = smoothness_loss(flow,mask[0])
    #temp_loss = temporal_consistency_loss_0_to_i(flow)
    #mag_loss = magnitude_regularization_loss_per_frame_with_mask_indexing(flow, mask[0], timestep, min_flow_threshold, max_flow_threshold)
    #mag_loss = magnitude_regularization_loss_bg(flow, mask[0], timestep)
    #cam_loss = camera_based_loss(flow, mask[0], timestep, object_motion)

    #mask_loss = mask_restriction_loss_indexing(flow, mask[0])

    # print('diff loss')
    # print(diff_loss)
    # print('regularization loss')
    # print(lambda_smooth * smooth_loss + lambda_temporal * temp_loss + lambda_magnitude * mag_loss + lambda_mask * mask_loss)

    #print('losses')
    #print(temp_loss)
    #print(mag_loss)
    #print(cam_loss)


    
    #total_loss = cam_loss #+ 10*temp_loss #+ lambda_temporal * temp_loss

    
    # Combined loss
    # total_loss = (
    #     lambda_smooth * smooth_loss + 
    #     lambda_temporal * temp_loss + 
    #     lambda_magnitude * mag_loss + 
    #     lambda_diff * diff_loss +
    #     lambda_mask * mask_loss
    #)
    #return total_loss