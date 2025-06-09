import numpy as np 
import torch 
import flow_vis_torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
from einops import rearrange
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
from PIL import Image, ImageFilter, ImageDraw
import os


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
        print(flow.shape)
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

	

def smooth_tracks(pred_tracks, subsample_factor=5):
    """
    Smooth the tracks across frames by subsampling and then interpolating.

    Parameters:
    - pred_tracks: torch.Tensor of shape (1, num_frames, num_points, 2) containing x, y coordinates.
    - subsample_factor: Factor for subsampling frames. Higher values mean fewer frames kept.

    Returns:
    - Smoothed tracks with the same shape as `pred_tracks`.
    """
    # Convert to numpy for processing
    pred_tracks_np = pred_tracks.cpu().numpy()
    
    # Extract shape information
    _, num_frames, num_points, _ = pred_tracks_np.shape
    
    # Frames we want to keep for subsampling
    subsample_indices = np.arange(0, num_frames, subsample_factor)
    if subsample_indices[-1] != num_frames - 1:
        subsample_indices = np.append(subsample_indices, num_frames - 1)
    
    # Subsampled frames
    subsampled_tracks = pred_tracks_np[:, subsample_indices, :, :]

    # Prepare array for smoothed tracks
    smoothed_tracks_np = np.zeros_like(pred_tracks_np)

    # Perform interpolation for each point (x and y) across frames
    for point in range(num_points):
        for coord in range(2):  # 0 for x, 1 for y
            # Get the values at subsampled frames
            subsampled_values = subsampled_tracks[0, :, point, coord]
            
            # Interpolate across the full range of frames
            interp_func = interp1d(subsample_indices, subsampled_values, kind='cubic', fill_value="extrapolate")
            smoothed_tracks_np[0, :, point, coord] = interp_func(np.arange(num_frames))
    
    # Convert back to a torch tensor
    smoothed_tracks = torch.from_numpy(smoothed_tracks_np).to(pred_tracks.device)
    
    return smoothed_tracks

def draw_tracks_on_image(
    image: Image.Image,
    tracks: torch.Tensor,
    visibility: torch.Tensor = None,
    segm_mask: torch.Tensor = None,
    gt_tracks: torch.Tensor = None,
    color_alpha: int = 255,
    linewidth: int = 2,
    color_map=plt.get_cmap('hsv'),
    mode='rainbow',
):
    """
    Draw tracks on a single PIL image, taking into account the visibility of the points.

    Parameters:
    - image: PIL.Image.Image, the image to draw on.
    - tracks: torch.Tensor of shape (1, T, N, 2), the tracks data.
    - visibility: torch.Tensor of shape (1, T, N), visibility data.
    - segm_mask: torch.Tensor, segmentation mask for coloring.
    - gt_tracks: torch.Tensor, ground truth tracks (optional).
    - color_alpha: int, alpha value for the track colors (0-255).
    - linewidth: int, width of the track lines.
    - color_map: matplotlib colormap, color map for tracks.
    - mode: str, 'rainbow' or 'optical_flow' for coloring tracks.

    Returns:
    - result_image: PIL.Image.Image, the image with tracks drawn on it.
    """
    B, T, N, D = tracks.shape
    assert B == 1, "Batch size should be 1."
    assert D == 2, "Tracks should have 2 coordinates (x, y)."

    # Create a copy of the image to avoid modifying the original
    result_image = image.copy()

    # Ensure the image is in RGBA mode to handle transparency
    if result_image.mode != 'RGBA':
        result_image = result_image.convert('RGBA')

    # Convert tracks and visibility to numpy arrays
    tracks_np = tracks[0].cpu().numpy()  # Shape: (T, N, 2)
    if visibility is not None:
        visibility_np = visibility[0].cpu().numpy().astype(bool)  # Shape: (T, N)
    else:
        visibility_np = np.ones((T, N), dtype=bool)

    # Prepare colors for each track
    vector_colors = np.zeros((N, 3))
    if mode == 'rainbow':
        norm = plt.Normalize(0, N)
        for n in range(N):
            color = color_map(norm(n))[:3]  # RGB values between 0 and 1
            vector_colors[n] = np.array(color) * 255  # Scale to 0-255
    else:
        # Default to a single color if not using 'rainbow' mode
        vector_colors[:] = np.array(color_map(0.5)[:3]) * 255  # Default color

    # Create a drawing context
    draw = ImageDraw.Draw(result_image, 'RGBA')

    # Draw tracks for each point
    for n in range(N):
        # Get frames where the point is visible
        visible_frames = np.where(visibility_np[:, n])[0]
        if len(visible_frames) < 2:
            continue  # Need at least two points to draw a line

        # Extract the points where the track is visible
        points = tracks_np[visible_frames, n, :]  # Shape: (num_visible_frames, 2)

        # Convert points to a list of tuples
        points_list = [tuple(p) for p in points]

        # Define the color with alpha channel
        color = tuple(vector_colors[n].astype(int).tolist() + [color_alpha])

        # Draw the track line
        draw.line(points_list, fill=color, width=linewidth)

        # Optionally, draw circles at each point along the track
        for p in points_list:
            r = linewidth  # Radius of the circle
            bbox = [p[0] - r, p[1] - r, p[0] + r, p[1] + r]
            draw.ellipse(bbox, fill=color)

    # Draw ground truth tracks if provided
    if gt_tracks is not None:
        gt_tracks_np = gt_tracks[0].cpu().numpy()  # Shape: (T, N, 2)
        for n in range(gt_tracks_np.shape[1]):
            points = gt_tracks_np[:, n, :]
            points_list = [tuple(p) for p in points]
            # Draw the ground truth track in red
            draw.line(points_list, fill=(255, 0, 0, color_alpha), width=linewidth)

    # Return the image with tracks drawn on it
    return result_image



def make_ghosted_visualization(masked_frames, frames):

    #set alpha to zero for all black pixels in masked_frames
    composite_image = frames[0].copy()

    # Blur the composite image
    composite_image = composite_image.filter(ImageFilter.GaussianBlur(radius=2))

    # make composite_image more transparent
    composite_image.putalpha(90)

    masked_frames = masked_frames

    # Number of frames
    num_frames = len(masked_frames)

    # Iterate over each subsequent frame and adjust its transparency
    for idx, frame in enumerate(masked_frames[1:], start=1):

        if(idx == num_frames - 1):
            transparency_factor = 1
        else:
            transparency_factor = abs((idx / num_frames))  # Decreases from 1 to 0

        # Make a copy of the current frame and modify its transparency
        transparent_frame = frame.copy()
        alpha = transparent_frame.split()[3]  # Get the alpha channel
        adjusted_alpha = alpha.point(lambda p: int(p * transparency_factor))  # Adjust transparency
        transparent_frame.putalpha(adjusted_alpha)

        # Compute the edge map for the frame
        frame = np.array(frame)

        # Blend the transparent frame onto the composite image
        composite_image = Image.alpha_composite(composite_image, transparent_frame)

    #add white background to the composite image
    background = Image.new('RGBA', composite_image.size, (255, 0, 0, 0))
    composite_image = Image.alpha_composite(background, composite_image)

    return composite_image

# example usage 

# Load the frames and optical flow maps
