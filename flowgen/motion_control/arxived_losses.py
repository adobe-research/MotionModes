def saturating_emergent_flow_loss(flow, mask, low_threshold=0.1, high_threshold=1.0, beta=1.0, epsilon=1e-2):
    """
    Compute a loss that encourages the emergence of non-zero flows,
    with saturation behavior once sufficient flow is achieved.
    
    Args:
    flow: tensor of shape (2, f, h, w)
    mask: tensor of shape (1, 1, h, w) or (1, f, h, w)
    low_threshold: threshold to consider a flow as non-zero
    high_threshold: threshold at which to consider flow magnitude as sufficient
    beta: parameter for controlling the sharpness of transitions
    epsilon: small value to avoid division by zero
    
    Returns:
    loss: a scalar loss value
    """
    # Compute flow magnitude
    flow_magnitude = torch.norm(flow, dim=0)
    
    # Ensure mask has the same number of frames as flow
    if mask.shape[1] == 1:
        mask = mask.expand(-1, flow.shape[1], -1, -1)
    
    # Apply mask
    masked_magnitude = flow_magnitude * (1 - mask.squeeze(0))
    
    # Compute smooth indicators for low and high thresholds
    above_low = torch.sigmoid((masked_magnitude - low_threshold) * beta)
    above_high = torch.sigmoid((masked_magnitude - high_threshold) * beta)
    
    # Compute the fraction of flows between low and high thresholds
    active_fraction = (above_low - above_high).mean()
    
    # Compute the average magnitude of active flows
    active_magnitude = (masked_magnitude * (above_low - above_high)).sum() / (active_fraction * masked_magnitude.numel() + epsilon)
    
    # Emergence loss: encourage flows to reach the low threshold
    emergence_loss = F.softplus((low_threshold - masked_magnitude) * beta).mean()
    
    # Growth loss: encourage flows between low and high thresholds to grow
    growth_loss = active_fraction * F.softplus((high_threshold - active_magnitude) * beta)
    
    # Saturation factor: reduce loss as more flows exceed the high threshold
    saturation_factor = torch.exp(-above_high.mean() * beta)
    
    # Combine losses with saturation
    loss = (emergence_loss + growth_loss) * saturation_factor
    
    return loss


def saturating_emergent_flow_loss(flow, mask, low_threshold=0.1, high_threshold=1.0, beta=1.0, epsilon=1e-2, saturation_factor=5.0):
    """
    Compute a loss that encourages the emergence of non-zero flows,
    with saturation behavior once sufficient flow is achieved.
    
    Args:
    flow: tensor of shape (2, f, h, w)
    mask: tensor of shape (1, 1, h, w) or (1, f, h, w)
    low_threshold: threshold to consider a flow as non-zero
    high_threshold: threshold at which to consider flow magnitude as sufficient
    beta: parameter for controlling the sharpness of transitions
    epsilon: small value to avoid division by zero
    saturation_factor: controls how quickly the loss saturates
    
    Returns:
    loss: a scalar loss value
    """
    # Ensure flow is in the range [-1, 1]
    flow = (flow * 2 - 1).clamp(-1, 1)
    
    # Compute flow magnitude
    flow_magnitude = torch.norm(flow, dim=0)
    
    # Ensure mask has the same number of frames as flow
    if mask.shape[1] == 1:
        mask = mask.expand(-1, flow.shape[1], -1, -1)
    
    # Apply mask
    masked_magnitude = flow_magnitude * (1 - mask.squeeze(0))
    
    # Compute smoothed indicators for low and high thresholds
    above_low = torch.sigmoid((masked_magnitude - low_threshold) * beta)
    above_high = torch.sigmoid((masked_magnitude - high_threshold) * beta)
    
    # Fraction loss: encourage more above-low-threshold flows
    fraction_loss = F.softplus(-above_low.mean() * beta) / beta
    
    # Magnitude loss: encourage larger magnitudes, especially near the low threshold
    magnitude_loss = F.softplus((low_threshold / (masked_magnitude + epsilon)) * beta).mean() / beta
    
    # Emergence loss: specifically encourage non-zero flows
    emergence_loss = F.softplus((low_threshold - masked_magnitude) * beta).mean() / beta
    
    # Combine base losses
    base_loss = fraction_loss + magnitude_loss + emergence_loss
    
    # Compute saturation based on flows above high threshold
    saturation = torch.exp(-above_high.mean() * saturation_factor)
    
    # Apply saturation to the loss
    loss = base_loss * saturation
    
    # Scale the loss as in the original function
    return 10 * loss

def flexible_magnitude_loss(flow, mask, threshold, percentile=60):
    """
    Compute a loss that encourages the presence of decent magnitude
    without requiring it in every pixel.
    
    Args:
    flow: tensor of shape (2, f, h, w)
    mask: tensor of shape (1, 1, h, w)
    threshold: the magnitude threshold considered "decent"
    percentile: the percentile of pixel magnitudes to consider
    
    Returns:
    loss: a scalar loss value
    """
    flow = (flow * 2 - 1).clamp(-1,1)
    
    # Compute flow magnitude
    flow_magnitude = torch.sqrt(flow[0]**2 + flow[1]**2 + 1e-6)
    
    # Apply mask
    masked_magnitude = flow_magnitude * (1 - mask.squeeze())
    
    # Flatten the masked magnitude tensor
    # flat_magnitude = masked_magnitude.view(-1)

    target_magnitude = 0.0
    
    # Compute the specified percentile of the magnitudes
    for i in range(masked_magnitude.shape[0]):
        flat_magnitude = masked_magnitude[i].view(-1)
        curr_magnitude = torch.quantile(flat_magnitude, q=percentile/100)
        target_magnitude = target_magnitude + curr_magnitude
    
    # Compute the specified percentile of the magnitudes for each frame
    # frame_percentiles = torch.tensor([
    #     torch.quantile(masked_magnitude[i].view(-1), q=percentile/100)
    #     for i in range(masked_magnitude.shape[0])
    # ])
    
    # # Compute mean of frame percentiles
    # mean_percentile = frame_percentiles.mean()
    target_magnitude = target_magnitude #/ masked_magnitude.shape[0]
    # target_magnitude = mean_percentile
    
    print('target mag')
    print(target_magnitude)
    print(1/ (1e-1 + target_magnitude))
    
    # Compute loss
    loss = F.softplus(1 / (1e-1 + target_magnitude) - threshold)
    
    return loss


def alt_flexible_magnitude_loss(flow, mask, threshold=0.5, percentile=80, sharpness=10.0, epsilon=1e-1):
    """
    Compute a loss that encourages the presence of decent magnitude
    in a subset of pixels and frames within a masked region, using smooth weighting.
    
    Args:
    flow: tensor of shape (2, f, h, w)
    mask: tensor of shape (1, 1, h, w) or (1, f, h, w)
    threshold: the magnitude threshold considered "decent"
    percentile: the percentile of pixel magnitudes to consider within each frame
    sharpness: controls how sharply the weighting favors higher percentiles
    epsilon: small value to avoid division by zero
    
    Returns:
    loss: a scalar loss value
    """
    # Ensure flow is in the range [-1, 1]
    flow = (flow * 2 - 1).clamp(-1, 1)
    
    # Compute flow magnitude
    flow_magnitude = torch.norm(flow, dim=0)
    
    # Ensure mask has the same number of frames as flow
    if mask.shape[1] == 1:
        mask = mask.expand(-1, flow.shape[1], -1, -1)
    
    # Apply mask
    masked_magnitude = flow_magnitude * (1 - mask.squeeze(0))
    
    # Compute the specified percentile of the magnitudes for all frames
    frame_percentiles = torch.stack([
        torch.quantile(masked_magnitude[i].view(-1), q=percentile/100)
        for i in range(masked_magnitude.shape[0])
    ])
    
    # Compute weights using a smooth function
    weights = torch.sigmoid(sharpness * (frame_percentiles - frame_percentiles.mean()))
    weights = weights / weights.sum()  # Normalize weights
    
    # Compute weighted average
    weighted_percentile = (frame_percentiles * weights).sum()

    print(weighted_percentile)
    print(1 / (epsilon + weighted_percentile))
    
    # Compute loss
    loss = F.softplus(1 / (epsilon + weighted_percentile) - threshold)
    
    return loss

def get_rotation(flow_sequence):
    c, f, h , w = flow_sequence.shape
    foc =  1655 / 4
    estimator = RobustRotationEstimatorTorch(h, w, foc, bin_size=0.001, max_angle=0.2, spatial_step=15)
    flow_sequence = (flow_sequence * 2 - 1).clamp(-1, 1)
    for i in range(f):
        # Extract the flow map for the current frame
        flow = flow_sequence[:, i, :, :]  # shape (2, h, w)    
        rot_est = estimator.estimate(flow)
        print('frame ' + str(i))
        print(f"Estimated rotation: {rot_est}")   
    return

def magnitude_regularization_loss_relative(flow, mask, epsilon=1e-8):
    """
    Penalizes flow magnitudes in the foreground if they are smaller than those in the background.
    
    Parameters:
    - flow: Tensor of shape (c, f, h, w), where c is the number of channels (u and v components),
      f is the number of frames, and h, w are the height and width.
    - mask: Binary mask of shape (1, 1, h, w), indicating the foreground region for the flow.
    - epsilon: Small value added to avoid division by zero.
    
    Returns:
    - A penalty encouraging higher flow in the masked region than in the background.
    """

    flow = (flow * 2 - 1).clamp(-1, 1)
    w = flow.shape[3]
    h = flow.shape[2]
    # flow[:, 0:1, ...] = flow[:, 0:1, ...] * w
    # flow[:, 1:2, ...] = flow[:, 1:2, ...] * h
    
    # Compute the magnitude of the flow (f, h, w)
    magnitude = torch.sqrt(torch.sum(flow ** 2, dim=0) + epsilon)  # Shape (f, h, w)
    
    # Squeeze the mask to match the shape (h, w)
    mask_squeezed = mask.squeeze(0).squeeze(0)  # Shape (h, w)

    # Initialize total penalty
    total_penalty = 0.0
    
    # Iterate over each frame and compute the loss for that frame
    for frame_idx in range(magnitude.shape[0]):
        # Extract the magnitude and mask for the current frame
        frame_magnitude = magnitude[frame_idx]  # Shape (h, w)
        
        # Mask the flow into foreground (fg) and background (bg)
        fg_magnitude = frame_magnitude[mask_squeezed == 0]
        bg_magnitude = frame_magnitude[mask_squeezed > 0]
        
        if fg_magnitude.numel() > 0 and bg_magnitude.numel() > 0:  # Ensure both regions exist
            # Compute the mean magnitudes for fg and bg
            fg_mean_magnitude = torch.mean(fg_magnitude)
            bg_mean_magnitude = torch.mean(bg_magnitude)
            
            # Enforce that the foreground flow should be larger than the background flow
            #margin = 1.0
            diff_penalty = torch.nn.functional.softplus(bg_mean_magnitude - fg_mean_magnitude)
            # similarity_penalty = torch.exp(-torch.abs(bg_mean_magnitude - fg_mean_magnitude))
            # quadratic_penalty = torch.pow(bg_mean_magnitude - fg_mean_magnitude, 2)
            i

def magnitude_regularization_loss(flow, min_flow_threshold=0.1, max_flow_threshold=1.0):
    """Penalizes optical flow magnitudes that are either too small or too large."""
    magnitude = torch.sqrt(torch.sum(flow ** 2, dim=0) + 1e-8)  # Compute the magnitude
    small_magnitude_penalty = torch.mean(torch.clamp(min_flow_threshold - magnitude, min=0) ** 2)
    large_magnitude_penalty = torch.mean(torch.clamp(magnitude - max_flow_threshold, min=0) ** 2)
    magnitude_penalty = small_magnitude_penalty + large_magnitude_penalty
    return magnitude_penalty

def magnitude_regularization_loss_mask(flow, mask, min_flow_threshold=0.1, max_flow_threshold=1.0, epsilon=1e-8):
    """
    Penalizes optical flow magnitudes that are either too small or too large, with a fall-off based on distance from the mask, across all frames.
    
    flow: Tensor of shape (c, f, h, w), where c is usually 2 (u, v components).
    mask: Binary mask of shape (1, 1, h, w) indicating the region where the loss should be applied.
    min_flow_threshold: Minimum allowed flow magnitude.
    max_flow_threshold: Maximum allowed flow magnitude.
    epsilon: Small value to avoid division by zero.
    """
    # Compute the magnitude of the flow across all frames
    magnitude = torch.sqrt(torch.sum(flow ** 2, dim=0) + epsilon)  # Shape (f, h, w)

    kernel = torch.ones(1, 1, 3, 3, device=flow.device)

    # Compute a distance map: 1 for mask regions, 0 for non-mask regions, and then use distance transform
    mask_binary = mask.squeeze()  # Shape (h, w)
    distance_map = F.conv2d(mask_binary[None, None], weight=kernel, padding=1).squeeze()  # Shape (h, w)
    distance_map = torch.where(mask_binary > 0, torch.tensor(0.0).to(distance_map.device), distance_map)  # 0 inside mask
    distance_map = distance_map + (distance_map == 0).float() * 1e8  # Large value to ignore non-mask pixels
    distance_map = torch.sqrt(distance_map)  # To get an approximate distance measure
    distance_map = torch.exp(-distance_map)  # Exponential fall-off with distance

    # Expand distance map across frames
    distance_map_expanded = distance_map.unsqueeze(0).expand(magnitude.size())  # Shape (f, h, w)

    # Apply the distance map to the magnitude
    weighted_magnitude = magnitude * distance_map_expanded  # Shape (f, h, w)

    # Penalize magnitudes that are too small or too large with the distance-based weight
    small_magnitude_penalty = torch.mean(torch.clamp(min_flow_threshold - weighted_magnitude, min=0) ** 2)
    
    large_magnitude_penalty = torch.mean(torch.clamp(weighted_magnitude - max_flow_threshold, min=0) ** 2)

    # Combine the penalties
    magnitude_penalty = small_magnitude_penalty + large_magnitude_penalty
    
    return magnitude_penalty

def magnitude_regularization_loss_sharp(flow, mask, min_flow_threshold=0.1, max_flow_threshold=1.0, epsilon=1e-8, falloff_strength=100):
    """
    Penalizes optical flow magnitudes that are either too small (strictly within the mask) or too large (with a sharp fall-off from the mask).
    
    flow: Tensor of shape (c, f, h, w), where c is usually 2 (u, v components).
    mask: Binary mask of shape (1, 1, h, w) indicating the region where the loss should be applied.
    min_flow_threshold: Minimum allowed flow magnitude (applied only inside the mask).
    max_flow_threshold: Maximum allowed flow magnitude (applied everywhere with sharp fall-off).
    epsilon: Small value to avoid division by zero.
    falloff_strength: Controls the sharpness of the fall-off for the large magnitude penalty.
    """
    # Ensure mask and flow are on the same device
    device = flow.device
    
    # Compute the magnitude of the flow across all frames
    magnitude = torch.sqrt(torch.sum(flow ** 2, dim=0) + epsilon)  # Shape (f, h, w)

    # Apply binary mask to disable small magnitude penalty outside the mask
    print(mask.squeeze(0).shape)
    mask_expanded = mask.squeeze(0).expand_as(magnitude)  # Shape (f, h, w)
    print(mask_expanded.shape)
    small_magnitude_penalty = torch.mean(torch.clamp(min_flow_threshold - magnitude, min=0) ** 2 * mask_expanded)

    # Create a sharp fall-off distance map
    distance_map = F.conv2d(mask.squeeze()[None, None].float(), weight=torch.ones(1, 1, 3, 3).to(device), padding=1).squeeze()  # Shape (h, w)
    distance_map = torch.where(mask > 0, torch.tensor(0.0).to(device), distance_map)  # 0 inside mask
    distance_map = distance_map + (distance_map == 0).float() * 1e8  # Large value to ignore non-mask pixels
    distance_map = torch.sqrt(distance_map)  # To get an approximate distance measure
    distance_map = torch.exp(-falloff_strength * distance_map)  # Sharp fall-off with distance

    print('distance map shape')
    print(distance_map.shape)
    # Expand distance map across frames
    distance_map_expanded = distance_map[0].expand_as(magnitude)  # Shape (f, h, w)
    print(distance_map_expanded.shape)

    # Apply the sharp distance map to the magnitude for the large magnitude penalty
    weighted_magnitude = magnitude * distance_map_expanded  # Shape (f, h, w)

    # Penalize magnitudes that are too large with the sharp distance-based weight
    large_magnitude_penalty = torch.mean(torch.clamp(weighted_magnitude - max_flow_threshold, min=0) ** 2)

    # Combine the penalties
    magnitude_penalty = small_magnitude_penalty + large_magnitude_penalty
    
    return magnitude_penalty    
ncrease_flow_penalty = torch.mean(torch.nn.functional.softplus(0.3 - fg_magnitude))           
            # Accumulate the penalty for this frame
            total_penalty = total_penalty + diff_penalty #+ increase_flow_penalty #* (1 + (1 + frame_idx) / magnitude.shape[0])

            print('frame penalty')
            print(frame_idx)
            print(diff_penalty)
            print(increase_flow_penalty)            
    
    # Return the total penalty averaged over frames
    return total_penalty / magnitude.shape[0]  # Averaging over frames


def magnitude_regularization_loss(flow, min_flow_threshold=0.1, max_flow_threshold=1.0):
    """Penalizes optical flow magnitudes that are either too small or too large."""
    magnitude = torch.sqrt(torch.sum(flow ** 2, dim=0) + 1e-8)  # Compute the magnitude
    small_magnitude_penalty = torch.mean(torch.clamp(min_flow_threshold - magnitude, min=0) ** 2)
    large_magnitude_penalty = torch.mean(torch.clamp(magnitude - max_flow_threshold, min=0) ** 2)
    magnitude_penalty = small_magnitude_penalty + large_magnitude_penalty
    return magnitude_penalty

def magnitude_regularization_loss_mask(flow, mask, min_flow_threshold=0.1, max_flow_threshold=1.0, epsilon=1e-8):
    """
    Penalizes optical flow magnitudes that are either too small or too large, with a fall-off based on distance from the mask, across all frames.
    
    flow: Tensor of shape (c, f, h, w), where c is usually 2 (u, v components).
    mask: Binary mask of shape (1, 1, h, w) indicating the region where the loss should be applied.
    min_flow_threshold: Minimum allowed flow magnitude.
    max_flow_threshold: Maximum allowed flow magnitude.
    epsilon: Small value to avoid division by zero.
    """
    # Compute the magnitude of the flow across all frames
    magnitude = torch.sqrt(torch.sum(flow ** 2, dim=0) + epsilon)  # Shape (f, h, w)

    kernel = torch.ones(1, 1, 3, 3, device=flow.device)

    # Compute a distance map: 1 for mask regions, 0 for non-mask regions, and then use distance transform
    mask_binary = mask.squeeze()  # Shape (h, w)
    distance_map = F.conv2d(mask_binary[None, None], weight=kernel, padding=1).squeeze()  # Shape (h, w)
    distance_map = torch.where(mask_binary > 0, torch.tensor(0.0).to(distance_map.device), distance_map)  # 0 inside mask
    distance_map = distance_map + (distance_map == 0).float() * 1e8  # Large value to ignore non-mask pixels
    distance_map = torch.sqrt(distance_map)  # To get an approximate distance measure
    distance_map = torch.exp(-distance_map)  # Exponential fall-off with distance

    # Expand distance map across frames
    distance_map_expanded = distance_map.unsqueeze(0).expand(magnitude.size())  # Shape (f, h, w)

    # Apply the distance map to the magnitude
    weighted_magnitude = magnitude * distance_map_expanded  # Shape (f, h, w)

    # Penalize magnitudes that are too small or too large with the distance-based weight
    small_magnitude_penalty = torch.mean(torch.clamp(min_flow_threshold - weighted_magnitude, min=0) ** 2)
    
    large_magnitude_penalty = torch.mean(torch.clamp(weighted_magnitude - max_flow_threshold, min=0) ** 2)

    # Combine the penalties
    magnitude_penalty = small_magnitude_penalty + large_magnitude_penalty
    
    return magnitude_penalty

def magnitude_regularization_loss_sharp(flow, mask, min_flow_threshold=0.1, max_flow_threshold=1.0, epsilon=1e-8, falloff_strength=100):
    """
    Penalizes optical flow magnitudes that are either too small (strictly within the mask) or too large (with a sharp fall-off from the mask).
    
    flow: Tensor of shape (c, f, h, w), where c is usually 2 (u, v components).
    mask: Binary mask of shape (1, 1, h, w) indicating the region where the loss should be applied.
    min_flow_threshold: Minimum allowed flow magnitude (applied only inside the mask).
    max_flow_threshold: Maximum allowed flow magnitude (applied everywhere with sharp fall-off).
    epsilon: Small value to avoid division by zero.
    falloff_strength: Controls the sharpness of the fall-off for the large magnitude penalty.
    """
    # Ensure mask and flow are on the same device
    device = flow.device
    
    # Compute the magnitude of the flow across all frames
    magnitude = torch.sqrt(torch.sum(flow ** 2, dim=0) + epsilon)  # Shape (f, h, w)

    # Apply binary mask to disable small magnitude penalty outside the mask
    print(mask.squeeze(0).shape)
    mask_expanded = mask.squeeze(0).expand_as(magnitude)  # Shape (f, h, w)
    print(mask_expanded.shape)
    small_magnitude_penalty = torch.mean(torch.clamp(min_flow_threshold - magnitude, min=0) ** 2 * mask_expanded)

    # Create a sharp fall-off distance map
    distance_map = F.conv2d(mask.squeeze()[None, None].float(), weight=torch.ones(1, 1, 3, 3).to(device), padding=1).squeeze()  # Shape (h, w)
    distance_map = torch.where(mask > 0, torch.tensor(0.0).to(device), distance_map)  # 0 inside mask
    distance_map = distance_map + (distance_map == 0).float() * 1e8  # Large value to ignore non-mask pixels
    distance_map = torch.sqrt(distance_map)  # To get an approximate distance measure
    distance_map = torch.exp(-falloff_strength * distance_map)  # Sharp fall-off with distance

    print('distance map shape')
    print(distance_map.shape)
    # Expand distance map across frames
    distance_map_expanded = distance_map[0].expand_as(magnitude)  # Shape (f, h, w)
    print(distance_map_expanded.shape)

    # Apply the sharp distance map to the magnitude for the large magnitude penalty
    weighted_magnitude = magnitude * distance_map_expanded  # Shape (f, h, w)

    # Penalize magnitudes that are too large with the sharp distance-based weight
    large_magnitude_penalty = torch.mean(torch.clamp(weighted_magnitude - max_flow_threshold, min=0) ** 2)

    # Combine the penalties
    magnitude_penalty = small_magnitude_penalty + large_magnitude_penalty
    
    return magnitude_penalty    

def magnitude_regularization_loss_bg(flow, mask, timestep, epsilon=1e-8):
    """
    Penalizes flow magnitudes in the foreground if they are smaller than those in the background.
    
    Parameters:
    - flow: Tensor of shape (c, f, h, w), where c is the number of channels (u and v components),
      f is the number of frames, and h, w are the height and width.
    - mask: Binary mask of shape (1, 1, h, w), indicating the foreground region for the flow.
    - epsilon: Small value added to avoid division by zero.
    
    Returns:
    - A penalty encouraging higher flow in the masked region than in the background.
    """

    flow = (flow * 2 - 1).clamp(-1, 1)
    w = flow.shape[3]
    h = flow.shape[2]
    # flow[:, 0:1, ...] = flow[:, 0:1, ...] * w
    # flow[:, 1:2, ...] = flow[:, 1:2, ...] * h
    
    # Compute the magnitude of the flow (f, h, w)
    magnitude = torch.sqrt(torch.sum(flow ** 2, dim=0) + epsilon)  # Shape (f, h, w)

    save_magnitude_gif(magnitude, 'debug_vis/magnitude_' + str(timestep) + '.gif')

    
    # Squeeze the mask to match the shape (h, w)
    mask_squeezed = mask.squeeze(0).squeeze(0)  # Shape (h, w)

    # Initialize total penalty
    total_penalty = 0.0
    
    # Iterate over each frame and compute the loss for that frame
    for frame_idx in range(magnitude.shape[0]):
        # Extract the magnitude and mask for the current frame
        frame_magnitude = magnitude[frame_idx]  # Shape (h, w)
        
        # Mask the flow into foreground (fg) and background (bg)
        fg_magnitude = frame_magnitude[mask_squeezed == 0]
        bg_magnitude = frame_magnitude[mask_squeezed > 0]
        
        if fg_magnitude.numel() > 0 and bg_magnitude.numel() > 0:  # Ensure both regions exist
            # Compute the mean magnitudes for fg and bg
            curr_frame_loss = torch.clamp(frame_magnitude - 0.0, min=0) ** 2
            # Convert tensor to numpy for plotting
            loss_numpy = curr_frame_loss.squeeze().cpu().detach().numpy()

            if(timestep > 15):
                # Plot the tensor with a color map
                plt.imshow(loss_numpy, cmap='viridis')  # You can change 'viridis' to other colormaps like 'plasma', 'inferno', etc.
                plt.colorbar()  # Adds a color bar for reference
                plt.title('Loss Visualization')
                
                # Save the figure to a file (e.g., 'loss_visualization.png')
                plt.savefig('debug_vis/loss_visualization' + str(frame_idx) + '.png')                
                plt.close()
                
            #penalty = torch.mean(torch.clamp(bg_magnitude - 0.0, min=0) ** 2)
            increase_flow_penalty = torch.mean(torch.clamp(0.2 - fg_magnitude, min = 0) **2)
            diff_penalty = torch.abs(torch.mean(fg_magnitude) - torch.mean(bg_magnitude))**(2)
            #diff_penalty = 1.0 - torch.sigmoid(100*diff_penalty)
            diff_penalty = 1 / (1 + diff_penalty + epsilon)

            print('diff penalty')
            print(diff_penalty)
            print('increase flow penalty')
            print(increase_flow_penalty)

            #max_penalty = torch.max(torch.clamp(bg_magnitude - 0.0, min=0) ** 2)
            
            total_penalty = total_penalty + diff_penalty + increase_flow_penalty #+ max_penalty #+ increase_flow_penalty #* (1 + (1 + frame_idx) / magnitude.shape[0])

            # print('frame penalty')
            # print(frame_idx)
            # print(diff_penalty)
            # print(increase_flow_penalty)            
    
    # Return the total penalty averaged over frames
    return 10.0 * total_penalty / magnitude.shape[0]  # Averaging over frames

def magnitude_regularization_loss_per_frame(flow, min_flow_threshold=0.1, max_flow_threshold=1.0):
    """
    Penalizes optical flow magnitudes that are either too small or too large on a per-frame basis.
    
    Parameters:
    - flow: Tensor of shape (c, f, h, w), where c is the number of channels (typically 2 for u and v components),
      f is the number of frames, and h, w are the height and width.
    - min_flow_threshold: Minimum flow magnitude allowed before penalty.
    - max_flow_threshold: Maximum flow magnitude allowed before penalty.
    
    Returns:
    - Per-frame magnitude penalty.
    """
    # Compute the magnitude for each frame (f, h, w)
    magnitude = torch.sqrt(torch.sum(flow ** 2, dim=0) + 1e-8)  # Shape (f, h, w)
    
    # Initialize total penalty
    total_magnitude_penalty = 0.0
    
    # Iterate over each frame and compute the loss for that frame
    for frame_idx in range(magnitude.shape[0]):
        # Extract the magnitude for the current frame
        frame_magnitude = magnitude[frame_idx]  # Shape (h, w)
        
        # Compute the small and large magnitude penalties
        small_magnitude_penalty = torch.mean(torch.clamp(min_flow_threshold - frame_magnitude, min=0) ** 2)
        large_magnitude_penalty = torch.mean(torch.clamp(frame_magnitude - max_flow_threshold, min=0) ** 2)
        
        # Accumulate the penalties for each frame
        frame_penalty = small_magnitude_penalty + large_magnitude_penalty
        total_magnitude_penalty += frame_penalty
    
    # Optionally, you can return the total penalty (sum or mean across frames)
    return total_magnitude_penalty / magnitude.shape[0]  # Averaging over frames


def magnitude_regularization_loss_per_frame_indexing(flow, min_flow_threshold=0.1, max_flow_threshold=1.0):
    """
    Penalizes optical flow magnitudes that are either too small or too large on a per-frame basis.
    
    Parameters:
    - flow: Tensor of shape (c, f, h, w), where c is the number of channels (typically 2 for u and v components),
      f is the number of frames, and h, w are the height and width.
    - min_flow_threshold: Minimum flow magnitude allowed before penalty.
    - max_flow_threshold: Maximum flow magnitude allowed before penalty.
    
    Returns:
    - Per-frame magnitude penalty (averaged across frames).
    """
    
    # Compute the magnitude for each frame (f, h, w)
    magnitude = torch.sqrt(torch.sum(flow ** 2, dim=0) + 1e-8)  # Shape (f, h, w)

    # Penalize small magnitudes (below the min_flow_threshold)
    small_magnitude_penalty = torch.clamp(min_flow_threshold - magnitude, min=0) ** 2

    # Penalize large magnitudes (above the max_flow_threshold)
    large_magnitude_penalty = torch.clamp(magnitude - max_flow_threshold, min=0) ** 2

    # Sum both penalties and compute the mean across frames (f, h, w)
    total_magnitude_penalty = torch.mean(small_magnitude_penalty + large_magnitude_penalty)
    
    return total_magnitude_penalty


def magnitude_regularization_loss_per_frame_with_mask_indexing(flow, mask, timestep, min_flow_threshold=0.1, max_flow_threshold=1.0, epsilon=1e-8):
    """
    Penalizes optical flow magnitudes that are either too small or too large on a per-frame basis within a mask.
    
    Parameters:
    - flow: Tensor of shape (c, f, h, w), where c is the number of channels (typically 2 for u and v components),
      f is the number of frames, and h, w are the height and width.
    - mask: Binary mask of shape (1, 1, h, w), indicating allowed regions for the flow.
    - min_flow_threshold: Minimum flow magnitude allowed before penalty.
    - max_flow_threshold: Maximum flow magnitude allowed before penalty.
    - epsilon: Small value added to avoid division by zero.
    
    Returns:
    - Per-frame magnitude penalty (summed or averaged across frames).
    """

    w = flow.shape[3]
    h = flow.shape[2]
    
    print('pre magnitude flow shape')
    print(flow.shape)
    
    flow = (flow * 2 - 1).clamp(-1, 1)

    #flow[:, 0:1, ...] = flow[:, 0:1, ...] * w
    #flow[:, 1:2, ...] = flow[:, 1:2, ...] * h

    # Compute the magnitude of the flow (f, h, w)
    magnitude = torch.sqrt(torch.sum(flow ** 2, dim=0) + epsilon)  # Shape (f, h, w)
    
    print('magnitude shape')
    print(magnitude.shape)
    
    save_magnitude_gif(magnitude, 'debug_vis/magnitude_' + str(timestep) + '.gif')

    visualize_mask(mask, 'debug_vis/mask.png')

    # Squeeze the mask to match the shape (h, w)
    mask_squeezed = mask.squeeze(0).squeeze(0)  # Shape (h, w)

    #mask_squeezed = 1 - mask_squeezed

    # Initialize total penalty
    total_magnitude_penalty = 0.0

    mag_fg = magnitude.clone()
    mag_bg = magnitude.clone()
    torch.mean(mag_fg, dim = 0)[mask_squeezed > 0] = 0 
    torch.mean(mag_bg, dim = 0)[mask_squeezed <= 0] = 0
    

    save_magnitude_gif(torch.clamp(min_flow_threshold - mag_fg, min=0) ** 2, 'debug_vis/small_mag_penalty_' + str(timestep) + '.gif')
    save_magnitude_gif(torch.clamp(mag_bg - max_flow_threshold, min=0) ** 2, 'debug_vis/large_mag_penalty_' + str(timestep) + '.gif')
    save_magnitude_gif(torch.clamp(mag_fg - 0.5, min=0) ** 2, 'debug_vis/mid_mag_penalty_' + str(timestep) + '.gif')

    # Iterate over each frame and compute the loss for that frame
    for frame_idx in range(magnitude.shape[0]):
        # Extract the magnitude and mask for the current frame
        frame_magnitude = magnitude[frame_idx]  # Shape (h, w)
        mask_expanded = mask_squeezed  # Mask is the same for each frame
        
        # Apply the mask using indexing: select only the pixels inside the mask
        masked_magnitude_bg = frame_magnitude[mask_expanded > 0]
        masked_magnitude_fg = frame_magnitude[mask_expanded <= 0]

        if masked_magnitude_fg.numel() > 0:  # Ensure there are masked pixels
            # Compute penalties for small and large magnitudes inside the mask
            small_magnitude_penalty = torch.mean(torch.clamp(min_flow_threshold - masked_magnitude_fg, min=0) ** 2)
            large_magnitude_penalty = torch.mean(torch.clamp(masked_magnitude_bg - 0.0, min=0) ** 2)
            mid_magnitude_penalty = torch.mean(torch.clamp(masked_magnitude_fg - max_flow_threshold, min=0) ** 2)
            increase_flow_penalty = torch.mean(torch.nn.functional.softplus(min_flow_threshold - masked_magnitude_fg))
            large_flow_penalty = torch.mean(torch.nn.functional.softplus(masked_magnitude_fg - max_flow_threshold))
            bg_flow_penalty = torch.mean(torch.nn.functional.softplus(masked_magnitude_bg - 0.0))
            # print('curr frame small mag loss')
            # print(small_magnitude_penalty)
            # print('low flow penalty')
            # print(increase_flow_penalty)
            # print('curr frame mean mag')
            # print(frame_magnitude.mean())
            # print('curr frame mask mean mag')
            # print(masked_magnitude_fg.mean())

            # Accumulate the penalties for this frame
            frame_penalty = 1.0*(increase_flow_penalty + large_flow_penalty + bg_flow_penalty) 
            #frame_penalty = 10.0*(small_magnitude_penalty + mid_magnitude_penalty + 0.75*large_magnitude_penalty) 

            #+ 1.0*mid_magnitude_penalty #+ 1.0*large_magnitude_penalty #+ mid_magnitude_penalty
            total_magnitude_penalty += frame_penalty

    # Return the total penalty averaged over frames
    return total_magnitude_penalty / magnitude.shape[0]  # Averaging over frames

def compute_divergence_and_curl(flow):
    """
    Compute the divergence and curl of the optical flow field.
    - flow: Tensor of shape (2, h, w) representing the optical flow (u, v components).
    Returns:
    - divergence: Tensor of shape (h, w) representing the divergence of the flow.
    - curl: Tensor of shape (h, w) representing the curl (rotational component) of the flow.
    """
    u = flow[0, :, :]  # u component of flow
    v = flow[1, :, :]  # v component of flow

    du_dx = F.pad(torch.diff(u, dim=1), (1, 0))  # derivative of u w.r.t x
    dv_dy = F.pad(torch.diff(v, dim=0), (0, 0, 1, 0))  # derivative of v w.r.t y
    du_dy = F.pad(torch.diff(u, dim=0), (0, 0, 1, 0))  # derivative of u w.r.t y
    dv_dx = F.pad(torch.diff(v, dim=1), (1, 0))  # derivative of v w.r.t x

    divergence = du_dx + dv_dy
    curl = dv_dx - du_dy

    return divergence, curl
