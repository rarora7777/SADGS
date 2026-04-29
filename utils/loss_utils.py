#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from math import exp, sqrt


C1 = 0.01 ** 2
C2 = 0.03 ** 2

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def tone_curve_loss(network_output, gt, epsilon=1e-6,norm=2):
    """
    Implements the "gradient supervision" tone curve loss.
    L = mean( ((pred - gt) / (sg(pred) + epsilon))**2 )
    where sg indicates a stop-gradient.
    """
    return (((network_output - gt) / (network_output.detach() + epsilon)) ** norm).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



def frequency_loss(means2D, cov2D, st_map, H, W):
    # Normalize coordinates to [-1,1] for grid_sample
    nx = (means2D[:,0] / (W - 1)) * 2 - 1
    ny = (means2D[:,1] / (H - 1)) * 2 - 1
    grid = torch.stack([nx, ny], dim=-1).view(1,1,-1,2)

    # Sample structure tensor
    Sxx, Sxy, Syy = F.grid_sample(st_map, grid, align_corners=True, padding_mode='border').view(3, -1)

    # Compute directional energy eta = u^T J u for the Gaussian principal axis directions
    a, b, c = cov2D[:,0], cov2D[:,1], cov2D[:,2]  # ellipse parameters

    # 1. Compute Determinant (Area^2)
    det = a * c - b * b
    
    # 2. Compute Linear Scale (Radius approx)
    # sqrt(det) = Area (s^2). sqrt(sqrt(det)) = Radius (s).
    sigma_area = torch.sqrt(det + 1e-6)
    sigma_linear = torch.sqrt(sigma_area + 1e-6) 

    # 3. Frequency Threshold (Wavelength in pixels)
    # Ensure st_map is normalized so freq_target is in meaningful units (e.g., 0.0 to 1.0)
    freq_target = torch.sqrt(Sxx + Syy + 1e-6) 
    
    wavelength_limit = 1.0 / (freq_target + 1e-6)

    # 4. Loss: Penalize if Radius > Wavelength
    # Using 'sigma_linear' ensures we compare pixels to pixels
    loss = torch.relu(sigma_linear - wavelength_limit).mean()

    return loss


def fast_gaussian_blur(img, kernel_size, sigma):
    """
    Optimized Gaussian blur that downsamples for large sigma values to save computation.
    Includes safety checks to prevent kernel size from exceeding image dimensions.
    """
    if sigma < 0.01:
        return img

    B, C, H, W = img.shape
    
    # Safety: If sigma is extremely large, return global average
    if sigma > max(H, W):
        return img.mean(dim=(2,3), keepdim=True).expand(-1, -1, H, W)

    if kernel_size is None:
        kernel_size = int(2 * 4 * sigma + 1) | 1

    if sigma <= 2.0:
        # Safety: kernel size cannot exceed image dimensions for reflect padding
        kernel_size = min(kernel_size, (H - 1) * 2 - 1, (W - 1) * 2 - 1)
        if kernel_size < 3:
            return img
        return torchvision.transforms.functional.gaussian_blur(img, kernel_size, sigma)
    
    # For large sigma, we can blur a downsampled version and then upsample.
    scale = int(sigma)
    scale = min(scale, H // 4, W // 4)
    
    if scale <= 1:
        kernel_size = min(kernel_size, (H - 1) * 2 - 1, (W - 1) * 2 - 1)
        if kernel_size < 3:
            return img
        return torchvision.transforms.functional.gaussian_blur(img, kernel_size, sigma)
    
    # Target resolution
    target_h, target_w = max(2, H // scale), max(2, W // scale)
    
    # Downsample
    img_down = F.interpolate(img, size=(target_h, target_w), mode='bilinear', align_corners=False)
    
    # Blurred Sigma in downsampled space
    sigma_down = sigma / (H / target_h) # Use actual scale factors
    k_size_down = int(2 * 4 * sigma_down + 1) | 1
    
    # Safety cap on downsampled kernel
    k_size_down = min(k_size_down, (target_h - 1) * 2 - 1, (target_w - 1) * 2 - 1)
    
    # Blur
    if k_size_down >= 3:
        img_blur_down = torchvision.transforms.functional.gaussian_blur(img_down, k_size_down, sigma_down)
    else:
        # If still too small, the image is basically just the blurred version already
        img_blur_down = img_down
    
    # Upsample back
    return F.interpolate(img_blur_down, size=(H, W), mode='bilinear', align_corners=False)

def get_structure_tensor_torch(image_tensor, sigma=1.0, rho=1.0): 
    """
    Computes Multi-Channel Structure Tensor (Di Zenzo's Method).
    Summing energy across RGB channels avoids missing iso-luminant edges.
    """
    B, C, H, W = image_tensor.shape
    
    # 1. Gaussian Blur (Pre-smoothing)
    # Important: Apply to all channels (RGB) independently
    k_size = int(2 * 4 * sigma + 1)
    if k_size % 2 == 0: k_size += 1
    
    # blur applies to all channels automatically if shape is (B, C, H, W)
    img_smooth = fast_gaussian_blur(image_tensor, k_size, sigma)

    # 2. Sobel Derivatives (Vectorized for RGB)
    # We repeat the kernel for each channel and use groups=C to keep channels separate
    sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float, device=image_tensor.device).view(1,1,3,3)
    sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float, device=image_tensor.device).view(1,1,3,3)
    
    # Repeat kernel for C channels (e.g., 3 for RGB)
    weight_x = sobel_x_kernel.repeat(C, 1, 1, 1)
    weight_y = sobel_y_kernel.repeat(C, 1, 1, 1)

    # Ix, Iy will be shape (B, C, H, W)
    Ix = F.conv2d(img_smooth, weight_x, padding=1, groups=C)
    Iy = F.conv2d(img_smooth, weight_y, padding=1, groups=C)

    # 3. Compute Products per Channel
    Ixx_c = Ix**2
    Ixy_c = Ix*Iy
    Iyy_c = Iy**2

    # 4. SUM ACROSS CHANNELS (Di Zenzo's Method)
    # We sum the energy of R, G, and B. 
    # Result is (B, 1, H, W)
    Ixx = Ixx_c.sum(dim=1, keepdim=True)
    Ixy = Ixy_c.sum(dim=1, keepdim=True)
    Iyy = Iyy_c.sum(dim=1, keepdim=True)

    # 5. Window Integration (Structure Tensor smoothing)
    # Now we smooth the summed energy
    k_size_rho = int(2 * 4 * rho + 1)
    if k_size_rho % 2 == 0: k_size_rho += 1
    
    Sxx = fast_gaussian_blur(Ixx, k_size_rho, rho)
    Sxy = fast_gaussian_blur(Ixy, k_size_rho, rho)
    Syy = fast_gaussian_blur(Iyy, k_size_rho, rho)

    # 6. Normalize
    # Normalizing by 9.0 (typical Sobel magnitude scaling) + max value is usually good practice
    # so thresholds don't drift wildly with brightness.
    magnitude = Sxx + Syy
    max_val = torch.amax(magnitude, dim=(1, 2, 3), keepdim=True) + 1e-6 # Use batch-wise maximum
    
    Sxx = Sxx / max_val
    Sxy = Sxy / max_val
    Syy = Syy / max_val

    # Output: (B, 3, H, W) where channels are Sxx, Sxy, Syy
    return torch.cat([Sxx, Sxy, Syy], dim=1)

def get_multiscale_structure_tensor_v1(image_tensor, levels=3, base_sigma=1., octave_step=1.5, smoothing_factor=1.0, power_factor=3.0, aggregation_mode='average'): 
    """
    Computes an Improved Multi-scale Structure Tensor.
    Analyzes the image at multiple scales (octaves) to determine the dominant frequency.
    """
    B, C, H, W = image_tensor.shape
    device = image_tensor.device
    
    # 1. Compute Base Orientation (finest scale)
    base_st = get_structure_tensor_torch(image_tensor, sigma=base_sigma)
    Sxx_base = base_st[:, 0:1]
    Sxy_base = base_st[:, 1:2]
    Syy_base = base_st[:, 2:3]

    # 2. Analyze Frequency Response and Aggregate Component-wise
    import math
    current_smooth = image_tensor
    
    accum_Sxx = torch.zeros((B, 1, H, W), device=device)
    accum_Sxy = torch.zeros((B, 1, H, W), device=device)
    accum_Syy = torch.zeros((B, 1, H, W), device=device)
    accum_weight = torch.zeros((B, 1, H, W), device=device)
    
    sigma_accum = 0.0
    
    # We use a lower power factor to allow more "mixing" of scales
    # This helps curved structures (captured at one scale) to influence 
    # the anisotropy of even the sharpest edges.
    # power_factor = 2.0  <-- REMOVED override to respect argument (default 3.0) 
    
    for i in range(levels):
        band_freq = 1.0 / (octave_step ** i)
        target_sigma = base_sigma * (octave_step ** i)
        
        sigma_inc = math.sqrt(max(1e-6, target_sigma**2 - sigma_accum**2))
        next_smooth = fast_gaussian_blur(current_smooth, None, sigma_inc)
        
        # Band Response (Difference of Gaussians)
        band_response = (current_smooth - next_smooth).pow(2).sum(dim=1, keepdim=True).sqrt()
        
        # Spatial Smoothing for Low-Frequency Context
        if i > 0 and smoothing_factor > 0:
            smoothing_sigma = target_sigma * 2.0
            band_response = fast_gaussian_blur(band_response, None, smoothing_sigma)
        
        # Integrate structural components at this scale
        # Larger rho (3.0 * sigma) is critical for seeing curvature
        integration_rho = target_sigma * 3.0 
        k_size_rho = int(2 * 4 * integration_rho + 1) | 1
        
        Sxx_i = fast_gaussian_blur(Sxx_base, k_size_rho, integration_rho)
        Sxy_i = fast_gaussian_blur(Sxy_base, k_size_rho, integration_rho)
        Syy_i = fast_gaussian_blur(Syy_base, k_size_rho, integration_rho)
        
        # Normalize to orientation-only
        trace_i = Sxx_i + Syy_i + 1e-6
        # Apply orientation component at this scale
        Sxx_norm_i = Sxx_i / trace_i
        Sxy_norm_i = Sxy_i / trace_i
        Syy_norm_i = Syy_i / trace_i
        
        # Weight by band energy
        weight_i = band_response.pow(power_factor)
        
        # We want the final sum to have Trace = freq^2
        # freq_i = band_freq
        target_wavelength_sq_inv = band_freq**2
        
        accum_Sxx += Sxx_norm_i * weight_i * target_wavelength_sq_inv
        accum_Sxy += Sxy_norm_i * weight_i * target_wavelength_sq_inv
        accum_Syy += Syy_norm_i * weight_i * target_wavelength_sq_inv
        accum_weight += weight_i
        
        current_smooth = next_smooth
        sigma_accum = target_sigma
        
    # Final normalization
    final_Sxx = accum_Sxx / (accum_weight + 1e-6)
    final_Sxy = accum_Sxy / (accum_weight + 1e-6)
    final_Syy = accum_Syy / (accum_weight + 1e-6)
    
    return torch.cat([final_Sxx, final_Sxy, final_Syy], dim=1)

def get_multiscale_structure_tensor_v2(image_tensor, levels=3, base_sigma=1., octave_step=1.5, smoothing_factor=1.0, power_factor=3.0, aggregation_mode='average'): 
    """
    Computes a True Multi-scale Structure Tensor.
    Unlike v1, this version computes the structure tensor at each scale level
    based on the blurred image at that level, rather than blurring a pre-computed
    base tensor. This properly captures orientation and frequency at each scale.
    """
    B, C, H, W = image_tensor.shape
    device = image_tensor.device
    
    import math
    current_smooth = image_tensor
    
    accum_Sxx = torch.zeros((B, 1, H, W), device=device)
    accum_Sxy = torch.zeros((B, 1, H, W), device=device)
    accum_Syy = torch.zeros((B, 1, H, W), device=device)
    accum_weight = torch.zeros((B, 1, H, W), device=device)
    
    sigma_accum = 0.0
    
    for i in range(levels):
        band_freq = 1.0 / (octave_step ** i)
        target_sigma = base_sigma * (octave_step ** i)
        
        # Progressively blur the image to reach target_sigma
        sigma_inc = math.sqrt(max(1e-6, target_sigma**2 - sigma_accum**2))
        next_smooth = fast_gaussian_blur(current_smooth, None, sigma_inc)
        
        # Band Response (Difference of Gaussians) - measures energy at this scale
        band_response = (current_smooth - next_smooth).pow(2).sum(dim=1, keepdim=True).sqrt()
        
        # Spatial Smoothing for Low-Frequency Context
        if i > 0 and smoothing_factor > 0:
            smoothing_sigma = target_sigma * 2.0
            band_response = fast_gaussian_blur(band_response, None, smoothing_sigma)
        
        # TRUE MULTI-SCALE: Compute structure tensor on the blurred image at this level
        # Use appropriate sigma and rho for this scale
        scale_sigma = target_sigma
        integration_rho = target_sigma * 3.0  # Larger rho is critical for seeing curvature
        
        # Compute structure tensor directly on the current smoothed image
        st_i = get_structure_tensor_torch(next_smooth, sigma=scale_sigma, rho=integration_rho)
        Sxx_i = st_i[:, 0:1]
        Sxy_i = st_i[:, 1:2]
        Syy_i = st_i[:, 2:3]
        
        # Normalize to orientation-only
        trace_i = Sxx_i + Syy_i + 1e-6
        Sxx_norm_i = Sxx_i / trace_i
        Sxy_norm_i = Sxy_i / trace_i
        Syy_norm_i = Syy_i / trace_i
        
        # Weight by band energy
        weight_i = band_response.pow(power_factor)
        
        # We want the final sum to have Trace = freq^2
        target_wavelength_sq_inv = band_freq**2
        
        accum_Sxx += Sxx_norm_i * weight_i * target_wavelength_sq_inv
        accum_Sxy += Sxy_norm_i * weight_i * target_wavelength_sq_inv
        accum_Syy += Syy_norm_i * weight_i * target_wavelength_sq_inv
        accum_weight += weight_i
        
        current_smooth = next_smooth
        sigma_accum = target_sigma
        
    # Final normalization
    final_Sxx = accum_Sxx / (accum_weight + 1e-6)
    final_Sxy = accum_Sxy / (accum_weight + 1e-6)
    final_Syy = accum_Syy / (accum_weight + 1e-6)
    
    return torch.cat([final_Sxx, final_Sxy, final_Syy], dim=1)
    
def frequency_loss_simple(rendered_image, st_map):
    pred_st = get_structure_tensor_torch(rendered_image.unsqueeze(0)) 
    loss= l1_loss(pred_st, st_map)  
    return loss

def estimate_required_gaussians(image, base_density=0.01, detail_sensitivity=0.5):
    """
    Estimates the theoretical number of Gaussians needed for a single view.
    
    Args:
        image: (B, C, H, W) input image tensor (0-1 range).
        base_density: Minimum GS per pixel for flat regions (0.01 = 1 GS per 10x10 block).
        detail_sensitivity: Scaling factor for frequency contribution.
        
    Returns:
        estimated_count: Integer estimation of GS count.
    """
    B, C, H, W = image.shape
    total_pixels = H * W
    
    # 1. Compute Structure Tensor (B, 3, H, W) -> channels are Sxx, Sxy, Syy
    # This uses the multiscale function
    st_map = get_multiscale_structure_tensor(image)
    
    # 2. Extract Energy (Trace = Sxx + Syy)
    # Sxx is at index 0, Syy is at index 2
    Sxx = st_map[:, 0, :, :]
    Syy = st_map[:, 2, :, :]
    
    # The trace represents the squared magnitude of local variations.
    # Higher trace = Higher Frequency = Needs more primitives.
    local_energy = Sxx + Syy
    
    # 3. Integrate Energy
    # Sum of energy across the image
    total_structure_energy = local_energy.sum().item()
    
    # 4. Apply Heuristic Formula
    # Base count: Even flat walls need *some* Gaussians to exist (coverage).
    count_coverage = total_pixels * base_density
    
    # Detail count: Extra Gaussians needed for edges/textures.
    # Since st_map is normalized to ~1.0 max, 
    # we assume max energy requires ~1.0 GS/pixel scaling.
    count_detail = total_structure_energy * detail_sensitivity
    
    estimated_count = int(count_coverage + count_detail)
    
    return estimated_count, count_coverage, count_detail

