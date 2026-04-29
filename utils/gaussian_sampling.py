"""
Vectorized 2D Gaussian sampling based on structure tensor analysis.
Extracted from fit_2d_image.py for reusability.
"""

import torch
import numpy as np
from utils.loss_utils import get_structure_tensor_torch


def sample_anisotropic_gaussians_2d(image_tensor, num_samples=2000, sigma=1.0, rho=1.5, seed=42):
    """
    Samples 2D Gaussian parameters based on the structure tensor of the image.
    Uses vectorized GPU computation for efficiency.
    
    Args:
        image_tensor: (B, C, H, W) torch tensor, normalized 0-1.
        num_samples: Number of Gaussians to sample.
        sigma: Gaussian blur sigma for derivatives.
        rho: Window size sigma for structure tensor integration.
        seed: Random seed for reproducibility.
        
    Returns:
        Dictionary containing:
            - 'coords': (num_samples, 2) tensor of [x, y] coordinates
            - 'eigenvalues': (num_samples, 2) tensor of [l1, l2] eigenvalues
            - 'eigenvectors': (num_samples, 2, 2) tensor of eigenvectors
            - 'v2': (num_samples, 2) tensor of edge direction vectors
            - 'angles': (num_samples,) tensor of angles in degrees
    """
    B, C, H, W = image_tensor.shape
    device = image_tensor.device
    
    # 1. Compute Structure Tensor (B, 3, H, W)
    st_map = get_structure_tensor_torch(image_tensor, sigma=sigma, rho=rho)
    
    # Extract components (assuming batch size 1 for now)
    Sxx_torch = st_map[0, 0, :, :]
    Sxy_torch = st_map[0, 1, :, :]
    Syy_torch = st_map[0, 2, :, :]
    
    # 2. Importance Map for Sampling (Trace of Tensor ~ Gradient Energy)
    energy_map_torch = Sxx_torch + Syy_torch
    
    # Convert energy map to numpy for sampling logic
    energy_map = energy_map_torch.detach().cpu().numpy()
    
    energy_flat = energy_map.flatten()
    sum_energy = energy_flat.sum()
    if sum_energy > 0:
        prob = energy_flat / sum_energy
    else:
        prob = np.ones_like(energy_flat) / len(energy_flat)
    
    # Sample indices
    rng = np.random.default_rng(seed)
    coords_idx = rng.choice(H * W, size=num_samples, p=prob, replace=False)
    y_coords, x_coords = np.unravel_index(coords_idx, (H, W))
    
    # Convert coordinates to torch tensors
    y_coords_torch = torch.from_numpy(y_coords).long().to(device)
    x_coords_torch = torch.from_numpy(x_coords).long().to(device)
    
    # Vectorized extraction of structure tensor components at sampled locations
    # Shape: (num_samples,)
    sxx = Sxx_torch[y_coords_torch, x_coords_torch]
    sxy = Sxy_torch[y_coords_torch, x_coords_torch]
    syy = Syy_torch[y_coords_torch, x_coords_torch]
    
    # Sanitize values: replace NaN/Inf with zeros and clamp to reasonable range
    sxx = torch.nan_to_num(sxx, nan=0.0, posinf=1e6, neginf=-1e6)
    sxy = torch.nan_to_num(sxy, nan=0.0, posinf=1e6, neginf=-1e6)
    syy = torch.nan_to_num(syy, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure positive semi-definite by adding small epsilon to diagonal
    epsilon = 1e-6
    sxx = sxx + epsilon
    syy = syy + epsilon
    
    # Construct 2x2 structure tensor matrices for all samples
    # Shape: (num_samples, 2, 2)
    S = torch.stack([
        torch.stack([sxx, sxy], dim=-1),
        torch.stack([sxy, syy], dim=-1)
    ], dim=-2)
    
    # Vectorized eigendecomposition for all samples at once
    # eigvals: (num_samples, 2), eigvecs: (num_samples, 2, 2)
    try:
        eigvals, eigvecs = torch.linalg.eigh(S)
    except RuntimeError as e:
        print(f"Warning: Eigendecomposition failed, using fallback. Error: {e}")
        # Fallback: use identity matrices
        eigvals = torch.ones((S.shape[0], 2), device=device)
        eigvecs = torch.eye(2, device=device).unsqueeze(0).expand(S.shape[0], -1, -1)
    
    # Sort eigenvalues in descending order
    # l1 is largest (Gradient direction), l2 is smallest (Edge direction)
    sorted_indices = torch.argsort(eigvals, dim=-1, descending=True)
    
    # Gather sorted eigenvalues
    # Shape: (num_samples,)
    l1 = torch.gather(eigvals, -1, sorted_indices[:, 0:1]).squeeze(-1)
    l2 = torch.gather(eigvals, -1, sorted_indices[:, 1:2]).squeeze(-1)
    
    # Get corresponding eigenvectors
    # v2 corresponds to l2 (edge direction)
    v2_indices = sorted_indices[:, 1:2].unsqueeze(-1).expand(-1, -1, 2)  # (num_samples, 1, 2)
    v2 = torch.gather(eigvecs, 1, v2_indices).squeeze(1)  # (num_samples, 2)
    
    # Calculate angles of the EDGE (v2) - vectorized
    # arctan2(y, x)
    angles = torch.rad2deg(torch.atan2(v2[:, 1], v2[:, 0]))
    
    # Stack coordinates as (num_samples, 2) [x, y]
    coords = torch.stack([x_coords_torch.float(), y_coords_torch.float()], dim=-1)
    
    # Stack eigenvalues as (num_samples, 2) [l1, l2]
    eigenvalues = torch.stack([l1, l2], dim=-1)
    
    return {
        'coords': coords,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigvecs,
        'v2': v2,
        'angles': angles,
        'energy_map': energy_map_torch
    }
