import torch
import numpy as np
from dataclasses import dataclass
from typing import Literal
from torch.distributions import Categorical
from transformers import AutoModel


@dataclass(frozen=True)
class DiffusionEnergyArgs:
    """Arguments for diffusion energy"""

    size: Literal[256, 512] = 256
    """Image size for diffusion model"""

    model: str = "stabilityai/stable-diffusion-2-base"
    """Diffusion model identifier"""

    step_min: int = 500
    """Minimum step for diffusion"""

    step_max: int = 1000
    """Maximum step for diffusion"""

    step_stride: int = 100
    """Step stride for diffusion"""

    batchsize: int = 10
    """Batch size for diffusion processing"""

    factor: float = 0
    """Weight for diffusion energy in total energy calculation"""


@dataclass(frozen=True)
class CLIPEnergyArgs:
    """Arguments for classifier energy"""

    logit_top_factor: float = -0.5
    """Weight for top logit in classifier energy"""

    factor: float = 0.0
    """Weight for classifier energy in total energy calculation"""

    use_entropy: bool = False
    """Whether to use Shannon entropy instead of raw logit heuristic"""

    temperature: float = 100.0
    """Temperature scaling for entropy calculation"""


@dataclass(frozen=True)
class CLIPEnergyNormPromptArgs:
    """Arguments for classifier energy"""

    logit_top_factor: float = -0.5
    """Weight for top logit in classifier energy"""

    factor: float = 1.0
    """Weight for classifier energy in total energy calculation"""

    normalizing_prompt_factor: float = 0.5
    """Weight for normalizing prompt in classifier energy"""

    use_normalizing_prompt: bool = True
    """Whether to use normalizing prompt for classifier"""

    normalizing_prompt: str = "a photo of an object on a bright white backdrop."
    """Normalizing prompt for classifier"""

@dataclass(frozen=True)
class DINOEnergyArgs:
    """Arguments for DINO feature variance energy"""
    factor: float = 1.0
    """Weight for DINO energy in total calculation"""
    
    model_id: str = "facebook/dinov2-base"
    """HuggingFace model identifier"""

def diff_energy(
    rot_ims: torch.Tensor, diffusion_model, args: DiffusionEnergyArgs
) -> np.ndarray:
    """Calculate diffusion energy for rotated images.

    Args:
        rot_ims: Batch of transformed images (N, 3, H, W)
        diffusion_model: Diffusion model to use
        args: Diffusion energy arguments

    Returns:
        Array of diffusion energy scores (N,)
    """
    assert rot_ims.dim() == 4, "Input images must be a 4D tensor (N, C, H, W)"
    assert rot_ims.size(1) == 3, (
        f"Input images must have 3 channels (RGB), got shape {rot_ims.shape}"
    )

    with torch.inference_mode():
        return diffusion_model.score(
            rot_ims.to(diffusion_model.device).half(),
            step_min=args.step_min,
            step_max=args.step_max,
            step_stride=args.step_stride,
            batchsize=args.batchsize,
        )


def uncond_clip_energy(
    rot_ims: torch.Tensor,
    clip_model,
    args: CLIPEnergyArgs,
) -> np.ndarray:
    """Calculate unconditional classification energy.

    Args:
        rot_ims: Batch of transformed images (N, 3, H, W)
        clip_model: CLIP model
        args: CLIP energy arguments

    Returns:
        Array of classification energy scores (N,)
    """
    assert rot_ims.dim() == 4, "Input images must be a 4D tensor (N, C, H, W)"
    assert rot_ims.size(1) == 3, (
        f"Input images must have 3 channels (RGB), got shape {rot_ims.shape}"
    )

    with torch.inference_mode():
        logits = clip_model(rot_ims)
        logits_mean = logits.mean(dim=-1).flatten().cpu().numpy()
        logits_max = logits.max(dim=-1).values.flatten().cpu().numpy()
        return logits_mean + logits_max * args.logit_top_factor


def uncond_clip_energy_norm_prompt(
    rot_ims: torch.Tensor,
    clip_model,
    args: CLIPEnergyArgs,
) -> float:
    """Calculate unconditional classification energy.

    Args:
        rot_ims: Batch of transformed images (N, 3, H, W)
        clip_model: CLIP model
        args: CLIP energy arguments

    Returns:
        Array of classification energy scores (N,)
    """
    assert rot_ims.dim() == 4, "Input images must be a 4D tensor (N, C, H, W)"
    assert rot_ims.size(1) == 3, (
        f"Input images must have 3 channels (RGB), got shape {rot_ims.shape}"
    )

    with torch.inference_mode():
        logits = clip_model(rot_ims)

        if args.use_normalizing_prompt and args.normalizing_prompt_factor > 0:
            normalizing_logits = logits[..., -1:]
            logits = logits[..., :-1] / (
                normalizing_logits**args.normalizing_prompt_factor
            )

        logits_mean = logits.mean(dim=-1).flatten().cpu().numpy()
        logits_max = logits.max(dim=-1).values.flatten().cpu().numpy()
        return logits_mean + logits_max * args.logit_top_factor

def entropy_clip_energy(
    rot_ims: torch.Tensor,
    clip_model,
    args: CLIPEnergyArgs,
) -> np.ndarray:
    """Calculate Shannon entropy of the classification distribution."""
    assert rot_ims.dim() == 4, "Input images must be a 4D tensor (N, C, H, W)"
    assert rot_ims.size(1) == 3, (
        f"Input images must have 3 channels (RGB), got shape {rot_ims.shape}"
    )

    with torch.inference_mode():
        # Get raw logits
        logits = clip_model(rot_ims)

        # Categorical natively handles softmax and log-sum-exp stabilization.
        # Scale by temperature to prevent distribution collapse to a one-hot vector.
        entropy = Categorical(logits=logits / args.temperature).entropy()

        return entropy.cpu().numpy()


def dino_variance_energy(
    rot_ims: torch.Tensor,
    dino_model: torch.nn.Module,
    args: DINOEnergyArgs,
) -> np.ndarray:
    """Calculate negative spatial variance of DINO patch tokens."""
    assert rot_ims.dim() == 4
    
    with torch.inference_mode():
        # rot_ims shape: [8, 3, 224, 224]
        # output.last_hidden_state shape: [8, 257, 768] (1 CLS + 256 patches)
        outputs = dino_model(rot_ims)
        patch_tokens = outputs.last_hidden_state[:, 1:, :] 
        
        # Calculate variance across the 256 patches (dim=1), then mean across the 768 features (dim=1)
        # Higher variance = canonical view. We return negative variance to minimize energy.
        variance = patch_tokens.var(dim=1).mean(dim=1)
        energy = -variance
        
        return energy.cpu().numpy()
