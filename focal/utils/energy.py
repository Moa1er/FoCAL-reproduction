import torch
import numpy as np
from dataclasses import dataclass
from typing import Literal
from torch.distributions import Categorical


@dataclass(frozen=True)
class DiffusionEnergyArgs:
    """arguments for diffusion energy"""

    size: Literal[256, 512] = 256
    """image size for diffusion model"""

    model: str = "stabilityai/stable-diffusion-2-base"
    """diffusion model identifier"""

    step_min: int = 500
    """minimum step for diffusion"""

    step_max: int = 1000
    """maximum step for diffusion"""

    step_stride: int = 100
    """step stride for diffusion"""

    batchsize: int = 10
    """batch size for diffusion processing"""

    factor: float = 0
    """weight for diffusion energy in total energy calculation"""


@dataclass(frozen=True)
class CLIPEnergyArgs:
    """arguments for classifier energy"""

    logit_top_factor: float = -0.5
    """weight for top logit in classifier energy"""

    factor: float = 0.0
    """weight for classifier energy in total energy calculation"""

    use_entropy: bool = False
    """whether to use shannon entropy instead of raw logit heuristic"""

    temperature: float = 100.0
    """temperature scaling for entropy calculation"""


@dataclass(frozen=True)
class DINOEnergyArgs:
    """arguments for DINO feature variance energy"""
    factor: float = 1.0
    """weight for DINO energy in total calculation"""
    
    model_id: str = "facebook/dinov2-base"
    """huggingface model identifier"""


def diff_energy(
    rot_ims: torch.Tensor, diffusion_model, args: DiffusionEnergyArgs
) -> np.ndarray:
    """calculate diffusion energy for rotated images."""
    assert rot_ims.dim() == 4, "input images must be a 4D tensor (N, C, H, W)"
    assert rot_ims.size(1) == 3, (
        f"input images must have 3 channels (RGB), got shape {rot_ims.shape}"
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
    """calculate unconditional classification energy."""
    assert rot_ims.dim() == 4, "input images must be a 4D tensor (N, C, H, W)"
    assert rot_ims.size(1) == 3, (
        f"input images must have 3 channels (RGB), got shape {rot_ims.shape}"
    )

    with torch.inference_mode():
        logits = clip_model(rot_ims)
        logits_mean = logits.mean(dim=-1).flatten().cpu().numpy()
        logits_max = logits.max(dim=-1).values.flatten().cpu().numpy()
        return logits_mean + logits_max * args.logit_top_factor


def entropy_clip_energy(
    rot_ims: torch.Tensor,
    clip_model,
    args: CLIPEnergyArgs,
) -> np.ndarray:
    """calculate shannon entropy of the classification distribution."""
    assert rot_ims.dim() == 4, "input images must be a 4D tensor (N, C, H, W)"
    assert rot_ims.size(1) == 3, (
        f"input images must have 3 channels (RGB), got shape {rot_ims.shape}"
    )

    with torch.inference_mode():
        # get raw logits
        logits = clip_model(rot_ims)

        # categorical natively handles softmax and log-sum-exp stabilization.
        # scale by temperature to prevent distribution collapse to a one-hot vector.
        entropy = Categorical(logits=logits / args.temperature).entropy()

        return entropy.cpu().numpy()


def dino_variance_energy(
    rot_ims: torch.Tensor,
    dino_model: torch.nn.Module,
    args: DINOEnergyArgs,
) -> np.ndarray:
    """calculate negative spatial variance of DINO patch tokens."""
    assert rot_ims.dim() == 4
    
    with torch.inference_mode():
        # rot_ims shape: [8, 3, 224, 224]
        # output.last_hidden_state shape: [8, 257, 768] (1 CLS + 256 patches)
        outputs = dino_model(rot_ims)
        patch_tokens = outputs.last_hidden_state[:, 1:, :] 
        
        # calculate variance across the 256 patches (dim=1), then mean across the 768 features (dim=1)
        # higher variance = canonical view. we return negative variance to minimize energy.
        variance = patch_tokens.var(dim=1).mean(dim=1)
        energy = -variance
        
        return energy.cpu().numpy()
