from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class STNConfig:
    """arguments for optionally adding an STN output to the FoCAL candidate set."""

    enabled: bool = False
    """if true, append STN(rotated_image) as an extra alignment candidate."""

    ckpt: Optional[str] = None
    """path to a trained STN checkpoint. required when enabled=True."""

    input_size: int = 32
    """spatial size expected by the CIFAR STN. the old STN code is CIFAR-style 32x32."""

    kernel_size: int = 3
    """kernel size used by the STN localization network."""


class SpatialTransformerModule(nn.Module):
    """stn module compatible with the aicaffeinelife/Pytorch-STN-style CIFAR STN."""

    def __init__(self, in_channels: int = 3, spatial_dims=(32, 32), kernel_size: int = 3):
        super().__init__()

        self._h, self._w = spatial_dims
        self._in_ch = in_channels
        self._ksize = kernel_size

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)

        pooled_h = self._h // 8
        pooled_w = self._w // 8
        self.fc1 = nn.Linear(32 * pooled_h * pooled_w, 1024)
        self.fc2 = nn.Linear(1024, 6)

        self._init_identity()

    def _init_identity(self) -> None:
        """initialize the final affine layer to identity."""
        nn.init.zeros_(self.fc2.weight)
        self.fc2.bias.data.copy_(
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=self.fc2.bias.dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_images = x

        # this intentionally follows the old STN code path:
        # conv1 -> conv2 -> pool -> conv3 -> pool -> conv3 -> pool.
        # conv4 exists for checkpoint compatibility but was not used in the original forward.
        z = F.relu(self.conv1(x.detach()))
        z = F.relu(self.conv2(z))
        z = F.max_pool2d(z, 2)

        z = F.relu(self.conv3(z))
        z = F.max_pool2d(z, 2)

        z = F.relu(self.conv3(z))
        z = F.max_pool2d(z, 2)

        z = z.view(z.size(0), -1)
        z = self.fc1(z)
        theta = self.fc2(z).view(-1, 2, 3)

        grid = F.affine_grid(
            theta,
            torch.Size((theta.size(0), self._in_ch, self._h, self._w)),
            align_corners=False,
        )
        warped = F.grid_sample(batch_images, grid, align_corners=False)
        return warped


class STNAlignmentCandidate(nn.Module):
    """wrapper used by FoCAL."""

    def __init__(self, stn: SpatialTransformerModule, input_size: int = 32):
        super().__init__()
        self.stn = stn
        self.input_size = input_size

    def forward(self, im: torch.Tensor) -> torch.Tensor:
        if im.dim() != 4:
            raise ValueError(f"expected image batch of shape (B, C, H, W), got {tuple(im.shape)}")

        original_size = im.shape[-2:]
        small = F.interpolate(
            im,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        warped_small = self.stn(small)
        warped = F.interpolate(
            warped_small,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
        return warped.clamp(0.0, 1.0)


def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {
        k[len(prefix):] if k.startswith(prefix) else k: v
        for k, v in state_dict.items()
    }


def _extract_stn_state_dict(raw_ckpt: Union[Dict, nn.Module]) -> Dict[str, torch.Tensor]:
    """accepts common checkpoint formats."""

    if isinstance(raw_ckpt, nn.Module):
        raw_ckpt = raw_ckpt.state_dict()

    if not isinstance(raw_ckpt, dict):
        raise TypeError(f"unsupported STN checkpoint type: {type(raw_ckpt)}")

    if "stn_state_dict" in raw_ckpt:
        state_dict = raw_ckpt["stn_state_dict"]
    elif "state_dict" in raw_ckpt:
        state_dict = raw_ckpt["state_dict"]
    elif "model_state_dict" in raw_ckpt:
        state_dict = raw_ckpt["model_state_dict"]
    else:
        state_dict = raw_ckpt

    state_dict = _strip_prefix_if_present(state_dict, "module.")

    # full old STN classifier checkpoints usually contain keys like:
    # stnmod.conv1.weight, stnmod.fc1.weight, ...
    stnmod_keys = {
        k.replace("stnmod.", "", 1): v
        for k, v in state_dict.items()
        if k.startswith("stnmod.")
    }
    if len(stnmod_keys) > 0:
        return stnmod_keys

    # already a direct SpatialTransformer state dict:
    # conv1.weight, conv2.weight, fc1.weight, ...
    return state_dict


def initialize_stn_model(args: STNConfig, device: torch.device) -> STNAlignmentCandidate:
    if not args.enabled:
        raise ValueError("initialize_stn_model was called, but args.enabled is False.")

    if args.ckpt is None:
        raise ValueError("STN is enabled, but no STN checkpoint was provided. Use --stn.ckpt PATH.")

    stn = SpatialTransformerModule(
        in_channels=3,
        spatial_dims=(args.input_size, args.input_size),
        kernel_size=args.kernel_size,
    )

    try:
        raw_ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    except Exception:
        # only do this for a trusted checkpoint from your groupmate.
        raw_ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    stn_state_dict = _extract_stn_state_dict(raw_ckpt)

    missing, unexpected = stn.load_state_dict(stn_state_dict, strict=False)

    required_missing = [
        key for key in missing
        if not key.startswith("conv4.")
    ]
    if len(required_missing) > 0:
        raise RuntimeError(
            "the STN checkpoint did not contain all required STN parameters.\n"
            f"missing: {required_missing}\n"
            f"unexpected: {unexpected}\n"
            "expected either direct STN keys like conv1.weight/fc1.weight, or full-model keys like stnmod.conv1.weight."
        )

    model = STNAlignmentCandidate(stn=stn, input_size=args.input_size)
    model.eval().to(device)
    return model