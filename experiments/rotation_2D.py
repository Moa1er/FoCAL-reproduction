"""
runs 2D rotation experiments with N angles.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode, rotate as torch_rotate
from tqdm import tqdm
import tyro
from dataclasses import dataclass, asdict, field
from transformers import AutoModel

from focal.utils.datasets import initialize_target_dataset, fetch_dataset_prompts
from focal.utils.classifiers import (
    ZeroShotCLIPClassifier,
    ZeroShotSigLIPClassifier,
    DINOv2ClassifierWrapper,
    PretrainedResNet50,
    PretrainedViTB,
    EquivariantResNet50,
    EquivariantViTB,
)
from focal.utils.diffusion import SDLossEstimator
from focal.utils.energy import (
    DiffusionEnergyArgs,
    CLIPEnergyArgs,
    DINOEnergyArgs,
    diff_energy,
    uncond_clip_energy,
    entropy_clip_energy,
    dino_variance_energy,
)
from focal.utils.stn_candidate import STNConfig, initialize_stn_model


@dataclass(frozen=True)
class ExperimentConfig:
    """configuration for C-8 Alignment Classifier experiments.

    this class defines parameters for running rotation experiments where images
    are randomly rotated and then aligned back using various methods.
    """

    num_angles: int = 8
    """number of angles for rotation"""

    dataset: Literal["cifar10", "cifar100", "stl10", "imagenet"] = "cifar10"
    """dataset to use for experiments"""

    model: Literal["clip", "siglip", "resnet", "vitb", "dino", "prlc_r50", "prlc_vit"] = "clip"
    """model architecture for downstream classification"""

    logits_model: Literal["clip", "siglip", "dino"] = "clip"
    """model architecture to calculate unsupervised logit energies from"""

    ckpt: Union[str, None] = None
    """path to the pretrained checkpoint for the model; only used for PRLC models"""

    N: int = -1
    """number of samples to process (-1 for all samples)"""

    diffusion: DiffusionEnergyArgs = DiffusionEnergyArgs()
    """diffusion energy arguments"""

    clip_energy: CLIPEnergyArgs = CLIPEnergyArgs(factor=1.0)
    """classifier energy arguments"""

    dino_energy: DINOEnergyArgs = DINOEnergyArgs(factor=0.0)
    """dino variance energy arguments"""

    stn: STNConfig = field(default_factory=STNConfig)
    """optional STN candidate arguments"""

    seed: int = 0
    """random seed for reproducibility"""

    device: str = "cuda:0"
    """device to run the experiment on"""


def resize_and_crop(im: torch.Tensor, size: int = 224) -> torch.Tensor:
    """resize and center crop an image tensor to the specified size."""
    im = transforms.Resize(size)(im)
    im = transforms.CenterCrop(size)(im)
    return im


def rotate_image(im: torch.Tensor, angle: float) -> torch.Tensor:
    """rotate an image by the given angle."""
    assert im.dim() == 4, "input image must be a 4D tensor (N, C, H, W)"
    assert im.size(1) == 3, (
        f"input image must have 3 channels (RGB), got shape {im.shape}"
    )

    padding = int(
        224 * 0.4
    )  # excessively large padding to ensure no cropping after rotation
    impad = transforms.Pad(padding, padding_mode="edge")(im)
    impadrot = torch_rotate(impad, angle, interpolation=InterpolationMode.BILINEAR)
    return transforms.CenterCrop(224)(impadrot)


def verify_prediction(
    img: torch.Tensor,
    classifier: Union[ZeroShotCLIPClassifier, PretrainedResNet50, PretrainedViTB, DINOv2ClassifierWrapper],
    label: int,
) -> int:
    """evaluate if classifier prediction matches the label."""
    assert img.dim() == 4 and img.size(0) == 1, (
        "input image must be a 4D tensor with batch size 1"
    )
    assert img.size(1) == 3, (
        f"input image must have 3 channels (RGB), got shape {img.shape}"
    )
    pred = classifier(img)
    pred_label = pred.argmax().cpu().item()
    return int(label == pred_label)


def generate_alignment_candidates(
    im: torch.Tensor,
    angles: np.ndarray,
    stn_model: Union[torch.nn.Module, None] = None,
) -> tuple[torch.Tensor, List[Union[float, None]], List[str]]:
    """build the FoCAL candidate set.

    normal FoCAL candidates:
        rotate_image(im, angle) for every angle in C_N

    optional STN candidate:
        stn(im)
    """
    candidate_images = torch.cat(
        [rotate_image(im.clone(), float(angle)) for angle in angles],
        dim=0,
    )
    candidate_angles: List[Union[float, None]] = [float(angle) for angle in angles]
    candidate_names = [f"rot_{float(angle):.1f}" for angle in angles]

    if stn_model is not None:
        stn_candidate = stn_model(im.clone())
        candidate_images = torch.cat([candidate_images, stn_candidate], dim=0)
        candidate_angles.append(None)
        candidate_names.append("stn")

    return candidate_images, candidate_angles, candidate_names


def _mean_numeric(values: List[Any]) -> float:
    numeric_values = [v for v in values if v is not None]
    if len(numeric_values) == 0:
        return float("nan")
    return float(np.mean(numeric_values))


def format_statistics(results: Dict[str, List[Any]]) -> str:
    """generate statistics string from results."""
    stats = [
        f"default: {np.mean(results['correct']) * 100:.1f}%",
        f"rot: {np.mean(results['correct_after_rot']) * 100:.1f}%",
        f"rot+unrot: {np.mean(results['correct_after_rot_and_unrot']) * 100:.1f}%",
        f"rot+realign: {np.mean(results['correct_after_rot_and_realign']) * 100:.1f}%",
        f"upright+realign: {np.mean(results['correct_upright_align']) * 100:.3f}%",
        f"pose acc: {_mean_numeric(results['pose_accuracy']) * 100:.3f}%",
        f"pose dist: {_mean_numeric(results['pose_dist']):.2f}",
    ]

    if "selected_is_stn" in results and len(results["selected_is_stn"]) > 0:
        stats.append(f"stn select: {np.mean(results['selected_is_stn']) * 100:.1f}%")

    if "upright_selected_is_stn" in results and len(results["upright_selected_is_stn"]) > 0:
        stats.append(f"upright stn select: {np.mean(results['upright_selected_is_stn']) * 100:.1f}%")

    return " ".join(stats)


def export_results_to_json(results: Dict[str, Any], args: ExperimentConfig) -> None:
    """save experiment results to JSON file."""
    print("saving results...")
    results["args"] = asdict(args)

    save_dir = Path(f"results/{args.dataset}/cyclicN")
    save_dir.mkdir(parents=True, exist_ok=True)

    stn_suffix = "_with_stn" if args.stn.enabled else ""
    json_filename = f"cyclic_alignment_results_cls{args.model}{stn_suffix}_seed{args.seed}.json"
    save_path = save_dir / json_filename

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"results saved to {save_path}")


def main() -> None:
    """main function to run the rotation experiments."""
    args = tyro.cli(ExperimentConfig)
    print(f"arguments: {args}")

    # setup environment
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    print(f"using device: {device}")

    # load dataset and prompts
    dataset = initialize_target_dataset(args.dataset, transform=transforms.ToTensor())
    prompts = fetch_dataset_prompts(args.dataset)

    # initialize classifier
    classifier_map = {
        "clip": ZeroShotCLIPClassifier,
        "siglip": ZeroShotSigLIPClassifier,
        "resnet": PretrainedResNet50,
        "vitb": PretrainedViTB,
        "dino": DINOv2ClassifierWrapper,
        "prlc_r50": EquivariantResNet50,
        "prlc_vit": EquivariantViTB,
    }
    classifier = classifier_map[args.model](
        prompts, dataset=args.dataset, device=device, ckpt=args.ckpt
    )

    # validate configuration
    use_diffusion = args.diffusion.factor > 0
    use_clip_energy = args.clip_energy.factor > 0
    use_dino_energy = args.dino_energy.factor > 0

    if not (use_diffusion or use_clip_energy or use_dino_energy):
        raise ValueError(
            "need to use at least one alignment method. did you mean to set clip_energy.factor, diffusion.factor, or dino_energy.factor to a positive value?"
        )

    # initialize alignment models
    clip_model = None
    if use_clip_energy:
        if args.logits_model == args.model:
            clip_model = classifier
        elif args.logits_model == "siglip":
            clip_model = ZeroShotSigLIPClassifier(prompts, dataset=args.dataset, device=device)
        elif args.logits_model == "dino":
            clip_model = DINOv2ClassifierWrapper(prompts, dataset=args.dataset, device=device)
        else:
            clip_model = ZeroShotCLIPClassifier(prompts, dataset=args.dataset, device=device)

    stn_model = None
    if args.stn.enabled:
        stn_model = initialize_stn_model(args.stn, device=device)

    diffusion_model = None
    octagon_mask = None
    if use_diffusion:
        diffusion_model = SDLossEstimator(
            model_id=args.diffusion.model, size=args.diffusion.size, device=device
        )
        # note: diffusion energy is sensitive to the padding at the corners from non-90 degree rotations.
        # to mitigate this, we use an octagon mask to mask out the corners of the rotated image.
        octagon_mask = torch.ones((1, 1, 224, 224), dtype=torch.float32, device=device)
        octagon_mask = torch_rotate(
            octagon_mask, 45, interpolation=InterpolationMode.NEAREST
        )

    dino_model = None
    dino_preprocess = None
    if use_dino_energy:
        # load raw base model, send to device, set to eval
        dino_model = AutoModel.from_pretrained(args.dino_energy.model_id)
        dino_model.eval().to(device)
        dino_preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # setup experiment parameters
    num_samples = len(dataset) if args.N == -1 else args.N
    angles = np.linspace(-180, 180, args.num_angles, endpoint=False)
    print(f"angles: {angles}")

    # initialize results dictionary
    results = {
        "correct": [],
        "correct_after_rot": [],
        "correct_after_rot_and_unrot": [],
        "correct_after_rot_and_realign": [],
        "correct_upright_align": [],
        "thetas": [],
        "pose_accuracy": [],
        "pose_dist": [],
        "inferred_angles": [],
        "selected_candidates": [],
        "selected_is_stn": [],
        "upright_inferred_angles": [],
        "upright_selected_candidates": [],
        "upright_selected_is_stn": [],
        "labels": [],
    }

    # run experiments
    with torch.inference_mode():
        pbar = tqdm(range(num_samples), dynamic_ncols=True)
        for idx in pbar:
            theta = np.random.choice(angles)
            results["thetas"].append(theta)

            im, label = dataset[idx]
            im = resize_and_crop(im).to(device).unsqueeze(0)

            results["labels"].append(int(label))

            # default accuracy
            results["correct"].append(verify_prediction(im, classifier, label))

            # rotate image
            im_rot = rotate_image(im.clone(), theta)
            results["correct_after_rot"].append(verify_prediction(im_rot, classifier, label))

            # oracle accuracy (rotate back)
            im_unrot = rotate_image(im_rot.clone(), -theta)
            results["correct_after_rot_and_unrot"].append(
                verify_prediction(im_unrot, classifier, label)
            )

            # process rotated realignment
            if use_diffusion or use_clip_energy or use_dino_energy:
                candidate_ims, candidate_angles, candidate_names = generate_alignment_candidates(
                    im_rot,
                    angles,
                    stn_model=stn_model,
                )

                diff_score = 0
                if use_diffusion and diffusion_model is not None:
                    masked_ims = candidate_ims
                    if octagon_mask is not None:
                        masked_ims = candidate_ims * octagon_mask
                    diff_score = diff_energy(
                        masked_ims, diffusion_model, args.diffusion
                    )

                uncond_cls_score = 0
                if use_clip_energy and clip_model is not None:
                    if hasattr(args.clip_energy, "use_entropy") and args.clip_energy.use_entropy:
                        uncond_cls_score = entropy_clip_energy(
                            candidate_ims, clip_model, args.clip_energy
                        )
                    else:
                        uncond_cls_score = uncond_clip_energy(
                            candidate_ims, clip_model, args.clip_energy
                        )

                dino_score = 0
                if use_dino_energy and dino_model is not None and dino_preprocess is not None:
                    dino_score = dino_variance_energy(
                        dino_preprocess(candidate_ims), dino_model, args.dino_energy
                    )

                final_score = (
                    args.diffusion.factor * diff_score
                    + args.clip_energy.factor * uncond_cls_score
                    + args.dino_energy.factor * dino_score
                )

                best_idx = int(np.argmin(final_score))
                best_angle = candidate_angles[best_idx]
                best_candidate_name = candidate_names[best_idx]
            else:
                raise ValueError("no alignment method specified")

            im_realign = candidate_ims[best_idx : best_idx + 1]
            results["correct_after_rot_and_realign"].append(
                verify_prediction(im_realign, classifier, label)
            )
            results["inferred_angles"].append(best_angle)
            results["selected_candidates"].append(best_candidate_name)
            results["selected_is_stn"].append(int(best_candidate_name == "stn"))

            # calculate pose accuracy and distance
            if best_angle is not None:
                best_angle_remapped = ((180 - best_angle) % 360) - 180
                results["pose_accuracy"].append(
                    int(abs(theta - best_angle_remapped) < 1e-6)
                )

                theta_cossin = np.array(
                    [np.cos(theta * np.pi / 180), np.sin(theta * np.pi / 180)]
                )
                pred_cossin = np.array(
                    [
                        np.cos(best_angle_remapped * np.pi / 180),
                        np.sin(best_angle_remapped * np.pi / 180),
                    ]
                )
                cosine_similarity = (theta_cossin * pred_cossin).sum()
                cosine_similarity = np.clip(cosine_similarity, -1, 1)
                results["pose_dist"].append(float(np.rad2deg(np.arccos(cosine_similarity))))
            else:
                results["pose_accuracy"].append(None)
                results["pose_dist"].append(None)

            # process upright alignment
            if use_diffusion or use_clip_energy or use_dino_energy:
                upright_candidate_ims, upright_candidate_angles, upright_candidate_names = generate_alignment_candidates(
                    im,
                    angles,
                    stn_model=stn_model,
                )

                diff_score = 0
                if use_diffusion and diffusion_model is not None:
                    masked_ims = upright_candidate_ims
                    if octagon_mask is not None:
                        masked_ims = upright_candidate_ims * octagon_mask
                    diff_score = diff_energy(
                        masked_ims, diffusion_model, args.diffusion
                    )

                uncond_cls_score = 0
                if use_clip_energy and clip_model is not None:
                    if hasattr(args.clip_energy, "use_entropy") and args.clip_energy.use_entropy:
                        uncond_cls_score = entropy_clip_energy(
                            upright_candidate_ims, clip_model, args.clip_energy
                        )
                    else:
                        uncond_cls_score = uncond_clip_energy(
                            upright_candidate_ims, clip_model, args.clip_energy
                        )

                dino_score = 0
                if use_dino_energy and dino_model is not None and dino_preprocess is not None:
                    dino_score = dino_variance_energy(
                        dino_preprocess(upright_candidate_ims), dino_model, args.dino_energy
                    )

                final_score = (
                    args.diffusion.factor * diff_score
                    + args.clip_energy.factor * uncond_cls_score
                    + args.dino_energy.factor * dino_score
                )

                upright_best_idx = int(np.argmin(final_score))
                upright_best_angle = upright_candidate_angles[upright_best_idx]
                upright_best_candidate_name = upright_candidate_names[upright_best_idx]
            else:
                raise ValueError("no alignment method specified")

            im_upright_realign = upright_candidate_ims[upright_best_idx : upright_best_idx + 1]
            results["correct_upright_align"].append(
                verify_prediction(im_upright_realign, classifier, label)
            )
            results["upright_inferred_angles"].append(upright_best_angle)
            results["upright_selected_candidates"].append(upright_best_candidate_name)
            results["upright_selected_is_stn"].append(int(upright_best_candidate_name == "stn"))

            # update progress bar
            pbar.set_postfix_str(format_statistics(results))

    # print and save final results
    print(format_statistics(results))
    export_results_to_json(results, args)


if __name__ == "__main__":
    main()