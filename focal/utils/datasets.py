"""dataset utilities and loaders for various image classification tasks."""

import os
import os.path as osp
from typing import Any, List, Optional

import pandas as pd
from torchvision import datasets
from torchvision.datasets import VisionDataset

from .imagenet_classnames import get_classnames

# constants
DATASET_ROOT = "./datasets"
IMAGENET_ROOT = "/home/utkarsh/data/imagenet"
PROMPTS_ROOT = "./focal/prompts"


def fetch_dataset_classes(datasetname: str) -> List[str]:
    """load classes from CSV file."""
    prompt_path = osp.join(PROMPTS_ROOT, f"{datasetname}_prompts.csv")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"prompts file not found: {prompt_path}")
    return list(pd.read_csv(prompt_path)["classname"].values)


def fetch_dataset_prompts(datasetname: str) -> List[str]:
    """load text prompts from CSV file."""
    prompt_path = osp.join(PROMPTS_ROOT, f"{datasetname}_prompts.csv")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"prompts file not found: {prompt_path}")
    return list(pd.read_csv(prompt_path)["prompt"].values)


def initialize_target_dataset(
    name: str,
    train: bool = False,
    transform: Optional[Any] = None,
    target_transform: Optional[Any] = None,
) -> VisionDataset:
    """get dataset by name with consistent attributes."""
    dataset_map = {
        "cifar10": (datasets.CIFAR10, {"train": train}),
        "cifar100": (datasets.CIFAR100, {"train": train}),
        "stl10": (datasets.STL10, {"split": "train" if train else "test"}),
    }

    if name in dataset_map:
        dataset_cls, extra_args = dataset_map[name]
        dataset = dataset_cls(
            root=DATASET_ROOT,
            transform=transform,
            target_transform=target_transform,
            download=True,
            **extra_args,
        )
    elif name == "imagenet":
        if train:
            dataset = datasets.ImageNet(
                root=IMAGENET_ROOT,
                split="train",
                transform=transform,
                target_transform=target_transform,
            )
        else:
            dataset = datasets.ImageFolder(
                root=osp.join(DATASET_ROOT, "./imgnet_testset"),
                transform=transform,
                target_transform=target_transform,
            )
            dataset.class_to_idx = None
            dataset.classes = get_classnames("openai")
            dataset.file_to_class = None
    else:
        raise ValueError(f"dataset {name} not supported")

    # post-processing for specific datasets
    if name == "stl10":
        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.classes)}
    elif name in {"cifar10", "cifar100", "stl10"}:
        dataset.file_to_class = {
            str(idx): dataset[idx][1] for idx in range(len(dataset))
        }

    return dataset
