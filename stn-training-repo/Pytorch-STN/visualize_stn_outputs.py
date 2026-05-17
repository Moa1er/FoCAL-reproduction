import os
import sys
import torch
from torchvision.utils import save_image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

import utils
from dataloader_utils import load_dataset
from models.SVHNet import STNSVHNet


def main():
    PARAM_PATH = "experiments/stn_svhn"
    CKPT_PATH = os.path.join(PARAM_PATH, "best.pth.tar")
    OUT_DIR = "stn_visual_outputs"
    NUM_IMAGES = 16

    os.makedirs(OUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    params = utils.ParamParser(os.path.join(PARAM_PATH, "params.json"))
    params.cuda = torch.cuda.is_available()

    # Important on Windows: avoid multiprocessing DataLoader in this quick viz script
    params.num_workers = 0

    model = STNSVHNet(
        spatial_dim=(params.height, params.width),
        in_channels=params.initial_channel,
        stn_kernel_size=params.stn_kernel_size,
        kernel_size=params.kernel_size,
        use_dropout=True,
    ).to(device)

    checkpoint = torch.load(CKPT_PATH, map_location=device)

    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("loaded checkpoint:", CKPT_PATH)

    val_loader = load_dataset("val", "cifar", params)

    images, labels = next(iter(val_loader))
    images = images[:NUM_IMAGES].to(device)
    labels = labels[:NUM_IMAGES]

    with torch.no_grad():
        stn_images, affine_grid = model.stnmod(images)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()

    comparison = torch.cat([images.cpu(), stn_images.cpu()], dim=0)

    save_path = os.path.join(OUT_DIR, "original_vs_stn.png")
    save_image(
        comparison,
        save_path,
        nrow=NUM_IMAGES,
        normalize=True,
    )

    stn_only_path = os.path.join(OUT_DIR, "stn_only.png")
    save_image(
        stn_images.cpu(),
        stn_only_path,
        nrow=NUM_IMAGES,
        normalize=True,
    )

    print("saved:", save_path)
    print("saved:", stn_only_path)
    print("labels:", labels.tolist())
    print("preds: ", preds.tolist())


if __name__ == "__main__":
    main()