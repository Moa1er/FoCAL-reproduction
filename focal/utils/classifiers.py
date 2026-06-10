"""image classification models for 2D rotation tasks."""

from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from open_clip import create_model_and_transforms, get_tokenizer
from transformers import AutoModelForImageClassification
from focal.utils.equiadapt_classifier_utils import setup_prediction_network


class ZeroShotCLIPClassifier:
    """clip-based image classifier."""

    def __init__(
        self, prompts: List[str], device: str = "cuda", *args, **kwargs
    ) -> None:
        """initialize clip classifier."""
        self.device = device
        self.clip_model, self.clip_preprocess, self.clip_tokenizer = (
            self._setup_clip_model()
        )
        self.text_enc = self._encode_text(self.clip_model, self.clip_tokenizer, prompts)

    def __call__(self, im: torch.Tensor) -> torch.Tensor:
        """classify input image."""
        return self._classifier(
            im, self.clip_model, self.clip_preprocess, self.text_enc
        )

    def _setup_clip_model(self) -> Tuple[torch.nn.Module, transforms.Compose, object]:
        """setup clip model components."""
        clip_preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        clip_model, _, _ = create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        clip_model.eval().to(self.device)
        clip_tokenizer = get_tokenizer("ViT-H-14")
        return clip_model, clip_preprocess, clip_tokenizer

    def _encode_text(
        self, clip_model: torch.nn.Module, clip_tokenizer: object, prompts: List[str]
    ) -> torch.Tensor:
        """encode text prompts using clip model."""
        with torch.inference_mode():
            batchsize = 100
            text_enc = torch.cat(
                [
                    clip_model.encode_text(
                        clip_tokenizer(prompts[i * batchsize : (i + 1) * batchsize]).to(
                            self.device
                        )
                    ).cpu()
                    for i in range(len(prompts) // batchsize + 1)
                ]
            )
            return F.normalize(text_enc, dim=-1)

    def _encode_im(
        self,
        im: torch.Tensor,
        clip_model: torch.nn.Module,
        clip_preprocess: transforms.Compose,
    ) -> torch.Tensor:
        """encode image using clip model (batched)."""
        if len(im.shape) == 3:
            im = im.unsqueeze(0)

        # apply preprocessing natively across the batch of rotations to avoid sequential bottlenecks.
        im_processed = clip_preprocess(im).to(self.device)
        return clip_model.encode_image(im_processed).cpu()

    def _classifier(
        self,
        im: torch.Tensor,
        clip_model: torch.nn.Module,
        clip_preprocess: transforms.Compose,
        text_enc: torch.Tensor,
    ) -> torch.Tensor:
        """classify image using clip model."""
        with torch.inference_mode():
            im_enc = self._encode_im(im, clip_model, clip_preprocess)
            im_enc = F.normalize(im_enc, dim=-1)
            return im_enc @ text_enc.T


class ZeroShotSigLIPClassifier:
    """siglip-based image classifier."""

    def __init__(
        self, prompts: List[str], device: str = "cuda", *args, **kwargs
    ) -> None:
        """initialize siglip classifier."""
        from transformers import AutoProcessor, AutoModel
        
        self.device = device
        model_name = "google/siglip-base-patch16-224"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(device)

        # encode text prompts once
        with torch.inference_mode():
            self.text_inputs = self.processor(text=prompts, return_tensors="pt", padding="max_length", truncation=True).to(device)
            out = self.model.get_text_features(**self.text_inputs)
            self.text_embeds = out.pooler_output if hasattr(out, 'pooler_output') else out[1]
            self.text_embeds = F.normalize(self.text_embeds, dim=-1)

    def __call__(self, im: torch.Tensor) -> torch.Tensor:
        """classify input image."""
        if len(im.shape) == 3:
            im = im.unsqueeze(0)
            
        with torch.inference_mode():
            # convert to uint8 for processor
            im_uint8 = [(im_ * 255).clamp(0, 255).to(torch.uint8) for im_ in im]
            inputs = self.processor(images=im_uint8, return_tensors="pt").to(self.device)
            
            out_img = self.model.get_image_features(**inputs)
            image_embeds = out_img.pooler_output if hasattr(out_img, 'pooler_output') else out_img[1]
            image_embeds = F.normalize(image_embeds, dim=-1)
            
            # siglip logits: (image_embeds @ text_embeds.T) * logit_scale + logit_bias
            scale = self.model.logit_scale.exp()
            bias = self.model.logit_bias
            logits = (image_embeds @ self.text_embeds.T) * scale + bias
            
        return logits


class DINOv2ClassifierWrapper:
    """dinov2 classifier wrapper."""

    def __init__(self, prompts, device: str = "cuda", *args, **kwargs) -> None:
        self.device = device
        self.model = AutoModelForImageClassification.from_pretrained(
            "facebook/dinov2-base-imagenet1k-1-layer"
        )
        self.model.eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, im, *args, **kwds):
        im = self.transform(im).to(self.device)
        return self.model(im).logits


class PretrainedResNet50:
    """resnet-50 classifier pretrained on imagenet."""

    def __init__(
        self, prompts: List[str], device: str = "cuda", *args, **kwargs
    ) -> None:
        """initialize resnet-50 classifier."""
        assert len(prompts) == 1000, (
            "resnet50 only supports 1000 classes (imagenet). your dataset is probably not imagenet."
        )
        self.device = device
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, im: torch.Tensor) -> torch.Tensor:
        """classify input image."""
        if len(im.shape) == 3:
            im = im.unsqueeze(0)
        im = self.transform(im).to(self.device)
        with torch.inference_mode():
            return self.model(im).cpu()


class PretrainedViTB:
    """vit-b/16 classifier pretrained on imagenet."""

    def __init__(
        self, prompts: List[str], device: str = "cuda", *args, **kwargs
    ) -> None:
        """initialize vit-b/16 classifier."""
        assert len(prompts) == 1000, (
            "vit-b only supports 1000 classes (imagenet). your dataset is probably not imagenet."
        )
        self.device = device
        self.model = torchvision.models.vit_b_16(pretrained=True)
        self.model.eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, im: torch.Tensor) -> torch.Tensor:
        """classify input image."""
        if len(im.shape) == 3:
            im = im.unsqueeze(0)
        im = self.transform(im).to(self.device)
        with torch.inference_mode():
            return self.model(im).cpu()


class EquivariantResNet50:
    """equivariant version of resnet-50."""

    def __init__(
        self, prompts: List[str], device: str, ckpt: str, dataset: str, *args, **kwargs
    ) -> None:
        """initialize resnet-50 classifier."""
        self.device = device

        self.model = setup_prediction_network(
            architecture="resnet50",
            dataset_name=dataset,
            use_pretrained=False,
            freeze_encoder=True,
            input_shape=(3, 32, 32),
            num_classes=len(prompts),
        ).to(device)

        ckpt_dict = torch.load(ckpt, map_location=device)
        if "state_dict" in ckpt_dict:
            prlc_dict = ckpt_dict["state_dict"]
            prediction_network_params = {
                ".".join(k.split(".")[1:]): v
                for k, v in prlc_dict.items()
                if "prediction_network" in k
            }
        elif "prediction_network_state_dict" in ckpt_dict:
            prediction_network_params = ckpt_dict["prediction_network_state_dict"]
            mapped_params = {}
            for k, v in prediction_network_params.items():
                if k.startswith("encoder.") or k.startswith("predictor."):
                    mapped_params[k] = v
                elif k.startswith("fc."):
                    mapped_params[k.replace("fc.", "predictor.", 1)] = v
                elif k.startswith("head."):
                    mapped_params[k.replace("head.", "predictor.", 1)] = v
                else:
                    mapped_params[f"encoder.{k}"] = v
            prediction_network_params = mapped_params
        else:
            raise KeyError(
                "checkpoint does not contain 'state_dict' or 'prediction_network_state_dict'"
            )
        self.model.load_state_dict(prediction_network_params)

        self.model.eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )

    def __call__(self, im: torch.Tensor) -> torch.Tensor:
        """classify input image."""
        if len(im.shape) == 3:
            im = im.unsqueeze(0)
        im = self.transform(im).to(self.device)
        with torch.inference_mode():
            return self.model(im).cpu()


class EquivariantViTB:
    """equivariant version of vit-b."""

    def __init__(
        self, prompts: List[str], device: str, ckpt: str, dataset: str, *args, **kwargs
    ) -> None:
        """initialize vit-b classifier."""
        self.device = device

        self.model = setup_prediction_network(
            architecture="vit",
            dataset_name=dataset,
            use_pretrained=False,
            freeze_encoder=True,
            input_shape=(3, 32, 32),
            num_classes=len(prompts),
        ).to(device)

        ckpt_dict = torch.load(ckpt, map_location=device)
        if "state_dict" in ckpt_dict:
            prlc_dict = ckpt_dict["state_dict"]
            prediction_network_params = {
                ".".join(k.split(".")[1:]): v
                for k, v in prlc_dict.items()
                if "prediction_network" in k
            }
        elif "prediction_network_state_dict" in ckpt_dict:
            prediction_network_params = ckpt_dict["prediction_network_state_dict"]
            mapped_params = {}
            for k, v in prediction_network_params.items():
                if k.startswith("heads.head."):
                    mapped_params[k.replace("heads.head.", "predictor.", 1)] = v
                else:
                    mapped_params[f"encoder.{k}"] = v
            prediction_network_params = mapped_params
        else:
            raise KeyError(
                "checkpoint does not contain 'state_dict' or 'prediction_network_state_dict'"
            )
        self.model.load_state_dict(prediction_network_params)

        self.model.eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )

    def __call__(self, im: torch.Tensor) -> torch.Tensor:
        """classify input image."""
        if len(im.shape) == 3:
            im = im.unsqueeze(0)
        im = self.transform(im).to(self.device)
        with torch.inference_mode():
            return self.model(im).cpu()
