from typing import List

import albumentations
import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import DictConfig
from PIL import Image

from pytorch_ocr.models.crnn import CRNN
from pytorch_ocr.utils.model_decoders import decode_padded_predictions, decode_predictions


def load_image(path: str, cfg: DictConfig):
    image = Image.open(path).convert("RGB")

    image = image.resize((cfg.processing.image_width, cfg.processing.image_height), resample=Image.BILINEAR)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    if cfg.model.gray_scale:
        image = transform(image)
    else:
        image = np.array(image)
        augmented = aug(image=image)
        image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float)

    return image


def infer(image_path, model: CRNN, classes: List[str], cfg: DictConfig):
    device = torch.device(cfg.processing.device)

    image = load_image(image_path, cfg)
    image = image[None, ...]

    if str(device) == "cuda":
        image = image.cuda()
    image = image.float()
    with torch.no_grad():
        preds, _ = model(images=image)

    if model.use_ctc:
        answer = decode_predictions(preds, classes)
    else:
        answer = decode_padded_predictions(preds, classes)
    return answer
