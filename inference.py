import albumentations
import hydra
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from pytorch_ocr.models.crnn import CRNN
from pytorch_ocr.utils.model_decoders import decode_padded_predictions, decode_predictions

# I use "∅" to denote the blank token. This list is automatically generated at training,
# but I recommend that you hardcode your characters at evaluation
classes = [
    "∅",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "j",
    "k",
    "m",
    "n",
    "o",
    "p",
    "q",  # THIS SHOULD NOT EXIST LOL
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]


def load_image(path, cfg):
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


def inference(image_path, model, cfg):
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


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg):
    # Setup model and load weights
    device = torch.device(cfg.processing.device)
    model = CRNN(
        resolution=(cfg.processing.image_width, cfg.processing.image_height),
        dims=cfg.model.dims,
        num_chars=len(classes) - 1,  # because of "∅"
        use_attention=cfg.model.use_attention,
        use_ctc=cfg.model.use_ctc,
        grayscale=cfg.model.gray_scale,
    ).to(device)
    model.load_state_dict(
        torch.load(cfg.paths.save_model_as),
        map_location=torch.device(device),
    )
    model.eval()
    filepath = "dataset/eBwsgwf.png"
    answer = inference(filepath, model, cfg)
    print(f"text: {answer}")


if __name__ == "__main__":
    main()
