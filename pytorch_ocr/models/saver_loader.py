import os
import shutil
from typing import List, Tuple

import torch
import yaml
from .crnn import CRNN


def save_model(model: CRNN, classes: List[str], path: str, override: bool = False):
    if os.path.exists(path) and not override:
        raise FileExistsError(f"save_dir {path} already exists. Set override=True to override.")
    if os.path.exists(path) and override:
        print(f"overriding save_dir {path}")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    state_path = os.path.join(path, f"state.pth")
    torch.save(model.state_dict(), state_path)
    print(f"saved model state to {state_path}")

    params_path = os.path.join(path, f"params.yaml")
    params = {}
    params["crnn"] = {
        "resolution": model.resolution,
        "dims": model.dims,
        "num_chars": model.num_chars,
        "use_attention": model.use_attention,
        "use_ctc": model.use_ctc,
        "grayscale": model.grayscale,
    }
    params["classes"] = classes
    with open(params_path, "w") as f:
        yaml.safe_dump(params, f)
    print(f"saved model params to {params_path}")


def load_model(path, device: str = "cpu") -> Tuple[CRNN, List[str]]:
    state_path = os.path.join(path, f"state.pth")
    params_path = os.path.join(path, f"params.yaml")

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    model = CRNN(
        resolution=params["crnn"]["resolution"],
        dims=params["crnn"]["dims"],
        num_chars=params["crnn"]["num_chars"],
        use_attention=params["crnn"]["use_attention"],
        use_ctc=params["crnn"]["use_ctc"],
        grayscale=params["crnn"]["grayscale"],
    ).to(device)
    model.load_state_dict(torch.load(state_path))
    model.eval()
    return model, params["classes"]
