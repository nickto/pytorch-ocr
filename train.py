import hydra
import torch
from rich.console import Console

from pytorch_ocr import train

# Setup rich console
console = Console()


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def run_training(cfg):
    train.run_training(cfg)


if __name__ == "__main__":
    try:
        torch.cuda.empty_cache()
        run_training()
    except Exception:
        console.print_exception()
