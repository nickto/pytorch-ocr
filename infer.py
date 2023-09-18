import argparse
from glob import glob

from hydra import compose, initialize

from pytorch_ocr.infer import infer
from pytorch_ocr.models.saver_loader import load_model


def main():
    parser = argparse.ArgumentParser(description="Infer classes from a file.")
    parser.add_argument("PATH", type=str, help="path(s) to the image file")
    parser.add_argument("overrides", nargs="*", default=[])
    args = parser.parse_args()

    with initialize(config_path="configs", version_base=None):
        cfg = compose(config_name="config", overrides=args.overrides)

    model, classes = load_model("logs/csdd", cfg.processing.device)

    for file in glob(args.PATH):
        answer = infer(file, model, classes, cfg)
        print("".join(answer))


if __name__ == "__main__":
    main()
