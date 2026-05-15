import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.depthpolyp import build_depthpolyp


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Export DepthPolyp to ONNX.")
    parser.add_argument("--checkpoint", default="checkpoints/DepthPolyp_Kvasir.pth")
    parser.add_argument("--output", default="checkpoints/DepthPolyp_Kvasir.onnx")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()
    model = build_depthpolyp(
        encoder_name="b0",
        in_channels=3,
        num_classes=2,
        decoder_channels=256,
        activation=None,
    )
    state_dict = load_checkpoint(args.checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, args.image_size, args.image_size)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["image"],
        output_names=["segmentation", "depth"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes={
            "image": {0: "batch"},
            "segmentation": {0: "batch"},
            "depth": {0: "batch"},
        },
    )
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    main()
