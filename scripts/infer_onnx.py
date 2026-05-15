import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run DepthPolyp ONNX inference on images.")
    parser.add_argument("--onnx", default="checkpoints/DepthPolyp_Kvasir.onnx")
    parser.add_argument("--input", default="samples/kvasir/images")
    parser.add_argument("--output", default="samples/kvasir/outputs")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.3)
    return parser.parse_args()


def list_images(input_path: Path):
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def preprocess(image_path: Path, image_size: int):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(resized).astype(np.float32) / 255.0
    tensor = np.transpose(array, (2, 0, 1))[None, ...]
    return image, original_size, tensor


def to_grayscale(probability: np.ndarray, size):
    probability = np.clip(probability, 0.0, 1.0)
    image = Image.fromarray((probability * 255).astype(np.uint8), mode="L")
    return image.resize(size, Image.BILINEAR)


def colorize_purple_yellow(probability: np.ndarray, size):
    probability = np.clip(probability, 0.0, 1.0)
    stops = np.array(
        [
            [38, 5, 84],
            [86, 33, 132],
            [141, 48, 140],
            [203, 71, 119],
            [245, 135, 48],
            [252, 231, 37],
        ],
        dtype=np.float32,
    )
    scaled = probability * (len(stops) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.clip(lower + 1, 0, len(stops) - 1)
    alpha = (scaled - lower)[..., None]
    colored = stops[lower] * (1.0 - alpha) + stops[upper] * alpha
    image = Image.fromarray(colored.astype(np.uint8), mode="RGB")
    return image.resize(size, Image.BILINEAR)


def make_overlay(image: Image.Image, mask: Image.Image):
    base = image.convert("RGBA")
    mask_array = np.asarray(mask).astype(np.float32) / 255.0
    color = np.zeros((mask_array.shape[0], mask_array.shape[1], 4), dtype=np.uint8)
    color[..., 0] = 252
    color[..., 1] = 231
    color[..., 2] = 37
    color[..., 3] = (mask_array * 155).astype(np.uint8)
    return Image.alpha_composite(base, Image.fromarray(color, mode="RGBA")).convert("RGB")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output)
    mask_dir = output_root / "masks"
    depth_dir = output_root / "depth"
    overlay_dir = output_root / "overlay"
    for directory in (mask_dir, depth_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    images = list_images(input_path)
    if not images:
        raise FileNotFoundError(f"No images found under {input_path}")

    for image_path in images:
        image, original_size, tensor = preprocess(image_path, args.image_size)
        segmentation, depth = session.run(None, {input_name: tensor})
        seg_prob = segmentation[0, 0]
        depth_prob = depth[0, 0]

        seg_image = to_grayscale(seg_prob, original_size)
        depth_image = colorize_purple_yellow(depth_prob, original_size)
        binary_mask = seg_image.point(lambda value: 255 if value >= int(args.threshold * 255) else 0)
        overlay = make_overlay(image, seg_image)

        stem = image_path.stem
        binary_mask.save(mask_dir / f"{stem}.png")
        depth_image.save(depth_dir / f"{stem}.png")
        overlay.save(overlay_dir / f"{stem}.jpg", quality=95)

    print(f"Processed {len(images)} image(s). Outputs saved to {output_root}")


if __name__ == "__main__":
    main()
