import argparse
import json
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import sys
from pathlib import Path

import albumentations as Aug
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.depthpolyp import build_depthpolyp
from setup.random_lightspot import AddLightSpots


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


DATASETS = {
    "Kvasir": {
        "image_dir": "/home/wangqj/workspace/endo/endoscope/dataset/Kvasir-SEG/images",
        "mask_dir": "/home/wangqj/workspace/endo/endoscope/dataset/Kvasir-SEG/masks",
    },
    "Clinic": {
        "image_dir": "/home/wuzy/DATASET/ClinicDB/PNG/Original",
        "mask_dir": "/home/wuzy/DATASET/ClinicDB/PNG/GroundTruth",
    },
    "Colon": {
        "image_dir": "/home/wuzy/DATASET/CVC-ColonDB/images",
        "mask_dir": "/home/wuzy/DATASET/CVC-ColonDB/masks",
    },
    "ETIS": {
        "image_dir": "/home/wuzy/DATASET/ETIS_LaribPolypDB_kaggle/extracted/images",
        "mask_dir": "/home/wuzy/DATASET/ETIS_LaribPolypDB_kaggle/extracted/masks",
    },
}


def list_images(root):
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def paired_items(image_dir, mask_dir):
    images = list_images(image_dir)
    masks = {p.stem: p for p in list_images(mask_dir)}
    pairs = [(img, masks[img.stem]) for img in images if img.stem in masks]
    missing = [str(img) for img in images if img.stem not in masks]
    return pairs, missing


def polypgen_items(root, sequences):
    items = []
    missing = []
    root = Path(root)
    for seq in sequences:
        image_dir = root / seq / "images"
        mask_dir = root / seq / "masks"
        pairs, seq_missing = paired_items(image_dir, mask_dir)
        items.extend((seq, image, mask) for image, mask in pairs)
        missing.extend(seq_missing)
    return items, missing


def make_blur_transform(dataset_name):
    motion_blur_limits = {
        "Kvasir": (3, 29),
        "Colon": (15, 29),
        "Clinic": (29, 29),
    }
    if dataset_name not in motion_blur_limits:
        return None

    return Aug.Compose([
        Aug.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.2, 0.2), p=1.0),
        Aug.GaussianBlur(blur_limit=(3, 7), p=0.2),
        Aug.MotionBlur(blur_limit=motion_blur_limits[dataset_name], p=1.0),
        Aug.ImageCompression(quality_lower=30, quality_upper=70, p=0.5),
        Aug.RandomFog(fog_coef_lower=0.5, fog_coef_upper=0.8, p=0.3),
        Aug.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
        AddLightSpots(radius_range=(5, 40), intensity=0.85, num_spots=1, always_apply=False, p=0.8),
    ], p=1.0)


def load_image(path, image_size, transform=None):
    image = Image.open(path).convert("RGB")
    original_size = image.size
    image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image)
    if transform is not None:
        array = transform(image=array)["image"]
    array = array.astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor, original_size


def load_mask(path, image_size):
    mask = Image.open(path).convert("L")
    mask = mask.resize((image_size, image_size), Image.NEAREST)
    array = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
    return array


def metrics(pred, target):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    intersection = float((pred * target).sum())
    pred_sum = float(pred.sum())
    target_sum = float(target.sum())
    union = pred_sum + target_sum - intersection

    if target_sum == 0 and pred_sum == 0:
        dice = 1.0
        iou = 1.0
        recall = 1.0
    else:
        dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-7)
        iou = intersection / (union + 1e-7)
        recall = intersection / (target_sum + 1e-7)
    return dice, iou, recall


def save_prediction(prob, output_dir, rel_name):
    output_dir = Path(output_dir)
    prob_dir = output_dir / "prob"
    bin_dir = output_dir / "binary"
    prob_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    rel_path = Path(rel_name).with_suffix(".png")
    (prob_dir / rel_path.parent).mkdir(parents=True, exist_ok=True)
    (bin_dir / rel_path.parent).mkdir(parents=True, exist_ok=True)

    prob_u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    bin_u8 = (prob >= save_prediction.threshold).astype(np.uint8) * 255
    Image.fromarray(prob_u8).save(prob_dir / rel_path)
    Image.fromarray(bin_u8).save(bin_dir / rel_path)


def evaluate_dataset(model, items, output_dir, device, image_size, threshold, transform=None):
    save_prediction.threshold = threshold
    dice_scores = []
    iou_scores = []
    recall_scores = []

    for rel_name, image_path, mask_path in tqdm(items, desc=Path(output_dir).name):
        image, _ = load_image(image_path, image_size, transform=transform)
        target = load_mask(mask_path, image_size)
        image = image.to(device)

        with torch.no_grad():
            pred_seg, _ = model(image)
            pred_seg = F.interpolate(
                pred_seg,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )
            prob = pred_seg.squeeze().detach().cpu().numpy()

        pred = (prob >= threshold).astype(np.float32)
        dice, iou, recall = metrics(pred, target)
        dice_scores.append(dice)
        iou_scores.append(iou)
        recall_scores.append(recall)
        save_prediction(prob, output_dir, rel_name)

    return {
        "count": len(items),
        "dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
        "recall": float(np.mean(recall_scores)) if recall_scores else 0.0,
    }


def apply_split(items, split, test_size, seed):
    if split == "all":
        return items
    train_items, val_items = train_test_split(items, test_size=test_size, random_state=seed)
    if split == "train":
        return train_items
    if split == "val":
        return val_items
    raise ValueError(f"Unknown split: {split}")


def build_items_for_dataset(name, cfg, polypgen_root, polypgen_sequences, split, test_size, seed):
    if name == "PolypGen":
        raw_items, missing = polypgen_items(polypgen_root, polypgen_sequences)
        items = [(str(Path(seq) / image.name), image, mask) for seq, image, mask in raw_items]
        return apply_split(items, split, test_size, seed), missing

    pairs, missing = paired_items(cfg["image_dir"], cfg["mask_dir"])
    items = [(image.name, image, mask) for image, mask in pairs]
    return apply_split(items, split, test_size, seed), missing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", default="checkpoints/DepthPolyp_Kvasir.pth")
    parser.add_argument("--output-dir", default="results/depthpolyp_open_weight")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--datasets", nargs="+", default=["Kvasir", "Clinic", "Colon", "PolypGen", "ETIS"])
    parser.add_argument("--variants", nargs="+", choices=["clean", "blur"], default=["clean"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--polypgen-root", default="/home/wuzy/DATASET/polypgen")
    parser.add_argument("--polypgen-sequences", nargs="+", default=["seq18", "seq19", "seq20", "seq21", "seq22"])
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = build_depthpolyp("b0", in_channels=3, num_classes=2, decoder_channels=256, activation=None)
    state_dict = torch.load(args.weight, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    dataset_cfgs = dict(DATASETS)
    dataset_cfgs["PolypGen"] = {}

    summary = {
        "weight": args.weight,
        "device": str(device),
        "image_size": args.image_size,
        "threshold": args.threshold,
        "seed": args.seed,
        "split": args.split,
        "test_size": args.test_size,
        "datasets": {},
    }

    for name in args.datasets:
        if name not in dataset_cfgs:
            raise ValueError(f"Unknown dataset: {name}")
        items, missing = build_items_for_dataset(
            name,
            dataset_cfgs[name],
            args.polypgen_root,
            args.polypgen_sequences,
            args.split,
            args.test_size,
            args.seed,
        )
        if not items:
            raise RuntimeError(f"No image/mask pairs found for {name}")

        summary["datasets"][name] = {}
        for variant in args.variants:
            transform = make_blur_transform(name) if variant == "blur" else None
            output_dir = Path(args.output_dir) / name / variant
            result = evaluate_dataset(
                model,
                items,
                output_dir,
                device,
                args.image_size,
                args.threshold,
                transform=transform,
            )
            result["missing_masks"] = len(missing)
            summary["datasets"][name][variant] = result
            print(
                f"{name:8s} {variant:5s} count={result['count']:4d} "
                f"dice={result['dice']:.4f} iou={result['iou']:.4f} recall={result['recall']:.4f} "
                f"missing_masks={len(missing)}"
            )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.output_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
