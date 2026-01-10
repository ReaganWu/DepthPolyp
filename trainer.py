import os
import datetime
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from PIL import Image
from transformers import pipeline

from setup.data_loader import get_dataloader
from setup.metrics import Dice_Metric, Jaccard_Index, Recall


# =========================
# Global Configuration
# =========================
PT_SAVED_DIR = "."          # Save checkpoints at project root for better portability
DEVICE = "cuda:0"


# =========================
# Depth Estimation Utility
# =========================
def get_depth_maps(images_tensor, depth_pipe):
    """
    Args:
        images_tensor: PyTorch tensor with shape (B, C, H, W), normalized to [0, 1]
        depth_pipe: HuggingFace depth estimation pipeline

    Returns:
        depth_maps: Normalized depth maps with shape (B, 1, H, W)
    """
    images_np = images_tensor.cpu().numpy()
    pil_images = []

    for img in images_np:
        # (C, H, W) -> (H, W, C)
        img = np.transpose(img, (1, 2, 0))

        # Normalize float image to uint8
        if img.dtype != np.uint8:
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            img = (img * 255).astype(np.uint8)

        # Ensure 3-channel RGB
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        pil_images.append(Image.fromarray(img))

    # Depth inference
    depth_maps = [depth_pipe(img)["depth"] for img in pil_images]
    depth_maps = np.array(depth_maps)

    # Min-max normalization per image
    depth_maps = np.array([
        (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        for depth in depth_maps
    ])

    depth_maps = torch.tensor(depth_maps, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    return depth_maps


# =========================
# Training Loop
# =========================
def train_model_endo_depth(
    model_name,
    model,
    train_loader,
    val_loader,
    criterion_uncertain,
    optimizer,
    lr_scheduler,
    num_epochs,
):
    print(f"[INFO] Initializing model: {model_name}")\

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_saved_dir = os.path.join(PT_SAVED_DIR, model_name)
    os.makedirs(model_saved_dir, exist_ok=True)

    model_save_path = os.path.join(model_saved_dir, f"{model_name}_{timestamp}.pth")

    print_step = max(1, len(train_loader) // 2)
    best_score = 0.60
    best_model_params = None

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total parameters: {num_params}")

    # Initialize depth estimation pipeline
    depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=DEVICE
    )

    for epoch in range(num_epochs):
        model.train()
        losses, dice_list = [], []

        print(f"\n========== Epoch [{epoch}/{num_epochs}] ==========")

        for i_step, (blur_data, clean_data, target_seg) in enumerate(tqdm(train_loader)):
            data = blur_data
            data = data.to(DEVICE).permute(0, 3, 1, 2) / 255.0
            target = target_seg.to(DEVICE).permute(0, 3, 1, 2) / 255.0

            optimizer.zero_grad()

            pred_rgb, pred_depth = model(data)
            depth_gt = get_depth_maps(data, depth_pipe)

            loss = criterion_uncertain(pred_rgb, target, pred_depth, depth_gt)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            pred_bin = pred_rgb.detach().cpu().numpy()
            pred_bin[pred_bin < 0.5] = 0.0
            pred_bin[pred_bin >= 0.5] = 1.0

            dice_list.append(Dice_Metric(pred_bin, target.cpu().numpy()))

            if i_step % print_step == 0:
                print(
                    f"[Iter {i_step}/{len(train_loader)}] "
                    f"Loss: {np.mean(losses):.4f}, "
                    f"Dice: {np.mean(dice_list):.4f}"
                )

        print(
            f"[Train Summary] "
            f"Loss: {np.mean(losses):.4f}, "
            f"Dice: {np.mean(dice_list):.4f}"
        )

        # ================= Validation =================
        val_results = Dice_Val_Metric_Joint_Depth(
            model, val_loader, depth_pipe, threshold=0.3
        )

        dice_clean = val_results["dice_clear"]
        dice_blur = val_results["dice_blur"]
        depth_loss = val_results["depth_loss"]

        print(
            f"[Validation]\n"
            f" Clean | Dice: {dice_clean:.4f}, IoU: {val_results['jac_clear']:.4f}, Recall: {val_results['recall_clear']:.4f}\n"
            f" Blur  | Dice: {dice_blur:.4f}, IoU: {val_results['jac_blur']:.4f}, Recall: {val_results['recall_blur']:.4f}\n"
            f" Depth | L1 Loss: {depth_loss:.4f}"
        )

        score = 0.7 * dice_clean + 0.3 * dice_blur

        if score > best_score:
            print(f"[BEST] Score improved {best_score:.4f} → {score:.4f}. Saving model.")
            best_score = score
            best_model_params = model.state_dict()
            torch.save(best_model_params, model_save_path)

        lr_scheduler.step()
        print(f"[LR] Current LR: {lr_scheduler.get_last_lr()[0]}")

    if best_model_params is not None:
        model.load_state_dict(best_model_params)
        best_state_performance_depthpolyp(model)


# =========================
# Validation Metrics
# =========================
def Dice_Val_Metric_Joint_Depth(
    model,
    loader,
    depth_pipe,
    criterion_depth=nn.L1Loss(),
    threshold=0.3,
    mode="Training"
):
    val_dice = val_jac = val_recall = val_depth = 0
    val_blur_dice = val_blur_jac = val_blur_recall = 0

    model.eval()

    with torch.no_grad():
        for blur_img, clear_img, target_seg in loader:
            blur = blur_img.to(DEVICE).permute(0, 3, 1, 2) / 255.0
            clear = clear_img.to(DEVICE).permute(0, 3, 1, 2) / 255.0
            target = target_seg.to(DEVICE).permute(0, 3, 1, 2) / 255.0

            depth_maps = (
                get_depth_maps(clear, depth_pipe)
                if mode == "Training"
                else torch.zeros_like(target)
            )

            blur_seg, blur_depth = model(blur)
            clear_seg, _ = model(clear)

            blur_bin = (blur_seg.cpu().numpy() >= threshold).astype(np.float32)
            clear_bin = (clear_seg.cpu().numpy() >= threshold).astype(np.float32)
            target_np = target.cpu().numpy()

            val_blur_dice += Dice_Metric(blur_bin, target_np)
            val_blur_jac += Jaccard_Index(blur_bin, target_np)
            val_blur_recall += Recall(blur_bin, target_np)
            val_depth += criterion_depth(blur_depth, depth_maps).item()

            val_dice += Dice_Metric(clear_bin, target_np)
            val_jac += Jaccard_Index(clear_bin, target_np)
            val_recall += Recall(clear_bin, target_np)

    n = len(loader)
    return {
        "dice_clear": val_dice / n,
        "jac_clear": val_jac / n,
        "recall_clear": val_recall / n,
        "dice_blur": val_blur_dice / n,
        "jac_blur": val_blur_jac / n,
        "recall_blur": val_blur_recall / n,
        "depth_loss": val_depth / n,
    }


# =========================
# Final Benchmark Evaluation
# =========================
def best_state_performance_depthpolyp(model):
    datasets = {
        "KVASIR-v2": dict(batch_size=16),
        "ClinicDB-v2": dict(batch_size=16),
        "ColonDB-v2": dict(batch_size=16),
    }

    print("\n[Final Evaluation on Public Benchmarks]")
    for name, cfg in datasets.items():
        _, val_loader, _ = get_dataloader(name, shuffle=True, **cfg)
        res = Dice_Val_Metric_Joint_Depth(model, val_loader, None, mode="val")

        print(
            f"{name:12s} | "
            f"Clean Dice: {res['dice_clear']:.4f}, "
            f"Blur Dice: {res['dice_blur']:.4f}"
        )
