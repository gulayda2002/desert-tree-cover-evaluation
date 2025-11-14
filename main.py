import os
from glob import glob
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

##########################################
# CONFIG
##########################################

DATASET_PATH = "dataset"
IMAGE_SIZE = 256
NUM_CLASSES = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 60  # More epochs with augmentation
BATCH_SIZE = 8  # Conservative batch size


##########################################
# MASK LOADING UTILITY
##########################################

def load_mask(mask_path):
    """Loads mask from annotation_mask or annotation_rgb and binarizes it."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Mask not found: {mask_path}")

    # If mask is small values (0/1), convert to 0/255 then binarize
    if np.max(mask) <= 5:
        mask = (mask > 0).astype(np.uint8) * 255

    # Standardize to binary 0/1
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
    mask = (mask > 127).astype(np.uint8)
    return mask


##########################################
# DATASET LOADER
##########################################

class DesertDataset(Dataset):
    def __init__(self, img_paths, mask_paths, augment=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        mask = load_mask(self.mask_paths[idx])

        # More augmentation for better generalization
        if self.augment:
            # Horizontal flip
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            # Vertical flip
            if random.random() > 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)
            # 90-degree rotations
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])  # 90, 180, 270 degrees
                img = np.rot90(img, k).copy()  # Copy to avoid negative strides
                mask = np.rot90(mask, k).copy()
        
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW

        # CrossEntropyLoss expects class indices (LongTensor) with shape (N, H, W)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)


##########################################
# LOAD DATASET FROM FOLDERS
##########################################

def collect_pairs(split):
    """Collect paired (image, mask) paths for a dataset split.

    - Images are under "images" with common extensions
    - Masks must be under "masks" (binary 0/1). "rgb_masks" are not used for training/eval.
    - Pairs with missing masks are skipped with a warning.
    """
    folder = f"{DATASET_PATH}/{split}"

    img_exts = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
    img_files = []
    for ext in img_exts:
        img_files.extend(glob(f"{folder}/images/*.{ext}"))
    img_files = sorted(img_files)

    if len(img_files) == 0:
        raise FileNotFoundError(
            f"No images found in {folder}/images (looked for extensions: {img_exts})."
        )

    paired_imgs, paired_masks = [], []

    for img_path in img_files:
        base = os.path.basename(img_path)
        name, _ = os.path.splitext(base)

        # Require a binary mask in masks/
        mask_path = f"{folder}/masks/{name}.png"
        if not os.path.exists(mask_path):
            # Try other common mask extensions within masks/
            candidates = []
            for ext in ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]:
                candidates.extend(glob(f"{folder}/masks/{name}.{ext}"))
            if len(candidates) > 0:
                mask_path = candidates[0]
            else:
                print(f"[WARN] No binary mask found for image: {img_path}. Skipping.")
                continue

        paired_imgs.append(img_path)
        paired_masks.append(mask_path)

    if len(paired_imgs) == 0:
        raise RuntimeError(
            f"Found {len(img_files)} images in {folder}/images but no paired masks in {folder}/masks."
        )

    return paired_imgs, paired_masks


##########################################
# RANDOM FOREST SEGMENTATION
##########################################

def train_random_forest(img_paths, mask_paths, max_pixels_per_image=10000, balance=True, n_estimators=80, max_depth=12, min_samples_split=2):
    """Train a RandomForest on sampled pixels.

    To avoid OOM, we sample up to max_pixels_per_image per image. If balance=True we
    sample equal foreground/background (where possible).
    """
    X_parts, y_parts = [], []

    for img_path, mask_path in zip(img_paths, mask_paths):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        mask = load_mask(mask_path)

        pixels = img.reshape(-1, 3)
        labels = mask.reshape(-1)

        total_pixels = pixels.shape[0]
        if max_pixels_per_image <= 0 or max_pixels_per_image >= total_pixels:
            # Use all pixels (still risky for large datasets)
            sample_idx = np.arange(total_pixels)
        else:
            if balance:
                pos_idx = np.where(labels == 1)[0]
                neg_idx = np.where(labels == 0)[0]
                per_class = max_pixels_per_image // 2
                # Sample with replacement if not enough
                if len(pos_idx) >= per_class:
                    pos_sample = np.random.choice(pos_idx, per_class, replace=False)
                else:
                    pos_sample = np.random.choice(pos_idx, per_class, replace=True) if len(pos_idx) > 0 else np.array([], dtype=int)
                if len(neg_idx) >= per_class:
                    neg_sample = np.random.choice(neg_idx, per_class, replace=False)
                else:
                    neg_sample = np.random.choice(neg_idx, per_class, replace=True) if len(neg_idx) > 0 else np.array([], dtype=int)
                sample_idx = np.concatenate([pos_sample, neg_sample])
            else:
                sample_idx = np.random.choice(total_pixels, max_pixels_per_image, replace=False)

        sampled_pixels = pixels[sample_idx]
        sampled_labels = labels[sample_idx]

        X_parts.append(sampled_pixels)
        y_parts.append(sampled_labels)

    if len(X_parts) == 0:
        raise RuntimeError(
            "Random Forest training received no samples. "
            f"Images: {len(img_paths)}, Masks: {len(mask_paths)}. Check dataset pairing."
        )

    X = np.vstack(X_parts).astype(np.uint8)  # keep compact then cast
    y = np.hstack(y_parts).astype(np.uint8)
    X = X.astype(np.float32)  # RF will cast anyway; ensures reasonable dtype

    approx_mem_mb = (X.nbytes + y.nbytes) / (1024**2)
    print(f"[RF] Training samples: {X.shape[0]} features per sample: {X.shape[1]} (~{approx_mem_mb:.1f} MB in arrays)")

    rf = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split,
        min_samples_leaf=1,  # Allow very fine splits
        n_jobs=-1
    )
    rf.fit(X, y)
    return rf


def rf_predict(rf, img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    pixels = img.reshape(-1, 3)
    pred = rf.predict(pixels)
    return pred.reshape(IMAGE_SIZE, IMAGE_SIZE)


##########################################
# U-NET MODEL
##########################################

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

        self.enc1 = CBR(3, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.center = CBR(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)

        self.final = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        c = self.center(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(c), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)


##########################################
# LOSS FUNCTIONS
##########################################

class DiceLoss(nn.Module):
    """Dice loss for segmentation - works better with class imbalance."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # pred: (N, C, H, W) logits
        # target: (N, H, W) class indices
        pred_soft = torch.softmax(pred, dim=1)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        intersection = (pred_soft * target_one_hot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Combines CrossEntropy and Dice loss."""
    def __init__(self, weight=None, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice(pred, target)
        return (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss


##########################################
# METRICS
##########################################

def metrics(gt, pred):
    gt = gt.flatten()
    pred = pred.flatten()

    tn, fp, fn, tp = confusion_matrix(gt, pred, labels=[0, 1]).ravel()

    OA = (tp + tn) / (tp + tn + fp + fn)
    IoU = tp / (tp + fp + fn + 1e-6)
    P0 = OA
    Pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (tp + tn + fp + fn) ** 2
    Kappa = (P0 - Pe) / (1 - Pe + 1e-6)

    return OA, IoU, Kappa


def ensure_outdir(outdir: str | None) -> Path:
    if outdir is None or outdir.strip() == "":
        outdir_path = Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    return outdir_path


def plot_metrics_bar(rf_mean, unet_mean=None, out_path: Path | None = None):
    labels = ["OA", "mIoU", "Kappa"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, rf_mean, width, label="Random Forest")
    if unet_mean is not None:
        ax.bar(x + width/2, unet_mean, width, label="U-Net")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_confmat(cm: np.ndarray, title: str, out_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=["BG", "FG"], yticklabels=["BG", "FG"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_qualitative(image_bgr: np.ndarray, gt_mask: np.ndarray, rf_pred: np.ndarray | None,
                     unet_pred: np.ndarray | None, out_path: Path):
    image_rgb = cv2.cvtColor(cv2.resize(image_bgr, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2RGB)
    fig_cols = 3 if unet_pred is None else 4
    fig, axs = plt.subplots(1, fig_cols, figsize=(4*fig_cols, 4))

    axs[0].imshow(image_rgb)
    axs[0].set_title("Image")
    axs[0].axis('off')

    axs[1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
    axs[1].set_title("GT Mask")
    axs[1].axis('off')

    axs[2].imshow(rf_pred, cmap='gray', vmin=0, vmax=1)
    axs[2].set_title("RF Pred")
    axs[2].axis('off')

    if unet_pred is not None:
        axs[3].imshow(unet_pred, cmap='gray', vmin=0, vmax=1)
        axs[3].set_title("U-Net Pred")
        axs[3].axis('off')

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


##########################################
# MAIN PIPELINE
##########################################

def main():
    parser = argparse.ArgumentParser(description="Desert tree cover evaluation")
    parser.add_argument("--rf-only", action="store_true", help="Only train/evaluate Random Forest; skip U-Net")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of U-Net training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="U-Net batch size")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples per split (0 = no limit)")
    parser.add_argument("--rf-max-pixels", type=int, default=10000, help="Max sampled pixels per image for RF (0 = all)")
    parser.add_argument("--rf-no-balance", action="store_true", help="Disable foreground/background balancing for RF sampling")
    parser.add_argument("--rf-n-estimators", type=int, default=80, help="Number of trees for RF")
    parser.add_argument("--rf-max-depth", type=int, default=12, help="Max depth for RF trees")
    parser.add_argument("--rf-min-samples", type=int, default=2, help="Min samples to split for RF trees")
    parser.add_argument("--outdir", type=str, default="", help="Directory to save outputs (metrics, plots, samples)")
    parser.add_argument("--viz-samples", type=int, default=6, help="Number of qualitative samples to save")
    args = parser.parse_args()

    # Use local training hyperparameters from CLI
    epochs = args.epochs
    batch_size = args.batch_size

    print("Loading dataset...")
    train_imgs, train_masks = collect_pairs("train")
    val_imgs, val_masks = collect_pairs("val")
    test_imgs, test_masks = collect_pairs("test")

    if args.limit and args.limit > 0:
        train_imgs, train_masks = train_imgs[:args.limit], train_masks[:args.limit]
        val_imgs, val_masks = val_imgs[:args.limit], val_masks[:args.limit]
        test_imgs, test_masks = test_imgs[:args.limit], test_masks[:args.limit]

    print(f"Train pairs: {len(train_imgs)} | Val pairs: {len(val_imgs)} | Test pairs: {len(test_imgs)}")
    outdir = ensure_outdir(args.outdir)
    print(f"Saving outputs to: {outdir}")

    ##########################################
    # TRAIN RF
    ##########################################
    print("\n=== Training Random Forest ===")
    rf = train_random_forest(
        train_imgs,
        train_masks,
        max_pixels_per_image=args.rf_max_pixels,
        balance=(not args.rf_no_balance),
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        min_samples_split=args.rf_min_samples,
    )

    # Evaluate RF on validation set
    print("Evaluating Random Forest on validation set...")
    rf_val_OA, rf_val_IoU, rf_val_Kappa = [], [], []
    rf_val_cm = np.zeros((2, 2), dtype=np.int64)
    for img, mask in zip(val_imgs, val_masks):
        pred = rf_predict(rf, img)
        gt = load_mask(mask)
        OA, IoU, Kappa = metrics(gt, pred)
        rf_val_OA.append(OA); rf_val_IoU.append(IoU); rf_val_Kappa.append(Kappa)
        rf_val_cm += confusion_matrix(gt.flatten(), pred.flatten(), labels=[0, 1])

    # Evaluate RF on test set
    print("Evaluating Random Forest on test set...")
    rf_test_OA, rf_test_IoU, rf_test_Kappa = [], [], []
    rf_test_cm = np.zeros((2, 2), dtype=np.int64)
    for img, mask in zip(test_imgs, test_masks):
        pred = rf_predict(rf, img)
        gt = load_mask(mask)
        OA, IoU, Kappa = metrics(gt, pred)
        rf_test_OA.append(OA); rf_test_IoU.append(IoU); rf_test_Kappa.append(Kappa)
        rf_test_cm += confusion_matrix(gt.flatten(), pred.flatten(), labels=[0, 1])

    ##########################################
    # TRAIN UNET (optional)
    ##########################################
    if not args.rf_only:
        print("\n=== Training U-Net ===")
        model = UNet().to(DEVICE)
        # Adam optimizer - try standard 1e-3 LR which worked better
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # Calculate class weights to handle imbalance
        print("Calculating class weights from training data...")
        total_pixels = 0
        fg_pixels = 0
        for mask_path in train_masks[:min(100, len(train_masks))]:  # Sample for speed
            mask = load_mask(mask_path)
            total_pixels += mask.size
            fg_pixels += np.sum(mask)
        
        bg_pixels = total_pixels - fg_pixels
        if fg_pixels > 0:
            # Weight inversely proportional to frequency
            weight_bg = total_pixels / (2.0 * bg_pixels)
            weight_fg = total_pixels / (2.0 * fg_pixels)
            class_weights = torch.tensor([weight_bg, weight_fg], dtype=torch.float32).to(DEVICE)
            print(f"Class weights - BG: {weight_bg:.3f}, FG: {weight_fg:.3f}")
        else:
            class_weights = None
            print("Warning: No foreground pixels found in sample!")
        
        # Use simple CrossEntropyLoss with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        # ReduceLROnPlateau scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        train_ds = DesertDataset(train_imgs, train_masks, augment=True)  # Enable augmentation for better generalization
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        # Mixed precision training for faster training and better memory usage
        scaler = torch.amp.GradScaler('cuda') if DEVICE == "cuda" else None
        
        best_val_iou = 0.0
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            pbar = tqdm(train_dl, desc=f"Epoch {epoch + 1}/{epochs}")
            for img, mask in pbar:
                img, mask = img.to(DEVICE), mask.to(DEVICE)

                optimizer.zero_grad()
                
                # Mixed precision forward pass
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        pred = model(img)
                        loss = criterion(pred, mask)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(img)
                    loss = criterion(pred, mask)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_loss = epoch_loss / len(train_dl)
            
            # Validate every epoch for better monitoring and model selection
            if True:  # Every epoch
                model.eval()
                val_iou_list = []
                with torch.no_grad():
                    for img_path, mask_path in zip(val_imgs[:100], val_masks[:100]):  # More samples for accuracy
                        image = cv2.imread(img_path)
                        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
                        x = torch.tensor(image / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
                        pred = model(x)
                        pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
                        gt = load_mask(mask_path)
                        _, iou, _ = metrics(gt, pred)
                        val_iou_list.append(iou)
                
                val_iou = np.mean(val_iou_list)
                print(f"\nEpoch {epoch+1}: Train Loss={avg_loss:.4f}, Val mIoU={val_iou:.4f}, LR={optimizer.param_groups[0]['lr']:.2e}")
                
                # Step ReduceLROnPlateau scheduler based on validation mIoU
                scheduler.step(val_iou)
                
                if val_iou > best_val_iou:
                    best_val_iou = val_iou
                    # Save best model
                    torch.save(model.state_dict(), outdir / "best_unet.pth")
                    print(f"Saved best model (mIoU: {best_val_iou:.4f})")
        
        # Load best model for final evaluation
        if (outdir / "best_unet.pth").exists():
            model.load_state_dict(torch.load(outdir / "best_unet.pth"))
            print(f"\nLoaded best model for evaluation (Val mIoU: {best_val_iou:.4f})")

        ##########################################
        # EVALUATE UNET ON VALIDATION
        ##########################################
        print("Evaluating U-Net on validation set...")
        model.eval()
        unet_val_OA, unet_val_IoU, unet_val_Kappa = [], [], []
        unet_val_cm = np.zeros((2, 2), dtype=np.int64)

        for img, mask in zip(val_imgs, val_masks):
            image = cv2.imread(img)
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            x = torch.tensor(image / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

            with torch.no_grad():
                pred = model(x)
                pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

            gt = load_mask(mask)
            OA, IoU, Kappa = metrics(gt, pred)
            unet_val_OA.append(OA); unet_val_IoU.append(IoU); unet_val_Kappa.append(Kappa)
            unet_val_cm += confusion_matrix(gt.flatten(), pred.flatten(), labels=[0, 1])

        ##########################################
        # EVALUATE UNET ON TEST
        ##########################################
        print("Evaluating U-Net on test set...")
        unet_test_OA, unet_test_IoU, unet_test_Kappa = [], [], []
        unet_test_cm = np.zeros((2, 2), dtype=np.int64)

        for img, mask in zip(test_imgs, test_masks):
            image = cv2.imread(img)
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            x = torch.tensor(image / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

            with torch.no_grad():
                pred = model(x)
                pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

            gt = load_mask(mask)
            OA, IoU, Kappa = metrics(gt, pred)
            unet_test_OA.append(OA); unet_test_IoU.append(IoU); unet_test_Kappa.append(Kappa)
            unet_test_cm += confusion_matrix(gt.flatten(), pred.flatten(), labels=[0, 1])

    ##########################################
    # PRINT RESULTS
    ##########################################
    print("\n" + "="*60)
    print("RANDOM FOREST RESULTS")
    print("="*60)
    
    # Validation results
    rf_val_mean = [float(np.mean(rf_val_OA)), float(np.mean(rf_val_IoU)), float(np.mean(rf_val_Kappa))]
    print("\n--- Validation Set ---")
    print(f"OA    : {rf_val_mean[0]:.4f}")
    print(f"mIoU  : {rf_val_mean[1]:.4f}")
    print(f"Kappa : {rf_val_mean[2]:.4f}")
    
    # Test results
    rf_test_mean = [float(np.mean(rf_test_OA)), float(np.mean(rf_test_IoU)), float(np.mean(rf_test_Kappa))]
    print("\n--- Test Set ---")
    print(f"OA    : {rf_test_mean[0]:.4f}")
    print(f"mIoU  : {rf_test_mean[1]:.4f}")
    print(f"Kappa : {rf_test_mean[2]:.4f}")
    
    # Save RF confusion matrices
    plot_confmat(rf_val_cm, "RF Val Confusion Matrix", outdir / "rf_val_confusion_matrix.png")
    plot_confmat(rf_test_cm, "RF Test Confusion Matrix", outdir / "rf_test_confusion_matrix.png")

    if not args.rf_only:
        print("\n" + "="*60)
        print("U-NET RESULTS")
        print("="*60)
        
        # Validation results
        unet_val_mean = [float(np.mean(unet_val_OA)), float(np.mean(unet_val_IoU)), float(np.mean(unet_val_Kappa))]
        print("\n--- Validation Set ---")
        print(f"OA    : {unet_val_mean[0]:.4f}")
        print(f"mIoU  : {unet_val_mean[1]:.4f}")
        print(f"Kappa : {unet_val_mean[2]:.4f}")
        
        # Test results
        unet_test_mean = [float(np.mean(unet_test_OA)), float(np.mean(unet_test_IoU)), float(np.mean(unet_test_Kappa))]
        print("\n--- Test Set ---")
        print(f"OA    : {unet_test_mean[0]:.4f}")
        print(f"mIoU  : {unet_test_mean[1]:.4f}")
        print(f"Kappa : {unet_test_mean[2]:.4f}")
        
        # Save U-Net confusion matrices
        plot_confmat(unet_val_cm, "U-Net Val Confusion Matrix", outdir / "unet_val_confusion_matrix.png")
        plot_confmat(unet_test_cm, "U-Net Test Confusion Matrix", outdir / "unet_test_confusion_matrix.png")
    else:
        unet_val_mean = None
        unet_test_mean = None

    # Save metrics JSON with both val and test
    results = {
        "rf": {
            "validation": {"OA": rf_val_mean[0], "mIoU": rf_val_mean[1], "Kappa": rf_val_mean[2]},
            "test": {"OA": rf_test_mean[0], "mIoU": rf_test_mean[1], "Kappa": rf_test_mean[2]}
        }
    }
    if unet_val_mean is not None and unet_test_mean is not None:
        results["unet"] = {
            "validation": {"OA": unet_val_mean[0], "mIoU": unet_val_mean[1], "Kappa": unet_val_mean[2]},
            "test": {"OA": unet_test_mean[0], "mIoU": unet_test_mean[1], "Kappa": unet_test_mean[2]}
        }
    (outdir / "metrics.json").write_text(json.dumps(results, indent=2))

    # Comparison bar plots (using test set results for main comparison)
    plot_metrics_bar(rf_test_mean, unet_test_mean if not args.rf_only else None, outdir / "comparison_test.png")
    plot_metrics_bar(rf_val_mean, unet_val_mean if not args.rf_only else None, outdir / "comparison_val.png")

    # Save qualitative examples
    num_samples = max(0, int(args.viz_samples))
    for i, (img_path, mask_path) in enumerate(zip(test_imgs, test_masks)):
        if i >= num_samples:
            break
        image_bgr = cv2.imread(img_path)
        gt = load_mask(mask_path)
        rf_pred = rf_predict(rf, img_path)
        unet_pred = None
        if not args.rf_only:
            x = torch.tensor(cv2.resize(image_bgr, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                pred = model(x)
                unet_pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
        save_qualitative(image_bgr, gt, rf_pred, unet_pred, outdir / f"sample_{i:02d}.png")


if __name__ == "__main__":
    main()
