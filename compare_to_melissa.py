import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.exposure import equalize_adapthist
from scipy.ndimage import median_filter
from scipy.spatial.distance import directed_hausdorff

from dataset import SHGPairDataset
from model import GeneratorUNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print("Using device:", DEVICE)

SPLIT_ROOT = "/Users/nicolaslo/Downloads/SHG research/data/split"
CHECKPOINT_DIR = "/Users/nicolaslo/Downloads/SHG research/checkpoints"
CHECKPOINT_TO_LOAD = os.path.join(CHECKPOINT_DIR, "generator_epoch100.pth")

BATCH_SIZE = 16
CLAHE_CLIP_LIMIT = 0.3
THRESHOLD = 0.7
MEDIAN_KERNEL = 3

test_ds = SHGPairDataset(
    os.path.join(SPLIT_ROOT, "test", "backward"),
    os.path.join(SPLIT_ROOT, "test", "forward")
)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

generator = GeneratorUNet().to(DEVICE)
generator.load_state_dict(torch.load(CHECKPOINT_TO_LOAD, map_location=DEVICE))
generator.eval()

def to_unit_range(img_tensor):
    img = img_tensor.detach().cpu().numpy()
    return (img + 1) / 2

def binarize(img_01):
    img_clahe = equalize_adapthist(img_01, clip_limit=CLAHE_CLIP_LIMIT)
    binary = (img_clahe >= THRESHOLD).astype(np.uint8)
    binary = median_filter(binary, size=MEDIAN_KERNEL)
    return binary

def dice_score(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    return (2.0 * intersection) / (pred.sum() + gt.sum() + 1e-8)

def iou_score(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-8)

def hd95(pred, gt):
    pred_pts = np.argwhere(pred > 0)
    gt_pts = np.argwhere(gt > 0)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan
    d1 = directed_hausdorff(pred_pts, gt_pts)[0]
    d2 = directed_hausdorff(gt_pts, pred_pts)[0]
    return max(d1, d2)

def block_avg_diff(pred, gt, block_size=16):
    h, w = pred.shape
    diffs = []
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            p_block = pred[i:i+block_size, j:j+block_size].mean()
            g_block = gt[i:i+block_size, j:j+block_size].mean()
            diffs.append(abs(p_block - g_block))
    return np.mean(diffs)

dice_scores, iou_scores, hd95_scores, block_diffs = [], [], [], []

with torch.no_grad():
    for real_A, real_B in test_loader:
        real_A, real_B = real_A.to(DEVICE), real_B.to(DEVICE)
        fake_B = generator(real_A)

        for b in range(real_B.shape[0]):
            gt_img = to_unit_range(real_B[b, 0])
            pred_img = to_unit_range(fake_B[b, 0])

            gt_binary = binarize(gt_img)
            pred_binary = binarize(pred_img)

            dice_scores.append(dice_score(pred_binary, gt_binary))
            iou_scores.append(iou_score(pred_binary, gt_binary))
            hd95_scores.append(hd95(pred_binary, gt_binary))
            block_diffs.append(block_avg_diff(pred_binary, gt_binary))

dice_scores = np.array(dice_scores)
iou_scores = np.array(iou_scores)
hd95_scores = np.array(hd95_scores)
block_diffs = np.array(block_diffs)

print(f"\nResults on {len(dice_scores)} test images:")
print(f"Dice score: {dice_scores.mean():.3f} ± {dice_scores.std():.4f}")
print(f"IoU: {iou_scores.mean():.3f} ± {iou_scores.std():.4f}")
print(f"HD95: {np.nanmean(hd95_scores):.2f}px ± {np.nanstd(hd95_scores):.2f}px")
print(f"Block-averaged difference: {block_diffs.mean():.3f} ± {block_diffs.std():.4f}")

print(f"\nMelissa's reported results: Dice 0.609 ± 0.0087, IoU 0.438 ± 0.0090, HD95 5.50px ± 0.64px, Block diff 0.250 ± 0.0098")