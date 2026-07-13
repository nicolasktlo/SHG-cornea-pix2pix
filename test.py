import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tifffile as tiff
import numpy as np

from dataset import SHGPairDataset
from model import GeneratorUNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print("Using device:", DEVICE)

SPLIT_ROOT = "/Users/nicolaslo/Downloads/SHG research/data/split"
CHECKPOINT_DIR = "/Users/nicolaslo/Downloads/SHG research/checkpoints"
TEST_SAMPLE_DIR = "/Users/nicolaslo/Downloads/SHG research/test_samples"
os.makedirs(TEST_SAMPLE_DIR, exist_ok=True)

BATCH_SIZE = 16
CHECKPOINT_TO_LOAD = os.path.join(CHECKPOINT_DIR, "generator_epoch100.pth")

test_ds = SHGPairDataset(
    os.path.join(SPLIT_ROOT, "test", "backward"),
    os.path.join(SPLIT_ROOT, "test", "forward")
)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

generator = GeneratorUNet().to(DEVICE)
generator.load_state_dict(torch.load(CHECKPOINT_TO_LOAD, map_location=DEVICE))
generator.eval()

criterion_l1 = nn.L1Loss()

def save_sample(idx, real_A, real_B, fake_B):
    real_A_img = ((real_A[0, 0].detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
    real_B_img = ((real_B[0, 0].detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
    fake_B_img = ((fake_B[0, 0].detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
    combined = np.concatenate([real_A_img, real_B_img, fake_B_img], axis=1)
    tiff.imwrite(os.path.join(TEST_SAMPLE_DIR, f"test_{idx:03d}.tif"), combined)

def evaluate():
    total_l1 = 0.0
    per_batch_l1 = []

    with torch.no_grad():
        for i, (real_A, real_B) in enumerate(test_loader):
            real_A, real_B = real_A.to(DEVICE), real_B.to(DEVICE)
            fake_B = generator(real_A)

            loss = criterion_l1(fake_B, real_B).item()
            total_l1 += loss
            per_batch_l1.append(loss)

            save_sample(i, real_A, real_B, fake_B)

    avg_l1 = total_l1 / len(test_loader)
    print(f"Test set size: {len(test_ds)} images, {len(test_loader)} batches")
    print(f"Average Test L1 Loss: {avg_l1:.4f}")

    np.savetxt(os.path.join(CHECKPOINT_DIR, "test_l1_losses.csv"), per_batch_l1, delimiter=",")
    print(f"Per-batch test L1 losses saved to {CHECKPOINT_DIR}/test_l1_losses.csv")
    print(f"Sample comparison images saved to {TEST_SAMPLE_DIR}")

if __name__ == "__main__":
    evaluate()