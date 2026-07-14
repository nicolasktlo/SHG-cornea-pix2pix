import tifffile as tiff
import numpy as np
import glob
import matplotlib.pyplot as plt
import hashlib
import torch

from dataset import SHGPairDataset
from model import GeneratorUNet


# --- Check 1: Normalization consistency across splits ---
for split in ["train", "val", "test"]:
    for channel in ["backward", "forward"]:
        files = glob.glob(f"/Users/nicolaslo/Downloads/SHG research/data/split/{split}/{channel}/*.tif")[:50]
        mins, maxs, means = [], [], []
        for f in files:
            img = tiff.imread(f).astype(np.float32)
            mins.append(img.min())
            maxs.append(img.max())
            means.append(img.mean())
        print(f"{split}/{channel}: min={np.mean(mins):.3f} max={np.mean(maxs):.3f} mean={np.mean(means):.3f}")


# --- Check 2: Duplicate patches ---
def file_hash(path):
    img = tiff.imread(path)
    return hashlib.md5(img.tobytes()).hexdigest()


all_hashes = {}
for split in ["train", "val", "test"]:
    files = glob.glob(f"/Users/nicolaslo/Downloads/SHG research/data/split/{split}/backward/*.tif")
    for f in files:
        h = file_hash(f)
        if h in all_hashes:
            print(f"Duplicate found: {f} matches {all_hashes[h]}")
        all_hashes[h] = f

print(f"Total files checked: {len(all_hashes)}")


# --- Load dataset and trained generator ---
ds = SHGPairDataset(
    "/Users/nicolaslo/Downloads/SHG research/data/split/train/backward",
    "/Users/nicolaslo/Downloads/SHG research/data/split/train/forward"
)

gen = GeneratorUNet()
gen.load_state_dict(torch.load(
    "/Users/nicolaslo/Downloads/SHG research/checkpoints/generator_epoch100.pth",
    map_location="cpu"
))
gen.eval()


# --- Check 3: Output range match (single sample check) ---
real_A, real_B = ds[0]
print("real_A range:", real_A.min().item(), real_A.max().item())
print("real_B range:", real_B.min().item(), real_B.max().item())

with torch.no_grad():
    fake_B = gen(real_A.unsqueeze(0))
print("fake_B range:", fake_B.min().item(), fake_B.max().item())


# --- Check 4: Pixel value histogram comparison (multi-sample) ---
sample_indices = [0, 100, 500, 1000, 2000]

real_B_all = []
fake_B_all = []

for i in sample_indices:
    real_A_i, real_B_i = ds[i]
    with torch.no_grad():
        fake_B_i = gen(real_A_i.unsqueeze(0))
    real_B_all.append(real_B_i.numpy().flatten())
    fake_B_all.append(fake_B_i[0].detach().numpy().flatten())

real_B_np = np.concatenate(real_B_all)
fake_B_np = np.concatenate(fake_B_all)

plt.figure(figsize=(8, 5))
plt.hist(real_B_np, bins=50, alpha=0.5, label="Real forward-SHG", density=True)
plt.hist(fake_B_np, bins=50, alpha=0.5, label="Generated forward-SHG", density=True)
plt.xlabel("Pixel value")
plt.ylabel("Density")
plt.legend()
plt.title(f"Real vs Generated Pixel Value Distribution ({len(sample_indices)} samples)")
plt.savefig("/Users/nicolaslo/Downloads/SHG research/checkpoints/pixel_histogram.png", dpi=150)
plt.show()