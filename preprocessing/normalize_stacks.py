import os
import numpy as np
import tifffile as tiff

BASE = "/Users/nicolaslo/Downloads/SHG research/data/patches_by_stack"
NORM_ROOT = "/Users/nicolaslo/Downloads/SHG research/data/normalized"

def normalize_stack(stack, channel):
    src = os.path.join(BASE, stack, channel)
    out = os.path.join(NORM_ROOT, stack, channel)
    os.makedirs(out, exist_ok=True)

    files = [f for f in os.listdir(src) if f.endswith(".tif")]
    all_pixels = []
    for f in files:
        img = tiff.imread(os.path.join(src, f))
        all_pixels.append(img.flatten())
    all_pixels = np.concatenate(all_pixels)

    p1, p99 = np.percentile(all_pixels, [1, 99])
    print(f"{stack}/{channel}: p1={p1:.1f}, p99={p99:.1f}")

    for f in files:
        img = tiff.imread(os.path.join(src, f)).astype(np.float32)
        norm = np.clip((img - p1) / (p99 - p1), 0, 1)
        tiff.imwrite(os.path.join(out, f), norm.astype(np.float32))

stacks = [s for s in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, s)) and not s.startswith(".")]
for stack in stacks:
    for channel in ["backward", "forward"]:
        normalize_stack(stack, channel)

print("Normalization complete.")