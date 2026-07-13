import os
import numpy as np
import tifffile as tiff

BASE = "/Users/nicolaslo/Downloads/SHG research/data/patches_by_stack"
EMPTY_THRESHOLD = 5  # max pixel value below this = likely empty patch

def analyze_folder(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".tif")]
    if not files:
        return None

    maxes = []
    means = []
    empty_count = 0

    for f in files:
        img = tiff.imread(os.path.join(folder, f))
        m = img.max()
        maxes.append(m)
        means.append(img.mean())
        if m <= EMPTY_THRESHOLD:
            empty_count += 1

    return {
        "total": len(files),
        "empty": empty_count,
        "pct_empty": 100 * empty_count / len(files),
        "avg_max": np.mean(maxes),
        "avg_mean": np.mean(means),
        "dtype": img.dtype,
    }

print(f"{'Stack':30} {'Channel':10} {'Total':>7} {'Empty':>7} {'%Empty':>8} {'AvgMax':>10} {'AvgMean':>10}")

for stack in sorted(os.listdir(BASE)):
    stack_path = os.path.join(BASE, stack)
    if not os.path.isdir(stack_path):
        continue
    for channel in ["backward", "forward"]:
        ch_path = os.path.join(stack_path, channel)
        if not os.path.isdir(ch_path):
            continue
        stats = analyze_folder(ch_path)
        if stats is None:
            continue
        print(f"{stack:30} {channel:10} {stats['total']:>7} {stats['empty']:>7} "
              f"{stats['pct_empty']:>7.1f}% {stats['avg_max']:>10.1f} {stats['avg_mean']:>10.2f}")