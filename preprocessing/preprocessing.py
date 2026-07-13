import os
import numpy as np
import tifffile as tiff

PATCH_SIZE = 256
BASE_DIR = "/Users/nicolaslo/Downloads/SHG research"
DATA_DIR = os.path.join(BASE_DIR, "data", "tif files")
PATCH_ROOT = os.path.join(BASE_DIR, "data", "patches_by_stack")


def extract_patches(image_path, stack_name, channel_subdir):
    img = tiff.imread(image_path)

    if img.ndim == 2:
        img = img[np.newaxis, ...]

    z_slices, H, W = img.shape
    n_ph = H // PATCH_SIZE
    n_pw = W // PATCH_SIZE

    # e.g. patches_by_stack/F+B+Lambda_starting_from_500um/backward/
    out_root = os.path.join(PATCH_ROOT, stack_name, channel_subdir)
    os.makedirs(out_root, exist_ok=True)

    print(f"  [{stack_name}/{channel_subdir}] {z_slices} slices, {n_ph*n_pw} patches/slice")

    for z in range(z_slices):
        slice_img = img[z]
        for i in range(n_ph):
            for j in range(n_pw):
                patch = slice_img[
                    i*PATCH_SIZE:(i+1)*PATCH_SIZE,
                    j*PATCH_SIZE:(j+1)*PATCH_SIZE,
                ]
                out_name = f"z{z:03d}_r{i}_c{j}.tif"
                tiff.imwrite(os.path.join(out_root, out_name), patch)


def clean_stack_name(fname):
    """Turn a backward filename into a safe stack folder name."""
    name = fname.replace("_ch2_backwards.tif", "")
    name = name.strip().replace("/", "-")
    return name


def main():
    pairs_found = 0
    pairs_processed = 0

    for root, dirs, files in os.walk(DATA_DIR):
        for fname in files:
            if not fname.endswith("_ch2_backwards.tif"):
                continue

            pairs_found += 1
            backward_path = os.path.join(root, fname)
            forward_name = fname.replace("_ch2_backwards.tif", "_ch5_forwards.tif")
            forward_path = os.path.join(root, forward_name)

            if not os.path.exists(forward_path):
                print(f"SKIP (no forward match): {fname}")
                continue

            stack_name = clean_stack_name(fname)
            print(f"Processing stack: {stack_name}")

            extract_patches(backward_path, stack_name, "backward")
            extract_patches(forward_path, stack_name, "forward")
            pairs_processed += 1

    print(f"\nFound {pairs_found} backward files, processed {pairs_processed} matched pairs.")


if __name__ == "__main__":
    main()
