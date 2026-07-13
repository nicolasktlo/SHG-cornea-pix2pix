import tifffile as tiff
import numpy as np
from PIL import Image
import os

folder = "/Users/nicolaslo/Downloads/SHG research/data/patches_by_stack/stack6_cxl_posterior/forward"
files = os.listdir(folder)
first_file = files[0]  # just grab the first one automatically

img = tiff.imread(os.path.join(folder, first_file))
print("Using file:", first_file)
print("min:", img.min(), "max:", img.max())

scaled = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
Image.fromarray(scaled).save("/Users/nicolaslo/Downloads/SHG research/test_view.png")
print("Saved preview to test_view.png")