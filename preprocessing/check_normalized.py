import tifffile as tiff
import os

folder = "/Users/nicolaslo/Downloads/SHG research/data/split/train/forward"
sample_file = os.listdir(folder)[0]
img = tiff.imread(os.path.join(folder, sample_file))

print("File:", sample_file)
print("dtype:", img.dtype)
print("min:", img.min(), "max:", img.max())