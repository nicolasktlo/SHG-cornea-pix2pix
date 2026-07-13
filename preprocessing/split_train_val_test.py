import os
import shutil

BASE = "/Users/nicolaslo/Downloads/SHG research/data/normalized"
SPLIT_ROOT = "/Users/nicolaslo/Downloads/SHG research/data/split"

split_assignment = {
    "stack1_500um_cornea10x": "train",
    "stack2_first100um_top": "train",
    "stack3_lenticule": "train",
    "stack5_post_cornea": "train",
    "stack4_small_area": "val",
    "stack6_cxl_posterior": "test",
}

for stack, split in split_assignment.items():
    for channel in ["backward", "forward"]:
        src = os.path.join(BASE, stack, channel)
        out_dir = os.path.join(SPLIT_ROOT, split, channel)
        os.makedirs(out_dir, exist_ok=True)

        for fname in os.listdir(src):
            new_name = f"{stack}__{fname}"
            shutil.copy(
                os.path.join(src, fname),
                os.path.join(out_dir, new_name)
            )
    print(f"Copied {stack} -> {split}")

print("Done.")