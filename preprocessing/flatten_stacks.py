import os
import shutil

BASE = "/Users/nicolaslo/Downloads/SHG research/data/patches_by_stack"

groups_to_flatten = ["stack5_post_cornea", "stack6_cxl_posterior"]

for stack in groups_to_flatten:
    stack_path = os.path.join(BASE, stack)
    subfolders = [f for f in os.listdir(stack_path)
                  if os.path.isdir(os.path.join(stack_path, f))]

    for channel in ["backward", "forward"]:
        out_dir = os.path.join(stack_path, channel)
        os.makedirs(out_dir, exist_ok=True)

        for sub in subfolders:
            if sub in ("backward", "forward"):
                continue
            src = os.path.join(stack_path, sub, channel)
            if not os.path.isdir(src):
                continue
            for fname in os.listdir(src):
                new_name = f"{sub}__{fname}"
                shutil.copy(
                    os.path.join(src, fname),
                    os.path.join(out_dir, new_name)
                )
    print(f"Flattened: {stack}")

print("Done.")