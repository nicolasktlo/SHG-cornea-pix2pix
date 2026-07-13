import numpy as np
import matplotlib.pyplot as plt

checkpoint_dir = "/Users/nicolaslo/Downloads/SHG research/checkpoints"

g_losses = np.loadtxt(f"{checkpoint_dir}/g_losses.csv", delimiter=",")
d_losses = np.loadtxt(f"{checkpoint_dir}/d_losses.csv", delimiter=",")
train_l1 = np.loadtxt(f"{checkpoint_dir}/train_l1_losses.csv", delimiter=",")
val_l1 = np.loadtxt(f"{checkpoint_dir}/val_losses.csv", delimiter=",")

epochs = np.arange(1, len(g_losses) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(epochs, g_losses, label="Generator loss")
axes[0].plot(epochs, d_losses, label="Discriminator loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Adversarial Losses")
axes[0].legend()

axes[1].plot(epochs, train_l1, label="Train L1")
axes[1].plot(epochs, val_l1, label="Validation L1")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("L1 Loss")
axes[1].set_title("L1 Reconstruction Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{checkpoint_dir}/loss_curves.png", dpi=150)
plt.show()