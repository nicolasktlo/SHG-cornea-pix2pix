import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tifffile as tiff
import numpy as np

from dataset import SHGPairDataset
from model import GeneratorUNet, Discriminator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print("Using device:", DEVICE)

SPLIT_ROOT = "/Users/nicolaslo/Downloads/SHG research/data/split"
CHECKPOINT_DIR = "/Users/nicolaslo/Downloads/SHG research/checkpoints"
SAMPLE_DIR = "/Users/nicolaslo/Downloads/SHG research/samples"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 100
LR = 2e-4
LAMBDA_L1 = 100

train_ds = SHGPairDataset(
    os.path.join(SPLIT_ROOT, "train", "backward"),
    os.path.join(SPLIT_ROOT, "train", "forward")
)

val_ds = SHGPairDataset(
    os.path.join(SPLIT_ROOT, "val", "backward"),
    os.path.join(SPLIT_ROOT, "val", "forward")
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, persistent_workers=True)

generator = GeneratorUNet().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

criterion_gan = nn.BCEWithLogitsLoss()
criterion_l1 = nn.L1Loss()

opt_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

def save_sample(epoch, real_A, real_B, fake_B):
    real_A_img = ((real_A[0, 0].detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
    real_B_img = ((real_B[0, 0].detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
    fake_B_img = ((fake_B[0, 0].detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)

    combined = np.concatenate([real_A_img, real_B_img, fake_B_img], axis=1)
    tiff.imwrite(os.path.join(SAMPLE_DIR, f"epoch_{epoch:03d}.tif"), combined)

def train():
    g_losses, d_losses, train_l1_losses, val_losses = [], [], [], []
    training_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        generator.train()
        discriminator.train()
        epoch_g_loss, epoch_d_loss = 0.0, 0.0

        for real_A, real_B in train_loader:
            real_A, real_B = real_A.to(DEVICE), real_B.to(DEVICE)

            fake_B = generator(real_A)

            opt_d.zero_grad()
            pred_real = discriminator(real_A, real_B)
            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            pred_fake = discriminator(real_A, fake_B.detach())
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad()
            pred_fake = discriminator(real_A, fake_B)
            loss_g_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            loss_g_l1 = criterion_l1(fake_B, real_B) * LAMBDA_L1
            loss_g = loss_g_gan + loss_g_l1
            loss_g.backward()
            opt_g.step()

            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()

        avg_g = epoch_g_loss / len(train_loader)
        avg_d = epoch_d_loss / len(train_loader)
        g_losses.append(avg_g)
        d_losses.append(avg_d)

        generator.eval()

        train_l1 = 0.0
        with torch.no_grad():
            for real_A, real_B in train_loader:
                real_A, real_B = real_A.to(DEVICE), real_B.to(DEVICE)
                fake_B = generator(real_A)
                train_l1 += criterion_l1(fake_B, real_B).item()
        avg_train_l1 = train_l1 / len(train_loader)
        train_l1_losses.append(avg_train_l1)

        val_l1 = 0.0
        with torch.no_grad():
            for real_A, real_B in val_loader:
                real_A, real_B = real_A.to(DEVICE), real_B.to(DEVICE)
                fake_B = generator(real_A)
                val_l1 += criterion_l1(fake_B, real_B).item()
        avg_val = val_l1 / len(val_loader)
        val_losses.append(avg_val)

        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - training_start
        eta = epoch_time * (EPOCHS - epoch)

        print(f"Epoch {epoch}/{EPOCHS} | G loss: {avg_g:.4f} | D loss: {avg_d:.4f} | "
              f"Train L1: {avg_train_l1:.4f} | Val L1: {avg_val:.4f} | "
              f"Epoch time: {epoch_time:.1f}s | Elapsed: {elapsed_total/60:.1f}m | ETA: {eta/60:.1f}m")

        if epoch % 5 == 0 or epoch == 1:
            save_sample(epoch, real_A, real_B, fake_B)
            torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, f"generator_epoch{epoch}.pth"))

    np.savetxt(os.path.join(CHECKPOINT_DIR, "g_losses.csv"), g_losses, delimiter=",")
    np.savetxt(os.path.join(CHECKPOINT_DIR, "d_losses.csv"), d_losses, delimiter=",")
    np.savetxt(os.path.join(CHECKPOINT_DIR, "train_l1_losses.csv"), train_l1_losses, delimiter=",")
    np.savetxt(os.path.join(CHECKPOINT_DIR, "val_losses.csv"), val_losses, delimiter=",")

if __name__ == "__main__":
    train()