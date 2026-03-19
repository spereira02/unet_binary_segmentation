import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm
import numpy as np

from eth_mugs_dataset import ETHMugsDataset
from model import build_model
from utils import compute_iou

class EarlyStopping:
    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_iou_max = -np.inf

    def __call__(self, val_iou, model):
        score = val_iou

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_iou, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_iou, model)
            self.counter = 0

    def save_checkpoint(self, val_iou, model):
        if self.verbose:
            print(
                f"Validation IOU increased ({self.val_iou_max:.6f} --> {val_iou:.6f}). Saving model ..."
            )
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_iou_max = val_iou

def split_train_val_dataset(data_root: str, val_split: float, seed: int):
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")

    train_dataset = ETHMugsDataset(data_root, mode="train")
    val_dataset = ETHMugsDataset(data_root, mode="val")

    dataset_size = len(train_dataset)
    val_size = max(1, int(dataset_size * val_split))
    train_size = dataset_size - val_size
    if train_size <= 0:
        raise ValueError("Validation split is too large for the dataset size")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return Subset(train_dataset, train_indices), Subset(val_dataset, val_indices)


def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(
    ckpt_dir: str,
    data_root: str,
    val_split: float,
    seed: int,
):
    val_batch_size = 1
    val_frequency = 1

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(seed)

    train_dataset, val_dataset = split_train_val_dataset(data_root, val_split, seed)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = build_model().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=10
    )
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(ckpt_dir, "checkpoint.pth"),
        patience=7,
        verbose=True,
    )

    epoch_num = 50
    print(f"[INFO]: Number of training epochs: {epoch_num}")
    print("[INFO]: Starting training...")
    for epoch in range(epoch_num):
        model.train()
        running_loss = 0.0

        for images, gt_mask in tqdm(train_loader):
            images, gt_mask = images.to(device), gt_mask.to(device)
            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss = criterion(logits, gt_mask)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pth"))

        if epoch % val_frequency == 0:
            model.eval()
            val_iou = 0.0
            with torch.no_grad():
                for val_image, val_gt_mask in val_loader:
                    val_image = val_image.to(device)
                    val_gt_mask = val_gt_mask.to(device)
                    val_probs = torch.sigmoid(model(val_image))
                    binVal_output = (val_probs > 0.5).float()
                    val_output_np = binVal_output.cpu().numpy().astype(int)
                    val_gt_mask_np = val_gt_mask.cpu().numpy().astype(int)
                    val_iou += compute_iou(val_output_np, val_gt_mask_np)
                val_iou /= len(val_loader)
            print(f"[INFO]: Validation IoU: {val_iou:.4f}")
            lr_scheduler.step(val_iou)

            early_stopping(val_iou, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SML Project 2.")
    parser.add_argument(
        "-d",
        "--data_root",
        default="./datasets",
        help="Path to the datasets folder.",
    )
    parser.add_argument(
        "--ckpt_dir",
        default="./checkpoints",
        help="Path to save the model checkpoints to.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Proportion of dataset to include in validation split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/validation splitting and training initialization.",
    )
    args = parser.parse_args()

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    ckpt_dir = os.path.join(args.ckpt_dir, dt_string)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("[INFO]: Model checkpoints will be saved to:", ckpt_dir)

    # Set data root
    train_data_root = os.path.join(args.data_root, "train_images_378_252")
    print(f"[INFO]: Train data root: {train_data_root}")
    print(f"[INFO]: Validation split: {args.val_split}")

    train(ckpt_dir, train_data_root, args.val_split, args.seed)
