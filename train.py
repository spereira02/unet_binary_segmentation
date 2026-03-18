import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from eth_mugs_dataset import ETHMugsDataset
from utils import compute_iou

def visualize_sample(image, gt_mask, pred_mask, return_pred_tensor=False):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # If the image is in CHW format, transpose it to HWC
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)

    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    if gt_mask.ndim == 3 and gt_mask.shape[0] == 1:
        gt_mask = gt_mask.squeeze(0)

    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
        pred_mask = pred_mask.squeeze(0)

    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    plt.show()

    if return_pred_tensor:
        return pred_mask

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

            #do convolution twice, 
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), #convoltion operation on picture
            nn.BatchNorm2d(mid_channels), # to enhance training stability and efficiency.
            nn.ReLU(inplace=True), #outputs input if positiv or 0 if negative, preventing vanishing gradient problem
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),                #take the max value inside a 2x2 cell 4 values in, out 1 max val
                                            #sample down in spatial dimension and increase 
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DeepUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(DeepUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        

        
        self.inc = DoubleConv(n_channels, 64) #asssemble unet first convulte 2x, downsample
        self.down1 = Down(64, 128) #reduce spatial dimensions increse receptive field, by increasing the feature cchannels. Context of objects
        #from pixel level of understanding to full image context understanding.
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 2048)
        self.up1 = Up(2048, 1024, bilinear=bilinear) #when upsamping we increase spatial dimensions again creating segmentation map.
        self.up2 = Up(1024, 512, bilinear=bilinear)
        self.up3 = Up(512, 256, bilinear=bilinear)
        self.up4 = Up(256, 128, bilinear=bilinear)
        self.up5 = Up(128, 64, bilinear=bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.up1(x6, x5)
        x8 = self.up2(x7, x4)
        x9 = self.up3(x8, x3)
        x10 = self.up4(x9, x2)
        x11 = self.up5(x10, x1)
        logits = self.outc(x11)
        return torch.sigmoid(logits)

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_iou_max = -np.Inf

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
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation IOU increased ({self.val_iou_max:.6f} --> {val_iou:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pth')
        self.val_iou_max = val_iou

def build_model():
    model = DeepUNet(3, 1)
    return model

def train(
    ckpt_dir: str,
    train_data_root: str,
    val_data_root: str,
):
    log_frequency = 10
    val_batch_size = 1
    val_frequency = 1
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #define train and val datasets defining root directory and mode train and val
    train_dataset = ETHMugsDataset(train_data_root, mode='train')
    val_dataset = ETHMugsDataset(val_data_root, mode='val')

    #DataLoader 
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)       
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    model = build_model().to(device)
 
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,patience=10)
    early_stopping = EarlyStopping(patience=7, verbose=True)

    print(f"[INFO]: Number of training epochs: {20}")
    print("[INFO]: Starting training...")
    epoch_num = 50
    for epoch in range(epoch_num):
        model.train()
        running_loss = 0.0

        for images, gt_mask in tqdm(train_loader):
            images, gt_mask = images.to(device), gt_mask.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.squeeze()
            #print(outputs.shape)
            #print(gt_mask.shape)
            loss = criterion(outputs, gt_mask)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Visualize every iteration in the last epoch
            if epoch == epoch_num - 1:
                visualize_sample(images[0].cpu().numpy(), gt_mask[0].cpu().numpy(), outputs[0].cpu().detach().numpy())

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")
        lr_scheduler.step(running_loss)
        # Save the model checkpoint
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pth"))
        

        if (epoch) % val_frequency == 0:
            model.eval()
            val_iou = 0.0
            with torch.no_grad():
                for val_image, val_gt_mask in val_loader:
                    val_image = val_image.to(device)
                    val_gt_mask = val_gt_mask.to(device)
                    val_outputs = model(val_image)
                    binVal_output = (val_outputs > 0.5).float()
                    binVal_mask = (val_gt_mask > 0.5).float()
                    val_output_np = binVal_output.cpu().numpy().astype(int)
                    val_gt_mask_np = binVal_mask.cpu().numpy().astype(int)
                    val_iou += compute_iou(val_output_np, val_gt_mask_np)
                val_iou /= len(val_loader)
            print(f"[INFO]: Validation IoU: {val_iou:.4f}")

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

    val_data_root = os.path.join(args.data_root, "public_test_images_378_252")
    print(f"[INFO]: Validation data root: {val_data_root}")

    train(ckpt_dir, train_data_root, val_data_root)

