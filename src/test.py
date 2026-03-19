import argparse
import os
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch

from eth_mugs_dataset import ETHMugsDataset
from model import build_model
from utils import IMAGE_SIZE, load_mask, compute_iou

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SML Project 2.")
    parser.add_argument(
        "-d",
        "--data_root",
        default="./datasets",
        type=str,
        help="Path to the datasets folder.",
    )
    parser.add_argument(
        "-s",
        "--split",
        choices=["public_test", "private_test"],
        default="private_test",
        help="Choose the data split. If using public test, then your model will also be evaluated.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to a checkpoint file or a checkpoint directory containing checkpoint.pth.",
    )
    args = parser.parse_args()

    if args.split == "public_test":
        test_data_root = os.path.join(args.data_root, "public_test_images_378_252")
        out_dir = os.path.join("public_test", "prediction")
    else:
        test_data_root = os.path.join(args.data_root, "private_test_images_378_252")
        out_dir = os.path.join("private_test", "prediction")
    print(f"[INFO]: Test data root: {test_data_root}")

    # Set output directory
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO]: Saving the predicted segmentation masks to {out_dir}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  
    model = build_model()  
    model.to(device) 

    # Load model checkpoint
    checkpoint_path = args.ckpt
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "checkpoint.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"[INFO]: Loaded checkpoint from {checkpoint_path}")
    model.eval()

    test_dataset = ETHMugsDataset(root_dir=test_data_root, mode="test")

    # Create dataloaders
    test_batch_size = 1
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False
    )

    with torch.no_grad():
        for test_image, pred_filename in tqdm(test_dataloader):
            test_image = test_image.to(device)

            test_logits = model(test_image)
            test_output = (torch.sigmoid(test_logits) > 0.5).float()

            pred_mask = test_output.squeeze().cpu().numpy().astype(np.uint8) * 255

            resized_pred_mask = Image.fromarray(pred_mask).resize(
                (IMAGE_SIZE[1], IMAGE_SIZE[0]), resample=Image.NEAREST
            )

            resized_pred_mask.save(os.path.join(out_dir, pred_filename[0]))

    # Run evaluation if using public test split
    if args.split == "public_test":
        gt_dir = os.path.join(args.data_root, "public_test_images_378_252", "masks")

        # Load GT and prediction mask filenames
        gt_mask_filenames = sorted([el for el in os.listdir(gt_dir) if el.endswith("_mask.png")])
        pred_mask_filenames = sorted([el for el in os.listdir(out_dir) if el.endswith("_mask.png")])

        # Ensure that the predictions are saved with the same file names as the GT files
        assert (
            gt_mask_filenames == pred_mask_filenames
        ), "Predictions must have been saved with the same file names"

        num_samples_to_evaluate = len(gt_mask_filenames)

        test_iou_sum = 0.0
        for idx in tqdm(range(num_samples_to_evaluate)):
            gt_mask_path = os.path.join(gt_dir, gt_mask_filenames[idx])
            pred_mask_path = os.path.join(out_dir, pred_mask_filenames[idx])

            gt_mask = load_mask(gt_mask_path)
            pred_mask = load_mask(pred_mask_path)

            iou = compute_iou(pred_mask, gt_mask)

            test_iou_sum += iou

        average_test_iou = test_iou_sum / num_samples_to_evaluate
        print(f"[INFO]: Average IoU: {average_test_iou}")
