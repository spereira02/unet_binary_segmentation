import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np
from utils import IMAGE_SIZE, load_mask

VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _list_image_files(directory):
    return sorted(
        os.path.join(directory, fname)
        for fname in os.listdir(directory)
        if fname.lower().endswith(VALID_IMAGE_EXTENSIONS)
    )


def _sample_id(path, suffix):
    filename = os.path.splitext(os.path.basename(path))[0]
    if not filename.endswith(suffix):
        raise ValueError(f"Expected filename ending with '{suffix}', got '{filename}'")
    return filename[: -len(suffix)]


class ETHMugsDataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported mode '{mode}'")

        self.mode = mode
        self.root_dir = root_dir
        self.rgb_dir = os.path.join(self.root_dir, "rgb")

        self.image_paths = _list_image_files(self.rgb_dir)
        self.image_stems = [_sample_id(path, "_rgb") for path in self.image_paths]

        self.mask_dir = None
        self.mask_paths = None
        if mode != "test":
            self.mask_dir = os.path.join(self.root_dir, "masks")
            self.mask_paths = _list_image_files(self.mask_dir)
            self.mask_stems = [_sample_id(path, "_mask") for path in self.mask_paths]

            if self.image_stems != self.mask_stems:
                raise ValueError(
                    "Image/mask pairs do not align in "
                    f"{self.root_dir}: found {len(self.image_paths)} images and {len(self.mask_paths)} masks"
                )

            self.mask_transform = transforms.Compose(
                [
                    transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
                    transforms.ToTensor(),
                ]
            )

        self.image_resize = transforms.Resize(
            IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR
        )
        self.image_to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.color_jitter = transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
        )
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.vertical_flip = transforms.RandomVerticalFlip(p=1.0)

        print(f"[INFO]: Loaded {len(self.image_paths)} {mode} samples from {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.image_resize(image)

        if self.mode != "test":
            mask_path = self.mask_paths[idx]
            mask = load_mask(mask_path)
            mask = Image.fromarray(mask.astype(np.uint8) * 255)
            mask = self.mask_transform(mask)

            if self.mode == "train":
                if torch.rand(1).item() < 0.5:
                    image = self.horizontal_flip(image)
                    mask = self.horizontal_flip(mask)
                if torch.rand(1).item() < 0.5:
                    image = self.vertical_flip(image)
                    mask = self.vertical_flip(mask)
                image = self.color_jitter(image)

            image = self.image_to_tensor(image)

            mask = torch.squeeze(mask, 0)
            mask = (mask > 0.5).float()

            return image, mask

        image = self.image_to_tensor(image)
        pred_filename = os.path.basename(image_path).replace("_rgb", "_mask").rsplit(".", 1)[0] + ".png"
        return image, pred_filename
