import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from utils import IMAGE_SIZE, load_mask

class ETHMugsDataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        self.mode = mode
        self.root_dir = root_dir
        self.rgb_dir = os.path.join(self.root_dir, 'rgb')

        # Paths for images
        self.image_paths = [os.path.join(self.rgb_dir, fname) for fname in os.listdir(self.rgb_dir)]
        self.image_paths.sort()

        print(f"[DEBUG] Looking for images in: {self.rgb_dir}")

        self.mask_dir = None
        self.mask_paths = None
        self.mask_transform = None
        if mode != "test":
            # Paths for masks
            self.mask_dir = os.path.join(self.root_dir, 'masks')
            self.mask_paths = [os.path.join(self.mask_dir, fname) for fname in os.listdir(self.mask_dir)]
            self.mask_paths.sort()

            print(f"[DEBUG] Looking for masks in: {self.mask_dir}")

            self.mask_transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor()
            ])

        self.images_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            # transforms.RandomGrayscale(p=0.2)
        ])

        self.augmentation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

        print("[INFO] Dataset mode:", mode)
        print("[INFO] Number of images in the ETHMugDataset: {}".format(len(self.image_paths)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Get an item from the dataset."""
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Load mask
        if self.mode != "test":
            mask_path = self.mask_paths[idx]
            mask = load_mask(mask_path)
            mask = Image.fromarray(mask)

            
            seed = np.random.randint(2147483647)  # Generate random seed
            torch.manual_seed(seed)
            image = self.augmentation_transform(image)
            torch.manual_seed(seed)
            mask = self.augmentation_transform(mask)

            image = self.images_transform(image)
            mask = self.mask_transform(mask)

            # Ensure the mask has the correct dimensions (H, W)
            mask = torch.squeeze(mask, 0)

            # Convert mask to binary (0 for background, 1 for mug)
            mask = (mask > 0.5).float()

            return image, mask

        image = self.images_transform(image)

        return image
