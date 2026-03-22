#!/usr/bin/env python3

import os
from pathlib import Path
import rasterio
import numpy as np
import albumentations as A
from tqdm import tqdm
import argparse

N_AUG = 5  # Fixed number of augmentations per image

def parse_args():
    parser = argparse.ArgumentParser(description="Augment georeferenced images and masks using Albumentations.")
    parser.add_argument("-input_img", type=str, required=True,
                        help="Path to the directory containing input images (.tif)")
    parser.add_argument("-input_mask", type=str, required=True,
                        help="Path to the directory containing input masks (.tif)")
    parser.add_argument("-outpt_img", type=str, required=True,
                        help="Directory to save augmented images")
    parser.add_argument("-output_masks", type=str, required=True,
                        help="Directory to save augmented masks")
    return parser.parse_args()

def main():
    args = parse_args()

    input_img_dir = Path(args.input_img)
    input_mask_dir = Path(args.input_mask)
    output_img_dir = Path(args.outpt_img)
    output_mask_dir = Path(args.output_masks)

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    # Albumentations transform
    transform = A.Compose([
        A.Rotate(limit=30, border_mode=0, interpolation=0, value=0, mask_value=0, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.GaussianBlur(blur_limit=(3,7), p=0.3),
        A.GaussNoise(var_limit=(10.0,50.0), p=0.3)
    ], additional_targets={"mask": "mask"})

    img_files = sorted(input_img_dir.glob("*.tif"))

    for img_file in tqdm(img_files, desc="Augmenting images"):
        img_name = img_file.stem
        mask_file = input_mask_dir / f"{img_name}.tif"

        # Read image
        try:
            with rasterio.open(img_file) as src:
                img = src.read()
                profile = src.profile
                img = np.transpose(img, (1, 2, 0))
        except Exception as e:
            print(f"[WARN] Cannot read image {img_file}: {e}")
            continue

        # Read mask
        if mask_file.exists():
            try:
                with rasterio.open(mask_file) as src:
                    mask = src.read(1)
            except Exception as e:
                print(f"[WARN] Cannot read mask {mask_file}: {e}")
                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        # Generate augmented samples
        for i in range(N_AUG):
            augmented = transform(image=img, mask=mask)
            aug_img = augmented["image"]
            aug_mask = augmented["mask"]

            # Save augmented image
            profile.update(dtype=rasterio.uint8, count=aug_img.shape[2])
            with rasterio.open(output_img_dir / f"{img_name}_aug{i}.tif", "w", **profile) as dst:
                for b in range(aug_img.shape[2]):
                    dst.write(aug_img[:,:,b], b+1)

            # Save augmented mask with .mask.tif extension
            profile.update(dtype=rasterio.uint8, count=1)
            mask_filename = f"{img_name}_aug{i}.mask.tif"
            with rasterio.open(output_mask_dir / mask_filename, "w", **profile) as dst:
                dst.write(aug_mask, 1)

if __name__ == "__main__":
    main()
