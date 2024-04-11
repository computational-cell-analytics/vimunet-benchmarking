import os
import shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path

import json
import numpy as np
import imageio.v3 as imageio

from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_cremi_loader


def _write_dataset_json_file(trg_dir, dataset_name):
    gt_dir = os.path.join(trg_dir, "nnUNet_raw", dataset_name, "labelsTr")

    train_ids = [Path(_path).stem for _path in glob(os.path.join(gt_dir, "*_train.tif"))]
    val_ids = [Path(_path).stem for _path in glob(os.path.join(gt_dir, "*_val.tif"))]

    json_file = os.path.join(trg_dir, "nnUNet_raw", dataset_name, "dataset.json")

    data = {
        "channel_names": {
            "0": "electron_microscopy"
        },
        "labels": {
            "background": 0,
            "boundaries": 1
        },
        "numTraining": len(val_ids) + len(train_ids),
        "file_ending": ".tif",
        "name": dataset_name,
        "description": "CREMI: https://cremi.org/"
    }

    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

    return train_ids, val_ids


def create_json_files(trg_dir, dataset_name):
    # now, let's create the 'dataset.json' file based on the available inputs
    train_ids, val_ids = _write_dataset_json_file(trg_dir=trg_dir, dataset_name=dataset_name)

    # let's try to store the splits file already
    preprocessed_dir = os.path.join(trg_dir, "nnUNet_preprocessed", dataset_name)
    os.makedirs(preprocessed_dir, exist_ok=True)

    # we create custom splits for all folds, to fit with the expectation
    all_split_inputs = [{'train': train_ids, 'val': val_ids} for _ in range(5)]
    json_file = os.path.join(preprocessed_dir, "splits_final.json")
    with open(json_file, "w") as f:
        json.dump(all_split_inputs, f, indent=4)


def convert_cremi_for_training(path, trg_root, dataset_name):
    # for consistency, we get patches of size (512, 512) from all respective
    # splits for train and val

    train_rois = {"A": np.s_[0:75, :, :], "B": np.s_[0:75, :, :], "C": np.s_[0:75, :, :]}
    val_rois = {"A": np.s_[75:100, :, :], "B": np.s_[75:100, :, :], "C": np.s_[75:100, :, :]}

    sampler = MinInstanceSampler()
    patch_shape = (1, 512, 512)

    train_loader = get_cremi_loader(
        path=path,
        patch_shape=patch_shape,
        batch_size=1,
        boundaries=True,
        rois=train_rois,
        ndim=2,
        defect_augmentation_kwargs=None,
        num_workers=16,
        sampler=sampler,
    )

    val_loader = get_cremi_loader(
        path=path,
        patch_shape=patch_shape,
        batch_size=1,
        boundaries=True,
        rois=val_rois,
        ndim=2,
        defect_augmentation_kwargs=None,
        num_workers=16,
        sampler=sampler,
    )

    image_dir = os.path.join(trg_root, "nnUNet_raw", dataset_name, "imagesTr")
    gt_dir = os.path.join(trg_root, "nnUNet_raw", dataset_name, "labelsTr")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # let's get the train data first
    for i, (x, y) in enumerate(train_loader):
        image = x.squeeze().numpy()
        gt = y.squeeze().numpy()

        image_path = os.path.join(image_dir, f"cremi_{i:04}_train_0000.tif")
        imageio.imwrite(image_path, image)

        gt_path = os.path.join(gt_dir, f"cremi_{i:04}_train.tif")
        imageio.imwrite(gt_path, gt)

    # now, let's sort the val data
    for i, (x, y) in enumerate(val_loader):
        image = x.squeeze().numpy()
        gt = y.squeeze().numpy()

        image_path = os.path.join(image_dir, f"cremi_{i:04}_val_0000.tif")
        imageio.imwrite(image_path, image)

        gt_path = os.path.join(gt_dir, f"cremi_{i:04}_val.tif")
        imageio.imwrite(gt_path, gt)


def convert_cremi_for_testing(path, trg_dir, dataset_name):
    # for consistency, I use the already generated splits for testing
    image_paths = sorted(glob(os.path.join(path, "slices_original", "raw", "cremi_test*")))
    gt_paths = sorted(glob(os.path.join(path, "slices_original", "labels", "cremi_test*")))

    assert len(image_paths) == len(gt_paths)

    # the idea for here is to move the data to a central location,
    # where we can automate the inference procedure
    image_dir = os.path.join(trg_dir, "test", dataset_name, "imagesTs")
    gt_dir = os.path.join(trg_dir, "test", dataset_name, "labelsTs")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
        image_id = Path(image_path).stem

        trg_image_path = os.path.join(image_dir, f"{image_id}_0000.tif")
        shutil.copy(src=image_path, dst=trg_image_path)

        trg_gt_path = os.path.join(gt_dir, f"{image_id}.tif")
        shutil.copy(src=gt_path, dst=trg_gt_path)


def main():
    path = "/scratch/projects/nim00007/sam/data/cremi"
    dataset_name = "Dataset305_CREMI"

    # space to store your top-level nnUNet files
    # trg_root = "/scratch/usr/nimanwai/experiments/nnunetv2_neurips_cellseg/"  # for nnUNetv2
    trg_root = "/scratch/usr/nimanwai/experiments/U-Mamba/data"  # for U-Mamba

    # convert_cremi_for_training(path, trg_root, dataset_name)
    # create_json_files(trg_root, dataset_name)

    convert_cremi_for_testing(path, trg_root, dataset_name)


if __name__ == "__main__":
    main()
