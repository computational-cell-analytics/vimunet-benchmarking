import os
import shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path

import json
import numpy as np
import imageio.v3 as imageio
from skimage.segmentation import find_boundaries

SRC_ROOT = "/scratch/usr/nimanwai/data/neurips-cell-seg/umamba/Dataset703_NeurIPSCell"
# data comes from: https://drive.google.com/drive/folders/18QSSiABS8H3qtx8SZA6RQb3aH1nbc3iF


def convert_inputs(split, src_dir, trg_dir):
    if split == "train":
        image_dir, label_dir = "imagesTr", "labelsTr"
    elif split == "val":
        image_dir, label_dir = "imagesVal", "labelsVal-instance-mask"
    else:
        raise ValueError

    all_images = sorted(glob(os.path.join(src_dir, image_dir, "*")))
    all_labels = sorted(glob(os.path.join(src_dir, label_dir, "*")))
    assert len(all_images) == len(all_labels)

    trg_image_dir = os.path.join(trg_dir, "nnUNet_raw", "Dataset703_NeurIPSCell", "imagesTr")
    trg_label_dir = os.path.join(trg_dir, "nnUNet_raw", "Dataset703_NeurIPSCell", "labelsTr")
    os.makedirs(trg_image_dir, exist_ok=True)
    os.makedirs(trg_label_dir, exist_ok=True)

    for src_image_path, src_label_path in tqdm(zip(all_images, all_labels), total=len(all_images)):
        image_fname = os.path.split(src_image_path)[-1]
        image_suffix = os.path.splitext(image_fname)[-1]
        label_fname = os.path.split(src_label_path)[-1]

        trg_image_path = os.path.join(trg_image_dir, f"{split}_{image_fname}")
        trg_label_path = os.path.join(trg_label_dir, f"{split}_{label_fname}")

        img = imageio.imread(src_image_path)
        gt = imageio.imread(src_label_path)

        assert gt.ndim == 2
        assert img.ndim == 3
        assert img.shape[-1] == 3

        if split == "val":
            # we need to convert the instances to foreground and boundaries
            trg_gt = np.zeros_like(gt.copy())
            trg_gt[gt > 0] = 1
            trg_gt[find_boundaries(gt)] = 2
            assert len(np.unique(trg_gt)) == 3

            imageio.imwrite(Path(os.path.splitext(trg_label_path)[0][:-6]).with_suffix(image_suffix), trg_gt)
        else:
            assert len(np.unique(gt)) == 3
            shutil.copy(src_label_path, trg_label_path)

        shutil.copy(src_image_path, trg_image_path)


def _update_dataset_file(src_dir, trg_dir):
    src_json_file = os.path.join(src_dir, "dataset.json")
    with open(src_json_file) as f:
        data = json.load(f)
        data["numTraining"] = 1101

    trg_json_file = os.path.join(trg_dir, "nnUNet_raw", "Dataset703_NeurIPSCell", "dataset.json")
    with open(trg_json_file, "w") as f:
        json.dump(data, f, indent=4)


def _sanity_check_for_dataset_file(trg_dir):
    trg_json_file = os.path.join(trg_dir, "nnUNet_raw", "Dataset703_NeurIPSCell", "dataset.json")
    with open(trg_json_file) as f:
        data = json.load(f)
        print(data["numTraining"])


def _get_split_file(trg_dir):
    preprocessed_dir = os.path.join(trg_dir, "nnUNet_preprocessed")
    if not os.path.exists(preprocessed_dir):
        raise AssertionError("You need to run preprocessing for nnUNet first.")

    data_dir = os.path.join(trg_dir, "nnUNet_raw", "Dataset703_NeurIPSCell")

    tr_images = sorted(glob(os.path.join(data_dir, "imagesTr", "train_*")))
    val_images = sorted(glob(os.path.join(data_dir, "imagesTr", "val_*")))

    tr_image_ids = [Path(image_id).stem[:-5] for image_id in tr_images]
    val_image_ids = [Path(image_id).stem[:-5] for image_id in val_images]

    # we create custom splits for all folds, to fit with the expectation
    all_split_inputs = [{'train': tr_image_ids, 'val': val_image_ids} for _ in range(5)]

    json_file = os.path.join(preprocessed_dir, "Dataset703_NeurIPSCell", "splits_final.json")
    with open(json_file, "w") as f:
        json.dump(all_split_inputs, f, indent=4)


def main():
    # trg_root = "/scratch/usr/nimanwai/experiments/nnunetv2_neurips_cellseg/"  # for nnUNetv2
    trg_root = "/scratch/usr/nimanwai/experiments/U-Mamba/data/"   # for U-Mamba

    # convert_inputs("train", SRC_ROOT, trg_root)
    # convert_inputs("val", SRC_ROOT, trg_root)

    # _update_dataset_file(SRC_ROOT, trg_root)
    # _sanity_check_for_dataset_file(trg_root)

    _get_split_file(trg_root)


if __name__ == "__main__":
    main()
