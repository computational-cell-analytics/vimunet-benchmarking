import os
import shutil
from tqdm import tqdm
from glob import glob
from pathlib import Path

import json
import numpy as np
import imageio.v3 as imageio

from torch_em.data.datasets.livecell import _download_livecell_annotations
from torch_em.transform import BoundaryTransform


def _get_paths(path, split):
    all_image_paths, all_gt_paths = _download_livecell_annotations(
        path=path, split=split, download=False, cell_types=None, label_path=None,
    )
    return all_image_paths, all_gt_paths


def _write_dataset_json_file(trg_dir, dataset_name):
    gt_dir = os.path.join(trg_dir, "nnUNet_raw", dataset_name, "labelsTr")

    train_ids = [Path(_path).stem for _path in glob(os.path.join(gt_dir, "*_train.tif"))]
    val_ids = [Path(_path).stem for _path in glob(os.path.join(gt_dir, "*_val.tif"))]

    json_file = os.path.join(trg_dir, "nnUNet_raw", dataset_name, "dataset.json")

    data = {
        "channel_names": {
            "0": "phase_contrast"
        },
        "labels": {
            "background": 0,
            "foreground": 1,
            "boundaries": 2
        },
        "numTraining": len(val_ids) + len(train_ids),
        "file_ending": ".tif",
        "name": dataset_name,
        "description": "LIVECell: https://doi.org/10.1038/s41592-021-01249-6"
    }

    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

    return train_ids, val_ids


def _update_label_image(input_path, trg_path, ):
    gt = imageio.imread(input_path)

    get_boundaries = BoundaryTransform(add_binary_target=True)
    outputs = get_boundaries(gt)
    fg, bd = outputs

    # let's map this to one gt image
    semantic_labels = np.zeros_like(gt)
    semantic_labels[fg] = 1
    semantic_labels[bd] = 2

    # now, let's save the image to the target path
    imageio.imwrite(trg_path, semantic_labels)


def convert_livecell_for_training(path, trg_dir, dataset_name):
    train_image_paths, train_gt_paths = _get_paths(path, "train")
    val_image_paths, val_gt_paths = _get_paths(path, "val")

    # the idea is we move all the images to one directory, write their image ids into a split.json file,
    # which nnunet will read to define the custom validation split
    image_dir = os.path.join(trg_dir, "nnUNet_raw", dataset_name, "imagesTr")
    gt_dir = os.path.join(trg_dir, "nnUNet_raw", dataset_name, "labelsTr")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    train_ids, val_ids = [], []

    # let's start with moving the validation
    assert len(val_gt_paths) == len(val_image_paths)
    for image_path, gt_path in tqdm(
        zip(sorted(val_image_paths), sorted(val_gt_paths)), total=len(val_image_paths)
    ):
        image_id = Path(image_path).stem

        trg_image_path = os.path.join(image_dir, f"{image_id}_val_0000.tif")
        shutil.copy(src=image_path, dst=trg_image_path)

        trg_gt_path = os.path.join(gt_dir, f"{image_id}_val.tif")
        _update_label_image(gt_path, trg_gt_path)

        val_ids.append(Path(trg_gt_path).stem)

    # next, let's move the train set
    assert len(train_gt_paths) == len(train_image_paths)
    for image_path, gt_path in tqdm(
        zip(sorted(train_image_paths), sorted(train_gt_paths)), total=len(train_image_paths)
    ):
        image_id = Path(image_path).stem

        trg_image_path = os.path.join(image_dir, f"{image_id}_train_0000.tif")
        shutil.copy(src=image_path, dst=trg_image_path)

        trg_gt_path = os.path.join(gt_dir, f"{image_id}_train.tif")
        _update_label_image(gt_path, trg_gt_path)

        train_ids.append(Path(trg_gt_path).stem)


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


def convert_livecell_for_testing(path, trg_dir, dataset_name):
    test_image_paths, test_gt_paths = _get_paths(path, "test")

    # the idea for here is to move the data to a central location,
    # where we can automate the inference procedure
    image_dir = os.path.join(trg_dir, "test", dataset_name, "imagesTs")
    gt_dir = os.path.join(trg_dir, "test", dataset_name, "labelsTs")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    assert len(test_image_paths) == len(test_gt_paths)
    for image_path, gt_path in tqdm(
        zip(sorted(test_image_paths), sorted(test_gt_paths)), total=len(test_image_paths)
    ):
        image_id = Path(image_path).stem

        trg_image_path = os.path.join(image_dir, f"{image_id}_0000.tif")
        shutil.copy(src=image_path, dst=trg_image_path)

        trg_gt_path = os.path.join(gt_dir, f"{image_id}.tif")
        shutil.copy(src=gt_path, dst=trg_gt_path)


def main():
    path = "/scratch/projects/nim00007/sam/data/livecell"
    dataset_name = "Dataset205_LIVECell"

    # space to store your top-level nnUNet files
    trg_root = "/scratch/usr/nimanwai/experiments/nnunetv2_neurips_cellseg/"  # for nnUNetv2
    # trg_root = "/scratch/usr/nimanwai/experiments/U-Mamba/data"  # for U-Mamba

    # convert_livecell_for_training(path, trg_root, dataset_name)
    # create_json_files(trg_root, dataset_name)

    convert_livecell_for_testing(path, trg_root, dataset_name)


if __name__ == "__main__":
    main()
