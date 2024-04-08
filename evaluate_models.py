import os
from glob import glob
from tqdm import tqdm

import numpy as np
import imageio.v3 as imageio

from torch_em.util.segmentation import watershed_from_components

from elf.evaluation import mean_segmentation_accuracy


def evaluate_predictions(gt_dir, model_choice):
    if model_choice == "nnunetv2":
        root_dir = "/scratch/usr/nimanwai/experiments/nnunetv2_neurips_cellseg/Testing/Public/predictionTs/"
    else:
        root_dir = "/scratch/usr/nimanwai/experiments/U-Mamba/data/Testing/Public/predictionTs/"

    all_predictions = sorted(glob(os.path.join(root_dir, "*.tif")))
    all_gt = sorted(glob(os.path.join(gt_dir, "*.tiff")))

    assert len(all_predictions) == len(all_gt)

    msa_list = []
    for prediction_path, gt_path in tqdm(zip(all_predictions, all_gt), total=len(all_gt)):
        gt = imageio.imread(gt_path)
        prediction = imageio.imread(prediction_path)

        fg = (prediction == 1).astype("int")
        bd = (prediction == 2).astype("int")

        instances = watershed_from_components(bd, fg)

        msa = mean_segmentation_accuracy(instances, gt)
        msa_list.append(msa)

    msa_score = np.mean(msa_list)
    print(msa_score)


def main():
    gt_dir = "/scratch/projects/nim00007/sam/data/neurips-cell-seg/zenodo/Testing/Public/labels/"

    # model_choice = "nnunetv2"
    model_choice = "umamba"

    evaluate_predictions(gt_dir, model_choice)


if __name__ == "__main__":
    main()
