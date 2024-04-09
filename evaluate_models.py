import os
from glob import glob
from tqdm import tqdm

import numpy as np
import imageio.v3 as imageio

from torch_em.util.segmentation import watershed_from_components

from elf.evaluation import mean_segmentation_accuracy

from train_nnunetv2 import DATASET_MAPPING


NNUNET_ROOT = "/scratch/usr/nimanwai/experiments/nnunetv2_neurips_cellseg"
UMAMBA_ROOT = "/scratch/usr/nimanwai/experiments/U-Mamba/data"


def evaluate_predictions(root_dir, dataset):
    all_predictions = sorted(glob(os.path.join(root_dir, "predictionTs", "*.tif")))
    all_gt = sorted(glob(os.path.join(root_dir, "labelsTs", "*.tif")))

    assert len(all_predictions) == len(all_gt)

    msa_list, sa50_list, sa75_list = [], [], []
    for prediction_path, gt_path in tqdm(zip(all_predictions, all_gt), total=len(all_gt)):
        gt = imageio.imread(gt_path)
        prediction = imageio.imread(prediction_path)

        if dataset == "cremi":
            bd = prediction
            instances = watershed_from_components(bd, np.ones_like(bd))
        else:  # for livecell and neurips-cellseg, we do have foreground
            fg = (prediction == 1).astype("int")
            bd = (prediction == 2).astype("int")
            instances = watershed_from_components(bd, fg)

        msa, sa = mean_segmentation_accuracy(instances, gt, return_accuracies=True)
        msa_list.append(msa)
        sa50_list.append(sa[0])
        sa75_list.append(sa[5])

    msa_score = np.mean(msa_list)
    sa50_score = np.mean(sa50_list)
    sa75_score = np.mean(sa75_list)
    print(msa_score, sa50_score, sa75_score)


def main(args):
    _, dataset_name = DATASET_MAPPING[args.dataset]

    root_dir = os.path.join(
        NNUNET_ROOT if args.model == "nnunetv2" else UMAMBA_ROOT,
        "test", dataset_name
    )

    evaluate_predictions(root_dir, args.dataset)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    args = parser.parse_args()
    main(args)
