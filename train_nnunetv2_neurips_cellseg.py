import os
import warnings
import argparse

import torch


NNUNET_PATH = "/scratch/usr/nimanwai/experiments/nnunetv2_neurips_cellseg"


def declare_paths(nnunet_path: str):
    """To let the system known of the path variables where the respective folders exist (important for all components)
    """
    warnings.warn(
        "Make sure you have created the directories mentioned in this functions (relative to the root directory)"
    )

    os.environ["nnUNet_raw"] = os.path.join(nnunet_path, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(nnunet_path, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(nnunet_path, "nnUNet_results")


def preprocess_neurips_cellseg():
    # let's check the preprocessing first
    #     - data comes from here - https://drive.google.com/drive/folders/18QSSiABS8H3qtx8SZA6RQb3aH1nbc3iF
    cmd = "nnUNetv2_plan_and_preprocess -d 703 --verify_dataset_integrity"
    os.system(cmd)


def train_nnunetv2(args):
    _have_splits = os.path.exists(
        os.path.join(NNUNET_PATH, "nnUNet_preprocessed", "Dataset703_NeurIPSCell", "splits_final.json")
    )
    assert _have_splits, "You need to create the splits yourself. See `convert_neurips_cellseg` for details."

    # train 2d nnUNet
    gpus = torch.cuda.device_count()
    cmd = f"nnUNet_compile=T nnUNet_n_proc_DA=8 nnUNetv2_train 703 2d {args.fold} -num_gpus {gpus} --c"
    os.system(cmd)


def predict_nnunetv2(args):
    input_dir = os.path.join(NNUNET_PATH, "Testing/Public/imagesTs/")
    assert os.path.exists(input_dir)

    output_dir = os.path.join(NNUNET_PATH, "Testing/Public/predictionTs/")

    cmd = f"nnUNetv2_predict -i {input_dir} -o {output_dir} -d 703 -c 2d -f {args.fold}"
    os.system(cmd)


def main(args):
    declare_paths(NNUNET_PATH)

    if args.preprocess:
        preprocess_neurips_cellseg()

    if args.train:
        train_nnunetv2(args)

    if args.predict:
        predict_nnunetv2(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--fold", type=str, default="0")
    args = parser.parse_args()
    main(args)
