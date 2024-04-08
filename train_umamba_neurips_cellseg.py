import os

import argparse


def preprocess_neurips_cellseg():
    # let's check the preprocessing first
    #     - data comes from here - https://drive.google.com/drive/folders/18QSSiABS8H3qtx8SZA6RQb3aH1nbc3iF
    cmd = "nnUNetv2_plan_and_preprocess -d 703 --verify_dataset_integrity"
    os.system(cmd)


def train_umamba(args):
    if args.for_all_encoder:
        # train 2d "U-Mamba-Enc" model
        #     - details: uses the mamba layers in the entire decoder
        cmd = f"nnUNetv2_train 703 2d {args.fold} -tr nnUNetTrainerUMambaEncNoAMP --c"  # NOTE: fixes the Nan issue
    else:
        # train 2d "U-Mamba-Bot" model
        #     - details: uses the mamba layer only in the bottleneck - b/w the encoder and decoder junction
        cmd = f"nnUNetv2_train 703 2d {args.fold} -tr nnUNetTrainerUMambaBot --c"

    os.system(cmd)


def predict_umamba(args):
    root_dir = "/scratch/usr/nimanwai/experiments/U-Mamba/data"
    input_dir = os.path.join(root_dir, "Testing/Public/imagesTs/")
    assert os.path.exists(input_dir)

    output_dir = os.path.join(root_dir, "Testing/Public/predictionTs/")

    if args.for_all_encoder:
        trainer = "nnUNetTrainerUMambaEncNoAMP"
    else:
        trainer = "nnUNetTrainerUMambaBot"

    cmd = f"nnUNetv2_predict -i {input_dir} -o {output_dir} -d 703 -c 2d -tr {trainer} -f {args.fold}"
    os.system(cmd)


def main(args):
    if args.preprocess:
        preprocess_neurips_cellseg()

    if args.train:
        train_umamba(args)

    if args.predict:
        predict_umamba(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--for_all_encoder", action="store_true")
    parser.add_argument("--fold", type=str, default="0")
    args = parser.parse_args()
    main(args)
