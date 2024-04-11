import os

import argparse

from train_nnunetv2 import DATASET_MAPPING


def preprocess_data(dataset_id):
    # let's check the preprocessing first
    #     - The NeurIPS CellSeg data comes from here
    #     - https://drive.google.com/drive/folders/18QSSiABS8H3qtx8SZA6RQb3aH1nbc3iF
    cmd = f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity"
    os.system(cmd)


def train_umamba(fold, dataset_id, for_all_encoder):
    if for_all_encoder:
        # train 2d "U-Mamba-Enc" model
        #     - details: uses the mamba layers in the entire decoder

        # NOTE: the Nan issues I faced before should be fixed now
        cmd = f"nnUNetv2_train {dataset_id} 2d {fold} -tr nnUNetTrainerUMambaEncNoAMP --c"
    else:
        # train 2d "U-Mamba-Bot" model
        #     - details: uses the mamba layer only in the bottleneck - b/w the encoder and decoder junction
        cmd = f"nnUNetv2_train {dataset_id} 2d {fold} -tr nnUNetTrainerUMambaBot --c"

    os.system(cmd)


def predict_umamba(fold, for_all_encoder, dataset_name, dataset_id):
    root_dir = "/scratch/usr/nimanwai/experiments/U-Mamba/data"

    input_dir = os.path.join(root_dir, "test", dataset_name, "imagesTs")
    assert os.path.exists(input_dir)

    if for_all_encoder:
        trainer = "nnUNetTrainerUMambaEncNoAMP"
    else:
        trainer = "nnUNetTrainerUMambaBot"

    output_dir = os.path.join(root_dir, "test", dataset_name, trainer, "predictionTs")

    cmd = f"nnUNetv2_predict -i {input_dir} -o {output_dir} -d {dataset_id} -c 2d -tr {trainer} -f {fold}"
    os.system(cmd)


def main(args):
    dataset_id, dataset_name = DATASET_MAPPING[args.dataset]

    if args.preprocess:
        preprocess_data(dataset_id)

    if args.train:
        train_umamba(fold=args.fold, dataset_id=dataset_id, for_all_encoder=args.for_all_encoder)

    if args.predict:
        predict_umamba(
            fold=args.fold,
            for_all_encoder=args.for_all_encoder,
            dataset_name=dataset_name,
            dataset_id=dataset_id
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--for_all_encoder", action="store_true")
    parser.add_argument("--fold", type=str, default="0")
    args = parser.parse_args()
    main(args)
