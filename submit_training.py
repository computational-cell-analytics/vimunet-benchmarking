import os
import shutil
import subprocess
from glob import glob
from datetime import datetime


def write_batch_script(out_path, for_all_encoder, total_training_folds=5):
    """Writing scripts with different fold-trainings for nnUNetv2
    """
    for i in range(total_training_folds):
        batch_script = f"""#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -A nim00007
#SBATCH --job-name=umamba-neurips-cellseg

source activate um2
python train_umamba_neurips_cellseg.py --train --fold {i} """

        _op = out_path[:-3] + f"_{i}.sh"

        if for_all_encoder:
            batch_script += "--for_all_encoder"

        with open(_op, "w") as f:
            f.write(batch_script)


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = "U-Mamba"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def submit_slurm():
    """Submit python script that needs gpus with given inputs on a slurm node.
    """
    tmp_folder = "./gpu_jobs"

    # run umamba training for all encoder and just bottleneck
    write_batch_script(get_batch_script_names(tmp_folder), for_all_encoder=False)
    write_batch_script(get_batch_script_names(tmp_folder), for_all_encoder=True)

    for my_script in glob(tmp_folder + "/*"):
        cmd = ["sbatch", my_script]
        subprocess.run(cmd)


if __name__ == "__main__":
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    submit_slurm()
