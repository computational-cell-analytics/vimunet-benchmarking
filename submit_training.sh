#!/bin/bash
#SBATCH --job-name=umamba-neurips-cellseg
#SBATCH -t 4-00:00:00
#SBATCH --mem 64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH -c 16
#SBATCH --qos=7d
#SBATCH --constraint=80gb

source activate um2
python train_neurips_cellseg.py --train --for_all_encoder
