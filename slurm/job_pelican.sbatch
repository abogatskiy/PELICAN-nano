#!/bin/bash

#SBATCH --job-name=n7
#SBATCH --output=./out/array_%A_%a.out
#SBATCH --error=./err/array_%A_%a.err
#SBATCH --array=0-24
#SBATCH --time=168:00:00
#SBATCH --partition=gpu
#SBATCH -C a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=32G

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# python3 ~/ceph/NBodyJetNets/NetworkDesign/scripts/train_lgn.py --datadir=./data/sample_data/v0 --batch-size=50 --ir-safe=True

nvidia-smi

CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate py310
A=(n7-{a..z})

# nano
sleep $SLURM_ARRAY_TASK_ID #sleep for job-index seconds to make sure the seeds are different
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_nano.py --datadir=../../PELICAN-private/data/v0 --cuda --nobj=80 --nobj-avg=49 --num-epoch=140 --num-train=-1 --num-valid=60000 --batch-size=512 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --optim=adamw --lr-decay-type=warm --no-factorize --masked --scale=1 --lr-init=0.01 --lr-final=1e-6 --drop-rate=0.01 --drop-rate-out=0.01 --weight-decay=0.0 --reproducible --no-fix-data --no-summarize --config=s --config-out=s --add-beams --activation=relu --batchnorm=b --n-hidden=2 --task=train #--seed="${S[$SLURM_ARRAY_TASK_ID]}" 
