#!/bin/bash
#SBATCH --job-name=cspca_train_47      # Specify job name
#SBATCH --partition=gpu               # Specify partition name
#SBATCH --mem=128G   
#SBATCH --gres=gpu:1             
#SBATCH --time=48:00:00               # Set a limit on the total run time
#SBATCH --output=/sc-scratch/sc-scratch-cc06-ag-ki-radiologie/prostate_foundation/WSAttention-Prostate/logs/%x/log.o%j      # File name for standard output
#SBATCH --error=/sc-scratch/sc-scratch-cc06-ag-ki-radiologie/prostate_foundation/WSAttention-Prostate/logs/%x/log.e%j        # File name for standard error output
#SBATCH --mail-user=anirudh.balaraman@charite.de
#SBATCH --mail-type=END,FAIL


source /etc/profile.d/conda.sh
conda activate foundation

RUNDIR="/sc-scratch/sc-scratch-cc06-ag-ki-radiologie/prostate_foundation/WSAttention-Prostate"


srun python -u $RUNDIR/run_cspca.py --mode train --config $RUNDIR/config/config_cspca_train.yaml