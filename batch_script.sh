#!/usr/bin/zsh 

### Job Parameters
#SBATCH --nodes=1
#SBATCH --ntasks=1              
#SBATCH --time=00:03:00         
#SBATCH --job-name=test_arlbench  
#SBATCH --output=stdout.txt     
#SBATCH -A thes1998
#SBATCH --array=0-26

### Program Code

module load GCCcore/12.2.0
module load Python/3.10.8

python arlbench/core/experiments/configs_grid.py

srun python run_arlbench.py environment=config_$SLURM_ARRAY_JOB_ID.yaml

