#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=an-tr043
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=48GB
#SBATCH --output=Nordeje_Neil_ISP_Output-%j.txt

# add modules
module load python/3.11.5

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index dask dask[dataframe] numpy pandas scipy matplotlib

python hpc_meter_proc.py
