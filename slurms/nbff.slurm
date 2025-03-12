#!/bin/bash
#SBATCH --job-name=lrRNN_2bff               # Job name
#SBATCH --output=output.txt           # Standard output file
#SBATCH --error=error.txt             # Standard error file
# #SBATCH --partition=MIG               # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --mem-per-cpu=8G              # memory per cpu-core (4G is default)
#SBATCH --gpus-per-node=1             # Number of gpus per node
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --time=0:59:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=all               # Send email at job completion
#SBATCH --mail-user=ad2002@princeton.edu    # Email address for notifications

#Load necessary modules (if needed)

module load anaconda3/2024.6
conda init
conda activate neuralnets

# Set PYTHONPATH to include the parent directory of 'synthetic_datasets' and 'main'
export PYTHONPATH="/home/ad2002/thesis/"

python main/train_cluster_nbff.py

#Optionally, you can include cleanup commands here (e.g., after the job finishes)
#For example:
#rm some_temp_file.txt