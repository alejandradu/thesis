#!/bin/bash

# run this in the command line
module load anaconda3/2024.6
conda init
conda activate neuralnets
python -c "import torch; print(torch.cuda.is_available())" > is_cuda.txt
python -c "import torch; print(torch.version.cuda)" > cuda_version.txt