#!/bin/bash
#SBATCH -p l40-gpu      # or l40-gpu depending on availability
#SBATCH --output=output_%j.txt  # Output log file with job ID
#SBATCH --error=error_%j.txt    # Error log file
#SBATCH -N 1                    # Number of nodes
#SBATCH -n 1                    # Number of tasks
#SBATCH --mem=14g                # Memory allocation
#SBATCH -t 00-10:00:00           # Job time (1 day)
#SBATCH --qos=gpu_access        # Quality of Service
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --mail-type=END         # Email notifications
#SBATCH --mail-user=janeqiu@ad.unc.edu

module purge 

# Source bashrc and load CUDA
source /nas/longleaf/home/janeqiu/.bashrc
module load cuda/12.6

# Activate conda environment with Python 3.12.4
conda activate "/nas/longleaf/home/janeqiu/.conda/envs/MRI_proj_v3"

# Check if CUDA is available (for debugging, optional)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Print current directory for verification (optional)
pwd

# Start logging GPU, CPU, and system stats every 60 seconds
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 60 >> gpu_usage.log &
vmstat 60 >> system_usage.log &
top -b -d 60 >> cpu_usage.log &

# Run the main Python script
echo "10 hours, l40 GPU, goal 30 epochs, MEM =14g"
python "/nas/longleaf/home/janeqiu/Desktop/BMME 575/Project/MRI-Denoising-Project/modelJQC_batchNorm_disc.py"