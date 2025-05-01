#!/bin/bash
#
#SBATCH -J therapist-gen                 # Job name
#SBATCH -o logs/%x_%j.out               # STDOUT log
#SBATCH -e logs/%x_%j.err               # STDERR log
#SBATCH -N 1                            # One node
#SBATCH --ntasks=1                      # One task
#SBATCH --cpus-per-task=6               # 6 CPUs
#SBATCH --gres=gpu:8                    # 8 GPUs
#SBATCH --mem=16G                       # Total memory
#SBATCH -t 24:00:00                     # 8-hour wall time
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=am8792L@princeton.edu

# Load Python (adjust if needed)
module load python/3.10.4

# Activate your virtualenv
source ~/venvs/gen_conv/bin/activate

# Go to your script directory
cd /u/am8792/COS598A_Project

# Start Ollama server (in background)
ollama serve &

# Optional: wait a few seconds to ensure server is up
sleep 5

# Run the script
python generate_conversations.py \
  --model llama3 \
  --turns 20 \
  --output_dir /scratch/$USER/conversations \
  --num_conversations 10 \
  --workers 8