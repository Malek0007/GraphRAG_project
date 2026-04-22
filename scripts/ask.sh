#!/bin/bash

#SBATCH --job-name=graphrag-ask
#SBATCH --output=ask-%j.out
#SBATCH --partition=class
#SBATCH --ntasks=1
#SBATCH --mem=8192
#SBATCH --time=02:00:00

echo "Starting job on $(hostname)"
echo "Time: $(date)"

# 👉 Go to your project root
cd /mnt/home/YOUR_USERNAME/GRAPHRAG_PROJECT

# 👉 Activate your venv
source venv/bin/activate

# 👉 Run your script
python src/Graph_index/ask.py

# 👉 Deactivate
deactivate

echo "Finished at $(date)"