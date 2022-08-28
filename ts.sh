#!/bin/bash
#SBATCH --partition=iris # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --cpus-per-task=4 # Request 8 CPUs for this task
#SBATCH --mem=24G # Request 8GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --exclude=iris4 # Don't run on iris1
#SBATCH --job-name="amzproto-mix-ts-val" # Name the job (for easier monitoring)
#SBATCH --mail-type=END
#SBATCH --mail-user=yatagait@stanford.edu

# source /iris/u/huaxiu/CrossMeta/venv/bin/activate
K_SHOTS=(1 5)
TEMPS=(0.2 0.6 1.0 1.4 1.8)
for (( k=0;k<${#K_SHOTS[@]};k++)); do
	for (( t=0;t<${#TEMPS[@]};t++)); do	
		# Test
		CUDA_LAUNCH_BLOCKING=1 python main.py --datasource=amazonreview --meta_lr=2e-5 --meta_batch_size=1 --update_batch_size=${K_SHOTS[${k}]} --update_batch_size_eval=5 --num_classes=5 --logdir=xxx --use_kg=0 --metatrain_iterations=15000 --ratio=1.0 --train=0 --temp_scaling=${TEMPS[${t}]} --mix
	done
done
