#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --cpus-per-task=4 # Request 8 CPUs for this task
#SBATCH --mem=24G # Request 8GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --exclude=iris4,iris5,iris6
#SBATCH --job-name="amzproto-default-test" # Name the job (for easier monitoring)
#SBATCH --mail-type=END
#SBATCH --mail-user=yatagait@stanford.edu

# source /iris/u/huaxiu/CrossMeta/venv/bin/activate
kshots=1
# Train
# CUDA_LAUNCH_BLOCKING=1 python main.py --datasource=amazonreview --meta_lr=2e-5 --meta_batch_size=1 --update_batch_size=$kshots --update_batch_size_eval=5 --num_classes=5 --logdir=xxx --use_kg=0 --metatrain_iterations=1000 --ratio=1.0 --train=1
# Test
# CUDA_LAUNCH_BLOCKING=1 python main.py --datasource=amazonreview --meta_lr=2e-5 --meta_batch_size=1 --update_batch_size=$kshots --update_batch_size_eval=5 --num_classes=5 --logdir=xxx --use_kg=0 --metatrain_iterations=1000 --ratio=1.0 --train=0 --mix

kshots=5
bert_model=albert-large-v2
# Train
# CUDA_LAUNCH_BLOCKING=1 python main.py --datasource=amazonreview --meta_lr=2e-5 --meta_batch_size=1 --update_batch_size=$kshots --update_batch_size_eval=5 --num_classes=5 --logdir=xxx --use_kg=0 --metatrain_iterations=1000 --ratio=1.0 --train=1
# Test
CUDA_LAUNCH_BLOCKING=1 python main.py --test_epoch=600 --bert_model=$bert_model --all_data --datasource=amazonreview --meta_lr=2e-5 --meta_batch_size=1 --update_batch_size=$kshots --update_batch_size_eval=5 --num_classes=5 --logdir=xxx --use_kg=0 --metatrain_iterations=1000 --ratio=1.0 --train=0
