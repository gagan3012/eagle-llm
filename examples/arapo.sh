#!/bin/bash
#SBATCH --time=4:59:59
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --account=def-hasanc
#SBATCH --job-name=arapo-llama3-oasis
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=ALL

module load python/3.11 scipy-stack gcc arrow cuda cudnn opencv StdEnv/2023 && source ~/ENV/bin/activate

model_name=$1

python eagle/arapo.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --per_device_train_batch_size 2 \
    --num_train_epochs 1 \
    --learning_rate 8e-5 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --eval_steps 500 \
    --output_dir="LLama3-Oasis" \
    --optim paged_adamw_8bit \
    --warmup_steps 150 \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --dataset_name=gagan3012/oasis \
    --lora_alpha=16 \ 