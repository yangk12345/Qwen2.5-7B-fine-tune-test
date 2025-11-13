# pip install math_verify # reward function
# pip install -U trl
# GPU memory: 80GiB
# You can set `--reward_model` to use a reward model to provide rewards.
# PTEngine(pytorch) to rollout

import os
import sys
from swift.cli.main import cli_main

if __name__ == "__main__":
    # ====== Environment Variables (previously export) ,w======
    # the wandb 
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_LOG_INTERVAL"] = "10"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["NODE_RANK"] = "0"
    os.environ["NPROC_PER_NODE"] = "4"
    os.environ["WANDB_PROJECT"] = "Qwen2.5-7B-Instruct-grpo"
    os.environ["WANDB_EXP_NAME"] = "Qwen2.5-7B-Instruct-grpo"
    os.environ["WANDB_SAVE_DIR"] = "./wandb"

    sys.argv = [
        "swift", "rlhf",
        "--rlhf_type", "grpo",
        "--model", "/root/autodl-tmp/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct",
        "--reward_funcs", "accuracy", "format",
        "--train_type", "lora",
        "--lora_rank", "8",
        "--lora_alpha", "32",
        "--target_modules", "all-linear",
        "--torch_dtype", "bfloat16",
        "--dataset", "AI-MO/NuminaMath-TIR#1000",
        "--load_from_cache_file", "true",
        "--max_completion_length", "1024",
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "8",
        "--per_device_eval_batch_size", "8",
        "--learning_rate", "1e-5",
        "--gradient_accumulation_steps", "1",
        "--eval_steps", "100",
        "--save_steps", "100",
        "--save_total_limit", "2",
        "--logging_steps", "5",
        "--max_length", "2048",
        "--output_dir", "output",
        "--warmup_ratio", "0.05",
        "--dataloader_num_workers", "4",
        "--dataset_num_proc", "4",
        "--num_generations", "4",
        "--temperature", "0.9",
        "--system", "examples/train/grpo/prompt.txt",
        "--log_completions", "true",
        "--report_to", "wandb"
    ]
    cli_main()

# ===============The following is the original shell script for grpo training=============== #
# export WANDB_MODE="online"
# export WANDB_LOG_INTERVAL="5"

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   swift rlhf \
#   --rlhf_type grpo \
#   --model /root/autodl-tmp/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct \
#   --reward_funcs accuracy format \
#   --train_type lora \
#   --lora_rank 8 \
#   --lora_alpha 32 \
#   --target_modules all-linear \
#   --torch_dtype bfloat16 \
#   --dataset 'AI-MO/NuminaMath-TIR#1000' \
#   --load_from_cache_file true \
#   --max_completion_length 1024 \
#   --num_train_epochs 1 \
#   --per_device_train_batch_size 4 \
#   --per_device_eval_batch_size 4 \
#   --learning_rate 1e-5 \
#   --gradient_accumulation_steps 1 \
#   --eval_steps 100 \
#   --save_steps 100 \
#   --save_total_limit 2 \
#   --logging_steps 5 \
#   --max_length 2048 \
#   --output_dir output \
#   --warmup_ratio 0.05 \
#   --dataloader_num_workers 4 \
#   --dataset_num_proc 4 \
#   --num_generations 4 \
#   --temperature 0.9 \
#   --system 'examples/train/grpo/prompt.txt' \
#   --log_completions true \
#   --wandb_project "Qwen2.5-7B-Instruct-grpo" \
#   --wandb_exp_name "Qwen2.5-7B-Instruct-grpo" \
#   --wandb_save_dir ./wandb
