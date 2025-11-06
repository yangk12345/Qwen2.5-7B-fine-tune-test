# 4 GPUs - Megatron training with mcore bridge for Qwen2.5-7B (Dense Model)
# Configuration: TP=2, PP=2, DP=1 (optimal for 7B model)
export WANDB_MODE="online"
export WANDB_LOG_INTERVAL="10"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NODE_RANK=0 \
NPROC_PER_NODE=4 \
megatron sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --load_safetensors true \
    --save_safetensors true \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#5000' \
              'AI-ModelScope/alpaca-gpt4-data-en#5000' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 1 \
    --sequence_parallel true \
    --packing true \
    --micro_batch_size 4 \
    --global_batch_size 8 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen2.5-7B-Instruct \
    --eval_interval 500 \
    --save_interval 500 \
    --max_length 8192 \
    --num_workers 4 \
    --dataset_num_proc 4 \
    --no_save_optim true \
    --no_save_rng true \
    --no_initialization false \
    --attention_backend flash \
    --wandb_project "Qwen2.5-7B-Instruct-mbridge" \
    --wandb_exp_name "Qwen2.5-7B-Instruct-mbridge" \
    --wandb_save_dir ./wandb

# This configuration will give you:
# - Data Parallel Size = 4 / (2 × 2) = 1
# - Each of 4 GPUs handles part of the model
# - Gradient accumulation = 8 / (1 × 1) = 8 steps
# - Effective batch size = 1 × 1 × 8 = 8
