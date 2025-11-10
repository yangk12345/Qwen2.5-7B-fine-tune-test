#run the following server in the commnd line

# Start vLLM server (sampling parameters are set per-request, not at server startup)
# Removed --enable-auto-tool-choice and --tool-call-parser as they cause 500 errors with standard evaluation
# VLLM_USE_MODELSCOPE=True \
#   CUDA_VISIBLE_DEVICES=2,3 \
#   vllm serve  /root/autodl-tmp/swift_v2/ms-swift/megatron_output/Qwen2.5-7B-Instruct/v17-20251107-095752/checkpoint-27\
#   --gpu-memory-utilization 0.7 \
#   --served-model-name qwen2.5 \
#   --trust-remote-code \
#   --port 8801 \
#   --tensor-parallel-size 2 \
#   --max-model-len 8192

# curl -X POST http://127.0.0.1:8801/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "qwen3-30B-MoE_trained",
#     "messages": [{"role": "user", "content": "What is 5+5?"}],
#     "max_tokens": 5000,
#     "temperature": 0.0
#   }'


# eval dataset list
# eval scope llm benmark list
# https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/llm.html
#cmmlu

  # --eval_limit 20 

# swift eval \
#   --model Qwen/Qwen3-30B-A3B-Instruct \
#   --eval_backend HuggingFace \
#   --eval_dataset arc IFEval \
#   --eval_limit 20 \
#   --eval_hf_device_map auto \
# gsm


# For GSM8K with zero-shot (no few-shot examples to avoid long prompts):
swift eval \
  --eval_url "http://127.0.0.1:8801/v1" \
  --model qwen2.5 \
  --eval_backend Native \
  --eval_dataset MMLU \
  --eval_limit 50 \
  --eval_dataset_args '{"gsm8k": {"few_shot_num": 0}}'
