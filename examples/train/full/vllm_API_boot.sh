CUDA_VISIBLE_DEVICES=2,3 \
evalscope eval \
 --model megatron_output/Qwen2.5-7B-Instruct \
 --api-url http://127.0.0.1:8801/v1 \
 --eval-type openai_api \
 --datasets gsm8k arc \
 --limit 5





 