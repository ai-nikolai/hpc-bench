NUM_GPUS=1

python ./ml_loads/simple_inference_vllm.py --num_gpus ${NUM_GPUS} --model_name "Qwen/Qwen3-8B" #"Qwen/Qwen2.5-7B-Instruct" #"Qwen/Qwen3-8B" #"Qwen/Qwen3-Coder-30B-A3B-Instruct"


# python ./ml_loads/simple_inference_vllm.py --num_gpus 1 --model_name "Qwen/Qwen3-8B" #"Qwen/Qwen2.5-7B-Instruct" #"Qwen/Qwen3-8B" #"Qwen/Qwen3-Coder-30B-A3B-Instruct"
