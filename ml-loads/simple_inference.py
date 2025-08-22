import torch
from vllm import LLM, SamplingParams
import time
import cProfile
import pstats

def run_inference(num_gpus=4):
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=512,
        stop=["</s>"]
    )

    llm = LLM(
        model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        tensor_parallel_size=num_gpus,
        trust_remote_code=True
    )

    prompt = "Write a Python function to calculate the fibonacci sequence."
    
    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params)
    end_time = time.time()

    print(f"Generated text: {outputs[0].outputs[0].text}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    
    run_inference()
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()
