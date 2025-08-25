import torch
from vllm import LLM, SamplingParams
import time
import cProfile
import pstats

import argparse

def run_inference(num_gpus=1):
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=512,
        stop=["</s>"]
    )
    print("\n\n\n===============\nSetting up LLM Engine:\n\n\n")

    llm = LLM(
        # model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        model="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=num_gpus,
        trust_remote_code=True
    )
    print("\n\n\n----\nSetting up LLM FINISHED\n\n\n")

    prompt = "Write a Python function to calculate the fibonacci sequence."
    
    print(f"\n\n\n===============\nRunning Prompt: ```prompt\n{prompt}\n```\n\n\n")

    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params)
    end_time = time.time()
    print("\n\n\n----\nGeneration finished.\n\n\n")

    print(f"Generated text: {outputs[0].outputs[0].text}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print("Running script with args...")
    args = build_args()
    print(args)

    # profiler = cProfile.Profile()
    # profiler.enable()
    print("--")
    run_inference(num_gpus=args.num_gpus)
    
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumulative')
    # stats.print_stats()
