# import torch
import sglang as sgl
import time
import cProfile
import pstats

import argparse



def get_sampling_params(**kwargs): #replacement for VLLM's sampling params...
    return kwargs

def run_inference(num_gpus=1, model_name="Qwen/Qwen2.5-7B-Instruct"):
    sampling_params = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
        "max_new_tokens": 512,
    }
    
    print("\n\n\n===============\nSetting up LLM Engine:\n\n\n")
    llm = sgl.Engine(
        model_path=model_name,
        # context_length=1048576,
        # page_size=256,
        # attention_backend="dual_chunk_flash_attn",
        tp_size=num_gpus, #NUM OF GPUS is here...
        # disable_radix_cache=True,
        # enable_mixed_chunk=False,
        # enable_torch_compile=False,
        # chunked_prefill_size=131072,
        mem_fraction_static=0.9,
        # log_level="DEBUG"
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
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print("Running script with args...")
    args = build_args()
    print(args)

    # profiler = cProfile.Profile()
    # profiler.enable()
    print("--")
    run_inference(num_gpus=args.num_gpus, model_name=args.model_name)
    
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumulative')
    # stats.print_stats()
