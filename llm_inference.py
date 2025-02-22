import json
import os
import argparse
import torch
import glob
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import numpy as np
import sys
from utils.prompt_lib import *
from colorama import init, Fore, Style





MODEL_LIST = ["llama3", "qwen"]


def inference(args):
    """
    Run inference using the specified model on the input JSON files and save the results.

    :param model_name: str, name of the pre-trained model
    :param input_dir: str, directory containing input JSON files
    :param out_dir: str, directory to save the output JSON files
    """
    print(f"Accessing {torch.cuda.device_count()} GPUs!")
    

    llm = LLM(model=args.model_path, max_model_len=1024,
              dtype="float16", tensor_parallel_size=1, 
              trust_remote_code=True, max_logprobs=1000)
    sampling_params = SamplingParams(logprobs=1000, max_tokens=32, temperature=args.temp, n=args.n) #  temperature=0代表greedy解码
    

    output_dir = os.path.join(args.out_dir, os.path.basename(args.model_path))
    os.makedirs(output_dir, exist_ok=True)

    if args.prompt in globals():
        func = globals()[args.prompt]
    else:
        raise NameError(f"Function {args.prompt} not found in utilities module.")

    
    prompts = []
    all_datas = []
    result = []


    with open(args.input_file_name, 'r', encoding='utf-8') as f:
        all_datas = json.load(f)


    for item in all_datas:
        prompts.append(func(item))

    print(f"{Fore.GREEN}Example prompt:{Style.RESET_ALL}\n{Fore.CYAN}{prompts[0]}{Style.RESET_ALL}")
    
    for i in tqdm(range(0, len(all_datas), args.bsz)):
        batch_prompts = prompts[i:i + args.bsz]
        if args.n == 1:
            outputs = llm.generate(batch_prompts, sampling_params)

            predictions = []
            for output in outputs:
                generated_text = output.outputs[0].text
                predictions.append(generated_text)


            for j in range(args.bsz):
                if i + j >= len(all_datas):
                    break
                data = all_datas[i + j]
                data["prediction"] = predictions[j]
                result.append(data)
        else:
            generate_questions = llm.generate(batch_prompts, sampling_params)
            # pdb.set_trace()
            for j in range(len(generate_questions)):
                for k in range(len(generate_questions[j].outputs)):
                    q = generate_questions[j].outputs[k].text
                    q = q.replace("Question: ", "")
                    result.append({
                        "ground_truth": all_datas[i+j]['ground_truth'],
                        "question": q
                    })


    input_file_name = os.path.basename(args.input_file_name)
    input_file_name = input_file_name.split(".")[0]
    output_file = os.path.join(output_dir, args.prompt + input_file_name)


    with open(output_file, "w") as json_file:
        json.dump(result, json_file, indent=4)  # indent=4 用于更美观的格式化




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on input JSON files using a specified model.")
    parser.add_argument('--model_path', type=str, help="Path to the pre-trained model.", default='./result_tab/Meta-Llama-3-8B-Instruct_popqa_12868_unbalanced_ckpt800')
    parser.add_argument('--input_file_name', type=str, help="Directory containing input JSON files.", default="./output/Meta-Llama-3-8B-Instruct/popqa/llama3_qa_popqa_test_binary.json")
    parser.add_argument('--out_dir', type=str, help="Directory to save the output JSON files.", default="./output")
    parser.add_argument('--prompt', type=str, help="Directory to save the output JSON files.", default="llama3_boundary")
    parser.add_argument('--bsz', type=int, help="Directory to save the output JSON files.", default=16)
    parser.add_argument('--temp', type=float, help="Directory to save the output JSON files.", default=0.0)
    parser.add_argument('--n', type=int, help="Directory to save the output JSON files.", default=5)
    args = parser.parse_args()

    inference(args)
