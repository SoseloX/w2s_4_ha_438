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
import pdb
from colorama import init, Fore, Style
init()  # 初始化 colorama



MODEL_LIST = ["llama3", "qwen"]


def calculate_ppl(logprobs_values, text_length):
    """
    计算PPL (Perplexity)
    PPL = exp(-1/N * sum(log P(x_i)))
    """

    avg_log_prob = sum(logprobs_values) / text_length
    return np.exp(-avg_log_prob)


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
              trust_remote_code=True, max_logprobs=100)
    sampling_params = SamplingParams(logprobs=100, max_tokens=32, temperature=args.temp, n=args.n) #  temperature=0代表greedy解码
    

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

    print(f"{Fore.RED}Example prompt:{Style.RESET_ALL}\n{Fore.CYAN}{prompts[0]}{Style.RESET_ALL}")
    for i in tqdm(range(0, len(all_datas), args.bsz)):
        batch_prompts = prompts[i:i + args.bsz]
        outputs = llm.generate(batch_prompts, sampling_params)

        batch_predictions = []
        batch_probabilities = []
        batch_ppls = []
        for output in outputs:
            all_texts = []
            all_probs = []
            all_ppls = []
            for out in output.outputs:
                text = out.text
                out_ids = out.token_ids
                logprobs_dict = out.logprobs

                if logprobs_dict:
                    # 获取每个token的概率
                    token_logprobs = []
                    for token_idx in range(len(out_ids)):
                        token_logprobs.append(logprobs_dict[token_idx][out_ids[token_idx]].logprob)
                    # 使用token级别的概率计算
                    logprobs_values = token_logprobs
                    sequence_prob = np.exp(sum(logprobs_values))
                    # 计算PPL
                    text_length = len(logprobs_values)
                    ppl = calculate_ppl(logprobs_values, text_length)
                else:
                    sequence_prob = 0.0
                    ppl = float('inf')
                
                all_texts.append(text)
                all_probs.append(sequence_prob)
                all_ppls.append(ppl)
            
            batch_predictions.append(all_texts)
            batch_probabilities.append(all_probs)
            batch_ppls.append(all_ppls)

        for j in range(args.bsz):
            if i + j >= len(all_datas):
                break
            data = all_datas[i + j]
            data["predictions"] = batch_predictions[j]
            data["probabilities"] = batch_probabilities[j]
            data["perplexities"] = batch_ppls[j]
            result.append(data)


    input_file_name = os.path.basename(args.input_file_name)
    output_file = os.path.join(output_dir, args.prompt + "_temp" + str(args.temp) + "_n" + str(args.n) + "_" + input_file_name)


    with open(output_file, "w") as json_file:
        json.dump(result, json_file, indent=4)  # indent=4 用于更美观的格式化




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on input JSON files using a specified model.")
    parser.add_argument('--model_path', type=str, help="Path to the pre-trained model.", default='Meta-Llama-3-8B-Instruct')
    parser.add_argument('--input_file_name', type=str, help="Directory containing input JSON files.", default="./data/popqa_test_binary.json")
    parser.add_argument('--out_dir', type=str, help="Directory to save the output JSON files.", default="./output")
    parser.add_argument('--prompt', type=str, help="Directory to save the output JSON files.", default="llama3_prudent_icl")
    parser.add_argument('--bsz', type=int, help="Directory to save the output JSON files.", default=16)
    parser.add_argument('--temp', type=float, help="Directory to save the output JSON files.", default=1.0)
    parser.add_argument('--n', type=int, help="Directory to save the output JSON files.", default=10)
    args = parser.parse_args()

    inference(args)
