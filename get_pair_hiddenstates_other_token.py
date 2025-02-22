import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn import functional as F
from tqdm import tqdm
import pickle
import sys
from utils.prompt_lib import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
import argparse
import torch
import pandas as pd
import random
from collections import defaultdict
import numpy as np



def batch_encode_prompt(prompt_list, tokenizer, batch_size):
    encoded_prompts = []
    last_special_token_positions = []  # 新增：存储128007出现的最后位置
    batch_start_token = []
    # 计算需要多少个批次
    num_batches = len(prompt_list) // batch_size + (1 if len(prompt_list) % batch_size else 0)
    
    # 分批次对prompt进行编码
    for i in range(num_batches):
        batch = prompt_list[i * batch_size: (i + 1) * batch_size]
        # 使用tokenizer对文本进行编码
        encoded_batch = tokenizer(batch, 
                                  padding=True, 
                                  truncation=True, 
                                  max_length=256,
                                  return_tensors='pt') 
        temp_point = []
        # 新增：找到每个序列中128007的最后位置
        for seq in encoded_batch['input_ids']:
            positions = (seq == 128007).nonzero(as_tuple=True)[0]
            last_pos = positions[-1].item() if len(positions) > 0 else -1
            temp_point.append(last_pos)
        batch_start_token.append(temp_point)
        # 将编码后的批次添加到encoded_prompt列表中
        encoded_prompts.append(encoded_batch)
    
    # # 将encoded_prompt中的所有批次合并为一个字典
    # all_encoded = {}
    # for key in encoded_prompt[0].keys():
    #     all_encoded[key] = torch.cat([batch[key] for batch in encoded_prompt]).cpu()
    
    return encoded_prompts, batch_start_token 


def get_last_token_hidden_states(encoded_prompts, model, batch_start_token):
    model.eval()  # 将模型设置为评估模式
    all_last_token_hidden_states = []
    point = 0

    with torch.no_grad():  # 不计算梯度
        for batch, strat_tokens in tqdm(zip(encoded_prompts, batch_start_token), total=len(encoded_prompts)):
            gpu_device = model.device
            gpu_batch = {k: v.to(gpu_device) for k, v in batch.items()}
            outputs = model(**gpu_batch, output_hidden_states=True)
            
            # 获取所有层的hidden states
            all_hidden_states = outputs.hidden_states
            
            # 获取最后一个token的hidden states
            last_token_hidden_states = []
            seq_len = gpu_batch['input_ids'].size(1)
            random_pos = []
            for i in range(len(strat_tokens)):
                random_pos.append(random.randint(strat_tokens[i], seq_len-1))

            for layer_hidden_states in all_hidden_states:
                layer_last_token_hidden_states = layer_hidden_states[torch.arange(layer_hidden_states.size(0)), random_pos]
                last_token_hidden_states.append(layer_last_token_hidden_states.cpu())
            
            last_token_hidden_states_torch = torch.stack(last_token_hidden_states, dim=1)
            
            all_last_token_hidden_states.append(last_token_hidden_states_torch)

    all_last_token_hidden_states = torch.cat(all_last_token_hidden_states, dim=0)

    return all_last_token_hidden_states





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Meta-Llama-3-8B-Instruct_nq")
    parser.add_argument("--original-file", type=str, default="./output/Meta-Llama-3-8B-Instruct_nq/llama3_honest_plustemp_0.8_n_40_llama3_qa_nq_test_label_processed.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="llama3_honest_plus_pair_test")
    parser.add_argument("--label_only", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # setup_env(gpu_s=args.gpus, seed=args.seed)

    print("======= Argument Values =======")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("===============================")

    model_name_wo_path = os.path.basename(args.model_name)
    # file_name = os.path.basename(args.original_file).split(".")[0]
    file_name = os.path.basename(args.original_file)
    file_name = file_name.replace(".json", "")


    model_name = args.model_name
    num_gpus = args.gpus
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    max_memory={0: "0GiB", 1: "0GiB", 2: "0GiB", 3: "0GiB"}
    max_memory[args.gpus] = "24GiB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=max_memory
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "llama" in model_name:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    bos_flag = (tokenizer.bos_token == tokenizer.pad_token)

    torch.autograd.set_grad_enabled(False)
    model.eval()


    if args.prompt in globals():
        func = globals()[args.prompt]
    else:
        raise NameError(f"Function {args.prompt} not found in utilities module.")


    with open(args.original_file, "r") as f:
        data_list = json.load(f)

    # data_list = data_list[:20]
    prompt_list = []

    for data in data_list:
        prompt_list.extend(func(data))



    encoded_prompt, batch_start_token = batch_encode_prompt(prompt_list, tokenizer, args.batch_size)


    if not args.label_only:
        hidden_states = get_last_token_hidden_states(encoded_prompt, model, batch_start_token) # 输出格式是(sample_num, layer_num, hidden_size)

    candidate_layers = [32, 30, 28, 26, 24]
    os.makedirs(f"./features/{model_name_wo_path}", exist_ok=True)

    if not args.label_only:
        for layer_idx in candidate_layers:
            layer_hidden_state = []
            for i in range(hidden_states.shape[0]):
                layer_hidden_state.append(hidden_states[i][layer_idx].tolist())

            layer_hidden_state = torch.tensor(layer_hidden_state)
            num, dim = layer_hidden_state.shape
            if "test" not in args.original_file:
                layer_hidden_state = layer_hidden_state.reshape(num//2, 2, dim)

            torch.save(layer_hidden_state, f"./features/{model_name_wo_path}/{args.prompt}_{file_name}_random_layer_{layer_idx}.pt")



    # with open(f"/home/xyf/paper/longtail_know/features/{args.prompt}_{model_name_wo_path}_{file_name}_labels.pkl", "wb") as f:
    #     pickle.dump(label_list, f)


            
            



