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
        # 将编码后的批次添加到encoded_prompt列表中
        encoded_prompts.append(encoded_batch)
    
    # # 将encoded_prompt中的所有批次合并为一个字典
    # all_encoded = {}
    # for key in encoded_prompt[0].keys():
    #     all_encoded[key] = torch.cat([batch[key] for batch in encoded_prompt]).cpu()
    
    return encoded_prompts



def get_last_token_hidden_states(encoded_prompts, model, tokenizer, bos_flag):
    model.eval()  # 将模型设置为评估模式
    all_last_token_hidden_states = []

    with torch.no_grad():  # 不计算梯度
        for batch in tqdm(encoded_prompts):
            gpu_device = model.device
            gpu_batch = {k: v.to(gpu_device) for k, v in batch.items()}
            outputs = model(**gpu_batch, output_hidden_states=True)
            
            # 获取所有层的hidden states
            all_hidden_states = outputs.hidden_states
            
            # 获取最后一个token的hidden states
            last_token_hidden_states = []
            seq_len = gpu_batch['input_ids'].size(1)
            for layer_hidden_states in all_hidden_states:
                # 找到最后一个非填充token的位置
                # if bos_flag:
                #     last_token_idx = (batch['input_ids'] != tokenizer.pad_token_id).sum(dim=1)  # 这里如果是start token和pad token相同 那么正好对应last token下标
                # else:
                #     last_token_idx = (batch['input_ids']!= tokenizer.pad_token_id).sum(dim=1) - 1  
                # last_token_idx = torch.where(last_token_idx > 0, last_token_idx, 0)  # 确保至少有一个token
                layer_last_token_hidden_states = layer_hidden_states[torch.arange(layer_hidden_states.size(0)), seq_len-1]
                last_token_hidden_states.append(layer_last_token_hidden_states.cpu())
            
            last_token_hidden_states_torch = torch.stack(last_token_hidden_states, dim=1)
            
            all_last_token_hidden_states.append(last_token_hidden_states_torch)

    all_last_token_hidden_states = torch.cat(all_last_token_hidden_states, dim=0)

    return all_last_token_hidden_states





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Meta-Llama-3-8B-Instruct_squad_2k")
    parser.add_argument("--original-file", type=str, default="./output/Meta-Llama-3-8B-Instruct_squad_2k/llama3_honest_plus_temp1.0_n40_squad_train_4k_list.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="llama3_honest_plus_list")
    parser.add_argument("--label_only", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # setup_env(gpu_s=args.gpus, seed=args.seed)

    print("======= Argument Values =======")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("===============================")

    model_name_wo_path = os.path.basename(args.model_name)
    file_name = os.path.basename(args.original_file)
    file_name = file_name.replace(".json", "")


    model_name = args.model_name
    num_gpus = args.gpus
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    max_memory={0: "0GiB", 1: "0GiB", 2: "0GiB"}
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
    answer_num_per_question = len(data_list[0])

    if "test" in args.prompt:
        for data in data_list:
            prompt_list.extend(func(data))
    else:
        for data in data_list:
            for item in data:
                prompt_list.append(func(item))



    encoded_prompt = batch_encode_prompt(prompt_list, tokenizer, args.batch_size)


    if not args.label_only:
        hidden_states = get_last_token_hidden_states(encoded_prompt, model, tokenizer, bos_flag) # 输出格式是(sample_num, layer_num, hidden_size)

    candidate_layers = [32, 30, 28, 26, 24, 22, 20, 18, 16]
    os.makedirs(f"./features/{model_name_wo_path}", exist_ok=True)

    if not args.label_only:
        for layer_idx in candidate_layers:
            layer_hidden_state = []
            for i in range(hidden_states.shape[0]):
                layer_hidden_state.append(hidden_states[i][layer_idx].tolist())

            layer_hidden_state = torch.tensor(layer_hidden_state)
            num, dim = layer_hidden_state.shape
            if "test" not in args.prompt:
                layer_hidden_state = layer_hidden_state.reshape(num//answer_num_per_question, answer_num_per_question, dim)

            torch.save(layer_hidden_state, f"./features/{model_name_wo_path}/{args.prompt}_{file_name}_layer_{layer_idx}.pt")



    # with open(f"/home/xyf/paper/longtail_know/features/{args.prompt}_{model_name_wo_path}_{file_name}_labels.pkl", "wb") as f:
    #     pickle.dump(label_list, f)


            
            



