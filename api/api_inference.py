import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
from openai import OpenAI
import argparse

sys.path.append('/home/xyf/paper/longtail_know/utils')
from prompt_lib import *

# 定义客户端池，每个线程使用不同的账号
clients = [
    OpenAI(api_key="sk-e6c43f7d18b14bd78c1bc7490ac015cf", base_url="https://api.deepseek.com"),
    OpenAI(api_key="sk-2e4c194ea6d548d398f8ac3927ca84a9", base_url="https://api.deepseek.com"),
    OpenAI(api_key="sk-7648a8acaafb43e1a60e4adfe78a1f0e", base_url="https://api.deepseek.com"),
    OpenAI(api_key="sk-c35d0b539d81446e82ba1b323dc6e3e4", base_url="https://api.deepseek.com"),
    OpenAI(api_key="sk-f6fe81fd11d44018b6669b3657df77a6", base_url="https://api.deepseek.com"),
    OpenAI(api_key="sk-7fd7a42d5685426b92318fe7f98c738a", base_url="https://api.deepseek.com"),
    OpenAI(api_key="sk-960debd2de8b41f7904c048031f0ec52", base_url="https://api.deepseek.com"),
    OpenAI(api_key="sk-e8215ffc9f8f4d1dbd41e6baf6ba03e6", base_url="https://api.deepseek.com"),
    OpenAI(api_key="sk-136aea61526147cc9e0cae37de842fbc", base_url="https://api.deepseek.com"),
    OpenAI(api_key="sk-4efaa8ce0b114ec8ae2c420bcfe20ad6", base_url="https://api.deepseek.com"),
    OpenAI(api_key="sk-aa23463ba30d4c91bbf6455422b3ec00", base_url="https://api.deepseek.com"),
]

def deepseek_check_answer(data):
    # 如果ground_truth是列表，则用逗号拼接；如果不是列表，则直接使用
    ground_truth = ', '.join(data['ground_truth']) if isinstance(data['ground_truth'], (list, tuple)) else data['ground_truth']
    
    prompt_template = f'''Given a question, ground truth, and answer, determine whether the answer is correct according to the ground truth. The answer can be a synonym or a more detailed or broader description of the ground truth, but it must not have a different meaning. Output "correct" if the answer is accurate, and "incorrect" if it is not. Do not provide any explanations.\n\n
    Question: {data['question']} \n
    Ground Truth: {ground_truth} \n
    Answer: {data['predictions'][0]} \n
    '''

    return prompt_template


entities = set()
return_list = []
write_lock = Lock()  # 用于保护写入操作的锁

INPUT_FILE = "/home/xyf/paper/ranking_decoding/output/Meta-Llama-3-8B-Instruct/squad/llama3_honest_plustemp_0.8_n_40_squad_test.json"
OUTPUT_FILE = INPUT_FILE.replace(".json", "_deepseek.json")

# 读取输入数据
with open(INPUT_FILE, "r") as f:
    data_list = json.load(f)

for i in range(len(data_list)):
    data_list[i] = {
        'question': data_list[i]['question'],
        'ground_truth': data_list[i]['ground_truth'],
        'predictions': [data_list[i]['predictions'][0]]
    }


def process_item(item, client):
    """
    处理单个任务项。
    """
    func = globals()["deepseek_check_answer"]
    prompt = func(item)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    api_output = response.choices[0].message.content
    new_item = item.copy()
    new_item["factuality"] = api_output
    if api_output == "correct":
        new_item["conflict_label"] = 0
    else:
        new_item["conflict_label"] = 1

    return new_item

# 多线程处理
with ThreadPoolExecutor(max_workers=11) as executor:
    futures = {
        executor.submit(process_item, item, random.choice(clients)): item
        for item in data_list
    }
    for future in tqdm(as_completed(futures), total=len(data_list)):
        try:
            result = future.result()
            with write_lock:
                return_list.append(result)  # 使用锁保护写入操作
        except Exception as e:
            print(f"Error processing item {futures[future]}: {e}")

# 写入输出数据
with open(OUTPUT_FILE, "w") as f:
    json.dump(return_list, f, indent=4)


