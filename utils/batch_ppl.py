import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm



def calculate_ppl(model, tokenizer, instructions, outputs):
    """
    计算给定 instruction 和 output 的 PPL 值
    """
    # Tokenize instruction and output
    DEVICE = model.device
    output_len = []
    for output in outputs:
        output_tokens = tokenizer(output, return_tensors="pt").input_ids
        output_len.append(output_tokens.shape[1])

    batch_instructions = tokenizer(instructions, return_tensors="pt", padding=True, truncation=True)
    batch_instructions_input = [{k: v[j] for k, v in batch_instructions.items()} for j in range(len(instructions))]
    input_ids = torch.stack([p['input_ids'] for p in batch_instructions_input]).to(DEVICE)
    attention_mask = torch.stack([p['attention_mask'] for p in batch_instructions_input]).to(DEVICE)
    labels = input_ids.clone()
    for i in range(len(output_len)):
        labels[i, :-output_len[i]] = -100
    


    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # 计算每个token位置的loss
        token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # 重塑为 [batch_size, sequence_length]
        token_loss = token_loss.view(shift_labels.size())
        # 使用attention mask来计算每个样本的平均loss
        mask = (shift_labels != -100).float()
        sample_losses = (token_loss * mask).sum(dim=1) / mask.sum(dim=1)

    sample_ppl = torch.exp(sample_losses)
    sample_probs = 1 / sample_ppl


    return sample_probs.tolist(), sample_losses.tolist(), sample_ppl.tolist()


