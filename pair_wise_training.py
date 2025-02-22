import torch
import torch.nn as nn
import torch.nn.functional as F
import json 
import os
from tqdm import tqdm
import argparse
import wandb


class Linear_probing(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一层后接ReLU激活函数
        x = self.dropout(x)      # 添加dropout层
        x = self.fc2(x)          # 输出层
        return x



def load_data(file_path):
    data = torch.load(file_path)
    return data


def train_pairwise(model, dataloader, optimizer, device, batch_size, epoch):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    
    # 假设data是一个包含pairs和labels的列表
    # dataset = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", position=1, leave=False)):
        pairs = batch[0].to(device)  # [batch_size, 2, dim]
        labels = batch[1].to(device)  # [batch_size]
        
        optimizer.zero_grad()
        
        # 分别获取每对样本的表示
        x1 = pairs[:, 0, :]  # [batch_size, dim]
        x2 = pairs[:, 1, :]  # [batch_size, dim]
        
        # 分别通过模型得到得分
        score1 = model(x1).squeeze(-1)  # [batch_size]
        score2 = model(x2).squeeze(-1)  # [batch_size]
        
        # 计算相对得分
        logits = score1 - score2  # [batch_size]
        
        # 计算损失
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss
        
        # 记录每个batch的loss
        wandb.log({
            'batch_loss': batch_loss,
            'epoch': epoch,
            'batch': batch_idx
        })
        
        # 每100个batch打印一次loss
        if (batch_idx + 1) % 100 == 0:
            tqdm.write(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {batch_loss:.4f}")
    
    avg_loss = total_loss / len(dataloader)
    wandb.log({
        'epoch_loss': avg_loss,
        'epoch': epoch
    })
    
    return avg_loss



def evaluate_pairwise(model, data, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data:
            pairs = batch[0].to(device)
            labels = batch[1].to(device)
            
            x1 = pairs[:, 0, :]
            x2 = pairs[:, 1, :]
            
            score1 = model(x1).squeeze(-1)
            score2 = model(x2).squeeze(-1)
            
            # 预测第一个样本的得分是否大于第二个
            predictions = (score1 > score2).float()
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return correct / total




def evaluate_listwise(model, data, device):
    """
    Evaluate a listwise ranking model.
    Accuracy is defined as having the document with the highest label ranked first.
    Args:
        model: The ranking model.
        data: DataLoader or dataset providing [batch_size, n, dim] and labels [batch_size, n].
        device: The device to run the evaluation on (e.g., 'cuda' or 'cpu').
    Returns:
        accuracy: The accuracy of the model.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data:
            items = batch[0].to(device)  # [batch_size, n, dim]
            labels = batch[1].to(device)  # [batch_size, n]

            # Flatten pairs for model prediction
            batch_size, n, dim = items.size()
            items_flat = items.view(-1, dim)  # [batch_size * n, dim]

            # Get scores from the model
            scores = model(items_flat).view(batch_size, n)  # [batch_size, n]
            
            # 创建掩码并应用
            mask = (labels == -1)
            scores = scores.masked_fill(mask, -1e9)  # 将标签为-1的位置的分数设为很大的负数

            # Get the predicted ranking (descending order)
            predicted_ranking = scores.argsort(dim=1, descending=True)  # [batch_size, n]

            # Find the index of the highest label for each query
            max_label_indices = labels.argmax(dim=1)  # [batch_size]

            # Check if the highest label is ranked first
            for i in range(batch_size):
                if labels[i, 0] == labels[i, predicted_ranking[i, 0]]:  # Check top-1
                    correct += 1

            total += batch_size

    return correct / total






def annotate_data(model, data):
    model.eval()
    # 修改：直接使用model所在的设备，而不是尝试访问model.device
    device = next(model.parameters()).device
    score_list = []
    
    with torch.no_grad():
        for batch in data:
            # 修改：因为使用了TensorDataset，需要获取batch中的第一个元素
            features = batch[0].to(device)
            score = model(features).squeeze(-1)
            score = score.cpu().tolist()
            score_list.extend(score)  # 使用extend而不是append来展平结果
            
    return score_list





def create_dataloader(data, batch_size):
    """
    data: 包含pairs和labels的字典列表
    返回: DataLoader对象
    """
    # pairs = torch.tensor([item['pair'] for item in data])  # [N, 2, DIM]
    labels = torch.ones(size=(data.shape[0],))
    
    dataset = torch.utils.data.TensorDataset(
        data,
        labels
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )



def create_test_dataloader(data, batch_size):
    """
    data: 包含pairs和labels的字典列表
    返回: DataLoader对象
    """
    
    dataset = torch.utils.data.TensorDataset(
        data
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )


def get_layer_num(file_path):
    # 从文件路径中提取layer数字
    layer_str = file_path.split('_layer_')[-1].replace('.pt', '')
    return int(layer_str)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./features/Meta-Llama-3-8B-Instruct_squad_2k/llama3_honest_plus_pair_llama3_honest_plus_temp1.0_n40_squad_train_4k_pair_layer_16.pt")
    parser.add_argument("--test_data_path", type=str, default="./features/Meta-Llama-3-8B-Instruct_squad_2k/llama3_honest_plus_pair_test_llama3_honest_plus_temp1.0_n40_squad_val_1k_processed_layer_16.pt")
    parser.add_argument("--test_text_file", type=str, default="./output/Meta-Llama-3-8B-Instruct_squad_2k/llama3_honest_plus_temp1.0_n40_squad_val_1k_processed.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--gpus", type=int, default=0)
    args = parser.parse_args()

    wandb.init(
        project="Conflict_Inst",
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        }
    )

    data = load_data(args.data_path)
    test_data = load_data(args.test_data_path)


    # 随机打乱数据
    indices = torch.randperm(len(data))
    data = data[indices]
    
    # 划分训练集和验证集
    train_data = data
    val_data = data[int(len(data)*0.8):]

    DEVICE = f"cuda:{args.gpus}" if torch.cuda.is_available() else "cpu"

    # 初始化模型
    input_dim = 4096  # 获取输入维度
    model = Linear_probing(input_dim, 1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 创建数据加载器
    train_loader = create_dataloader(train_data, batch_size=args.batch_size)
    val_loader = create_dataloader(val_data, batch_size=args.batch_size)
    test_loader = create_test_dataloader(test_data, batch_size=args.batch_size)
    
    
    best_model_state = None
    best_val_acc = 0
    best_test_acc = 0
    best_probing_json = None


    with open(args.test_text_file, "r") as f:
        test_json = json.load(f)

    # 训练循环
    for epoch in tqdm(range(args.epochs), desc="Training epochs", position=0):
        loss = train_pairwise(model, train_loader, optimizer, DEVICE, args.batch_size, epoch)
        val_acc = evaluate_pairwise(model, val_loader, DEVICE)
        score_list = annotate_data(model, test_loader)

        probing_json = test_json.copy()
        idx = 0
        for item in probing_json:
            for candidate in item['candidate_answers']:
                candidate['probing_score'] = score_list[idx]
                idx += 1

        test_correct = 0
        for item in probing_json:
            group_prob = []
            for candidate in item['candidate_answers']:
                group_prob.append({'label': candidate['label'], 'output': candidate['output'], 'score': candidate['score'], 'probing_score': candidate['probing_score']})
            label_sorted = sorted(group_prob, key=lambda x: x['label'], reverse=True)
            score_sorted = sorted(group_prob, key=lambda x: x['probing_score'], reverse=True)
            if score_sorted[0]['label'] == label_sorted[0]['label']:
                test_correct += 1

        test_acc = test_correct / len(probing_json)


        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict()
            best_probing_json = probing_json
            # torch.save(model.state_dict(), f"best_model.pth")
        wandb.log({
            'val_acc': val_acc,
            'test_acc': test_acc,
            'epoch': epoch
        })
        
    print(f"Best test accuracy: {best_test_acc:.4f}")

    wandb.finish()

    # model.load_state_dict(best_model_state)


    
    # score_list = annotate_data(model, test_loader)

    with open(args.test_text_file, "r") as f:
        test_json = json.load(f)

    idx = 0
    probing_json = test_json.copy()
    for item in probing_json:

        for candidate in item['candidate_answers']:
            candidate['probing_score'] = score_list[idx]
            idx += 1

    layer_num = get_layer_num(args.data_path)
    test_text_file_name = args.test_text_file.split('/')[-1]
    folder_path = os.path.dirname(args.test_text_file)
    test_text_file_name = test_text_file_name.replace('.json', f'_layer_{layer_num}_pair.json')
    test_text_file_name = os.path.join(folder_path, test_text_file_name)


    with open(test_text_file_name, "w") as f:
        json.dump(best_probing_json, f, indent=4)


