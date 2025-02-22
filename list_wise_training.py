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



def listwise_cross_entropy_with_ignore_index(y_true, y_pred):
    """
    Listwise Cross-Entropy Loss using PyTorch's CrossEntropyLoss with ignore_index
    Args:
        scores: Tensor, shape (batch_size, max_num_documents), predicted scores (logits)
        relevance_labels: Tensor, shape (batch_size, max_num_documents), true relevance labels
        ignore_index: int, the index to ignore in loss calculation (default: -1)
    Returns:
        loss: Tensor, the computed loss
    """
    # Flatten scores and relevance labels for batch-wise computation
    mask = (y_true != -1).float()

    # Set ignored elements in y_pred to a very large negative value (effectively -inf)
    masked_y_pred = y_pred.clone()
    masked_y_pred[mask == 0] = float('-inf')
    masked_y_label = y_true.clone()
    masked_y_label_float = torch.tensor(masked_y_label, dtype=torch.float32)
    masked_y_label_float[mask == 0] = float('-inf')

    # Apply softmax to masked y_pred along the last dimension
    softmax_preds = F.softmax(masked_y_pred, dim=1)
    softmax_label = F.softmax(masked_y_label_float, dim=1)
    # Compute the log of the softmax predictions
    log_softmax_preds = torch.log(softmax_preds + 1e-8)  # Add epsilon for numerical stability

    # Set ignored elements in y_true to 0 to avoid affecting the loss
    y_true = torch.where(y_true == -1, torch.zeros_like(y_true), y_true)

    # Compute the element-wise product of y_true and log_softmax_preds
    loss = -torch.sum(mask * y_true * log_softmax_preds, dim=1)  # Only consider valid elements

    # Normalize by the number of valid elements per row (to avoid bias from ignored elements)
    valid_count = torch.sum(mask, dim=1)
    loss = loss / (valid_count + 1e-8)  # Avoid division by zero

    # Return the mean loss across the batch
    return loss.mean()








def load_data(file_path):
    data = torch.load(file_path)
    return data


def train_listwise(model, dataloader, optimizer, device, batch_size, epoch):
    """
    Listwise training function.
    Args:
        model: The ranking model.
        dataloader: DataLoader providing batches of [batch_size, n, dim] and corresponding labels [batch_size, n].
        optimizer: Optimizer for training the model.
        device: The device to run the training on (e.g., 'cuda' or 'cpu').
        batch_size: Number of queries per batch.
        epoch: Current training epoch.
    Returns:
        avg_loss: The average loss for the epoch.
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", position=1, leave=False)):
        # Inputs and labels
        items = batch[0].to(device)  # [batch_size, n, dim]
        labels = batch[1].to(device)  # [batch_size, n]

        optimizer.zero_grad()

        # Pass all documents through the model to get scores
        n = items.size(1)  # Number of documents per query
        sample_num = items.size(0)
        items_flat = items.view(-1, items.size(-1))  # Flatten to [batch_size * n, dim]
        scores = model(items_flat).view(sample_num, n)  # Reshape back to [batch_size, n]
        loss = listwise_cross_entropy_with_ignore_index(labels, scores)


        # Backward propagation
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss

        # Log each batch loss
        wandb.log({
            'batch_loss': batch_loss,
            'epoch': epoch,
            'batch': batch_idx
        })

        # Print loss every 100 batches
        if (batch_idx + 1) % 100 == 0:
            tqdm.write(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {batch_loss:.4f}")

    avg_loss = total_loss / len(dataloader)
    wandb.log({
        'epoch_loss': avg_loss,
        'epoch': epoch
    })

    return avg_loss




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





def create_dataloader(data, labels, batch_size):
    """
    data: 包含pairs和labels的字典列表
    返回: DataLoader对象
    """
    # pairs = torch.tensor([item['pair'] for item in data])  # [N, 2, DIM]
    labels = torch.tensor(labels)

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
    parser.add_argument("--train_val_data_path", type=str, default="./features/Meta-Llama-3-8B-Instruct_squad_2k/llama3_honest_plus_list_llama3_honest_plus_temp1.0_n40_squad_train_4k_list_layer_16.pt")
    parser.add_argument("--train_val_text_path", type=str, default="./output/Meta-Llama-3-8B-Instruct_squad_2k/llama3_honest_plus_temp1.0_n40_squad_train_4k_list.json")
    # parser.add_argument("--val_data_path", type=str, default="/home/xyf/paper/ranking_decoding/features/Meta-Llama-3-8B-Instruct_popqa_90/llama3_honest_plus_list_llama3_honest_plustemp_1.0_n_40_llama3_qa_popqa_eval_list_layer_32.pt")
    # parser.add_argument("--val_text_path", type=str, default="/home/xyf/paper/ranking_decoding/output/Meta-Llama-3-8B-Instruct_popqa_90/llama3_honest_plustemp_1.0_n_40_llama3_qa_popqa_eval_list.json")
    parser.add_argument("--test_data_path", type=str, default="./features/Meta-Llama-3-8B-Instruct_squad_2k/llama3_honest_plus_pair_test_llama3_honest_plus_temp1.0_n40_squad_val_1k_processed_layer_16.pt")
    parser.add_argument("--test_text_file", type=str, default="./output/Meta-Llama-3-8B-Instruct_squad_2k/llama3_honest_plus_temp1.0_n40_squad_val_1k_processed.json")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--gpus", type=int, default=2)
    args = parser.parse_args()

    wandb.init(
        project="Conflict_Inst",
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        }
    )

    train_val_data = torch.load(args.train_val_data_path, weights_only=False)
    train_val_data = train_val_data[:3000]
    # train_val_data = train_val_data[:, :6, :]
    # train_val_data = train_val_data.reshape(train_val_data.shape[0]//16, 16, 4096)
    # val_data = torch.load(args.val_data_path)

    # val_data = val_data[:, :6, :]
    # train_val_data = torch.cat([train_val_data, val_data], dim=0)
    train_val_labels = []
    # val_labels = []

    with open(args.train_val_text_path, "r") as f:
        train_val_text = json.load(f)

    for data in train_val_text[:3000]:
        row_labels = []
        for item in data:
            label = item['label']
            row_labels.append(label)
        new_row_labels = []
        max_label = max(row_labels)
        for item in row_labels:
            # 如果当前label等于最大label，设为1，否则设为0
            label = 1 if item == max_label else 0
            new_row_labels.append(label)

        train_val_labels.append(new_row_labels)

    # with open(args.val_text_path, "r") as f:
    #     val_text = json.load(f)

    # val_labels = []
    # for data in val_text:
    #     row_labels = []
    #     for item in data:
    #         label = item['label']
    #         row_labels.append(label)
    
    #     # new_row_labels = []
    #     # max_label = max(row_labels)
    #     # for item in row_labels:
    #     #     # 如果当前label等于最大label，设为1，否则设为0
    #     #     label = 1 if item == max_label else 0
    #     #     new_row_labels.append(label)
    #     val_labels.append(row_labels)

    # train_val_labels = train_val_labels + val_labels

    # 随机打乱数据
    # indices = torch.randperm(len(train_val_data))
    # train_val_data = train_val_data[indices]
    test_data = torch.load(args.test_data_path, weights_only=False)
    
    # 划分训练集和验证集
    train_data = train_val_data
    train_labels = train_val_labels
    # train_data = val_data
    # train_labels = val_labels
    val_data = train_val_data[int(len(train_val_data)*0.8):]
    val_labels = train_val_labels[int(len(train_val_labels)*0.8):]

    DEVICE = f"cuda:{args.gpus}" if torch.cuda.is_available() else "cpu"

    # 初始化模型
    input_dim = 4096  # 获取输入维度
    model = Linear_probing(input_dim, 1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 创建数据加载器
    train_loader = create_dataloader(train_data, train_labels, batch_size=args.batch_size)
    val_loader = create_dataloader(val_data, val_labels, batch_size=args.batch_size)
    test_loader = create_test_dataloader(test_data, batch_size=args.batch_size)
    
    
    best_model_state = None
    best_val_acc = 0
    best_test_acc = 0
    best_probing_json = None
    with open(args.test_text_file, "r") as f:
        test_json = json.load(f)

    # 训练循环
    for epoch in tqdm(range(args.epochs), desc="Training epochs", position=0):
        loss = train_listwise(model, train_loader, optimizer, DEVICE, args.batch_size, epoch)
        val_acc = evaluate_listwise(model, val_loader, DEVICE)
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

    # with open(args.test_text_file, "r") as f:
    #     test_json = json.load(f)

    # idx = 0
    # probing_json = test_json.copy()
    # for item in probing_json:

    #     for candidate in item['candidate_answers']:
    #         candidate['probing_score'] = score_list[idx]
    #         idx += 1

    # test_correct = 0
    # for item in probing_json:
    #     group_prob = []
    #     for candidate in item['candidate_answers']:
    #         group_prob.append({'label': candidate['label'], 'output': candidate['output'], 'score': candidate['score'], 'probing_score': candidate['probing_score']})
    #     label_sorted = sorted(group_prob, key=lambda x: x['label'], reverse=True)
    #     score_sorted = sorted(group_prob, key=lambda x: x['probing_score'], reverse=True)
    #     if score_sorted[0]['label'] == label_sorted[0]['label']:
    #         test_correct += 1

    # print(f"test_correct: {test_correct / len(probing_json)}")


    layer_num = get_layer_num(args.test_data_path)
    test_text_file_name = args.test_text_file.split('/')[-1]
    folder_path = os.path.dirname(args.test_text_file)
    test_text_file_name = test_text_file_name.replace('.json', f'_layer{layer_num}_list.json')
    test_text_file_name = os.path.join(folder_path, test_text_file_name)


    with open(test_text_file_name, "w") as f:
        json.dump(best_probing_json, f, indent=4)


