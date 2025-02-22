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

    # Apply softmax to masked y_pred along the last dimension
    softmax_preds = F.softmax(masked_y_pred, dim=1)

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
