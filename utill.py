import os
import pdb
import yaml
import json
import numpy as np
import torch
import pickle
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve, matthews_corrcoef  
\
def save_checkpoint(model_checkpoints_folder, state):
    with open(os.path.join(model_checkpoints_folder, 'top_performance.json'), 'w') as outfile:
        json.dump(state, outfile)


def save_test(model_checkpoints_folder, state): 
    with open(os.path.join(model_checkpoints_folder, 'test_performance.json'), 'w') as outfile:
        json.dump(state, outfile)



def get_all_metrics(y_true, y_pred, y_pred_proba, n_classes, prefix=""):
    metrics_dict = {}
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    
    metrics_dict.update({
        f'{prefix}_accuracy': accuracy,
        f'{prefix}_precision': precision,
        f'{prefix}_recall': recall,
        f'{prefix}_f1': f1,
        f'{prefix}_mcc': mcc,
    })
    
    roc_aucs = []
    pr_aucs = []
    
    for i in range(n_classes):
        y_true_binary = (y_true == i)
        
        # ROC-AUC
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        metrics_dict[f'{prefix}_roc_auc_class_{i}'] = roc_auc
        
        # PR-AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, y_pred_proba[:, i])
        pr_auc = auc(recall_curve, precision_curve)
        pr_aucs.append(pr_auc)
        metrics_dict[f'{prefix}_pr_auc_class_{i}'] = pr_auc

    metrics_dict[f'{prefix}_mean_roc_auc'] = np.mean(roc_aucs)
    metrics_dict[f'{prefix}_mean_pr_auc'] = np.mean(pr_aucs)

    print(f"\n{prefix} Metrics:")
    print(f"  Basic Classification Metrics:")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall: {recall:.4f}")
    print(f"    F1-score: {f1:.4f}")
    print(f"    MCC: {mcc:.4f}")
    
    print(f"  ROC-AUC scores per class:")
    for i, roc_auc in enumerate(roc_aucs):
        print(f"    Class {i}: {roc_auc:.4f}")
    print(f"    Mean ROC-AUC: {np.mean(roc_aucs):.4f}")
    
    print(f"  PR-AUC scores per class:")
    for i, pr_auc in enumerate(pr_aucs):
        print(f"    Class {i}: {pr_auc:.4f}")
    print(f"    Mean PR-AUC: {np.mean(pr_aucs):.4f}")
    
    return metrics_dict

from torch_geometric.utils import to_dense_adj

def batch_to_adj_matrices(edge_index, ptr):
    adj_matrices = []
    num_graphs = ptr.size(0) - 1

    for i in range(num_graphs):
        start, end = ptr[i].item(), ptr[i+1].item()
        num_nodes = end - start

        mask = ((edge_index[0] >= start) & (edge_index[0] < end)) & \
               ((edge_index[1] >= start) & (edge_index[1] < end))

        local_edge_index = edge_index[:, mask] - start  

        if local_edge_index.numel() == 0:
            
            adj = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        else:

            max_val = local_edge_index.max().item()
            if max_val >= num_nodes:
                raise ValueError(f"[❗] Index {max_val} out of bound for graph {i} with num_nodes={num_nodes}")
            adj = to_dense_adj(local_edge_index, max_num_nodes=num_nodes)[0]

        adj_matrices.append(adj)

    return adj_matrices


def bernoulli_sampling (prob):
  """ Sampling Bernoulli distribution by given probability.
  
  Args:
    - prob: P(Y = 1) in Bernoulli distribution.
    
  Returns:
    - samples: samples from Bernoulli distribution
  """  

  n, d = prob.shape
  samples = torch.bernoulli(prob)
        
  return samples    



class CosineWarmupSchedulerForLambda:
    def __init__(self, warmup_step, min_lambda=0.1, max_lambda=0.5):
        self.warmup_step = warmup_step
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.epoch = 0 

    def step(self):
        if self.epoch < self.warmup_step:
            cos_inner = math.pi * self.epoch / (2 * self.warmup_step)
            lambda_val = self.min_lambda + (self.max_lambda - self.min_lambda) * (1 - math.cos(cos_inner))
        else:
            lambda_val = self.max_lambda

        self.epoch += 1
        return lambda_val

def get_rank(test_sample_prob, thresholds):
    for key, sample in test_sample_prob.items():
        probs = np.array(sample['prob'])

        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)

        rank_list = [0] * len(probs)
        for rank, idx in enumerate(sorted_indices):
            rank_list[idx] = rank + 1  # 1-based rank

        sample['rank'] = rank_list

        for th in thresholds:
            th_key_mask = f'th_{th:.2f}_gate'
            th_key_count = f'th_{th:.2f}_count'

            mask = (probs > th).astype(int)
            count = mask.sum()

            sample[th_key_mask] = mask.tolist()
            sample[th_key_count] = int(count)               

    return test_sample_prob



def drop_and_pad_tokens_gpu(input_ids: torch.Tensor,
                            preserve_mask: torch.Tensor,
                            pad_token_id: int):

    B, L = input_ids.shape
    lengths = preserve_mask.sum(dim=1)                  # (B,)
    max_len = lengths.max()

    kept_input = input_ids[preserve_mask.bool()] 

    padded_inputs = input_ids.new_full((B, max_len), pad_token_id)  # (B, max_len)

    idx = torch.arange(max_len, device=input_ids.device)[None, :]   # (1, max_len)
    mask = idx < lengths[:, None]                                    # (B, max_len)

    padded_inputs[mask] = kept_input

    return padded_inputs