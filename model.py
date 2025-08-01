import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import math
from torch_geometric.nn import NNConv, Set2Set, GCNConv
from torch_geometric.nn.aggr import MLPAggregation
from torch_geometric.nn import MLP
from torch_geometric.utils import softmax

from torch_scatter import scatter_mean, scatter_add, scatter_std
import pdb
import numpy as np
from utill import batch_to_adj_matrices, bernoulli_sampling, drop_and_pad_tokens_gpu

import networkx as nx
from torch.special import erf

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_gates, gate_threshold=0.5):
        
        
        discrete_gates = (token_gates >= gate_threshold).float()
        return discrete_gates

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
        
class STE_S_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_gates):
        
        discrete_gates = bernoulli_sampling(token_gates).float()
        return discrete_gates

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
def find_first_one_argmax(attention_mask: torch.Tensor) -> torch.Tensor:

    
    has_one = attention_mask.bool().any(dim=1)
    
    
    first_one = (attention_mask.float().argmax(dim=1))
    
    
    first_one = first_one.masked_fill(~has_one, -1)
    
    return first_one

def find_last_one_argmax(attention_mask: torch.Tensor) -> torch.Tensor:

    
    has_one = attention_mask.bool().any(dim=1)

    
    reversed_mask = attention_mask.flip(dims=[1])  
    last_one_from_end = reversed_mask.float().argmax(dim=1)
    last_one = attention_mask.size(1) - 1 - last_one_from_end

    
    last_one = last_one.masked_fill(~has_one, -1)

    return last_one

def create_token_to_node_mapping(gate_inputs, data_ptr, meta_data, batch_size, seq_length, token_start_idx, token_end_idx, device):
    
    total_nodes = len(gate_inputs)
    token_to_node_mapping = torch.zeros((total_nodes, batch_size, seq_length), device=device)
    node_idx_offset = 0

    for batch_idx in range(batch_size):

        start_ptr = data_ptr[batch_idx].item()
        end_ptr = data_ptr[batch_idx + 1].item()
        batch_gates = gate_inputs[start_ptr:end_ptr]

        node2token_start = meta_data['node2token'][batch_idx][0][:batch_gates.shape[0]].to(device)
        node2token_length = meta_data['node2token'][batch_idx][1][:batch_gates.shape[0]].to(device)

        st_idx = token_start_idx[batch_idx].item()
        ed_idx = token_end_idx[batch_idx].item()

        for node_idx, gate in enumerate(batch_gates):
            if node_idx>=seq_length:
                break
            start = node2token_start[node_idx].item()
            length = node2token_length[node_idx].item()

            if not torch.isnan(node2token_start[node_idx]) and start >= 0 and length > 0:
                start_idx = st_idx + int(start)
                end_idx = min(start_idx + int(length), ed_idx)  

                if 0 <= start_idx < ed_idx:
                    node_global_idx = node_idx_offset + node_idx
                    if node_global_idx < total_nodes:
                        token_to_node_mapping[node_global_idx, batch_idx, start_idx:end_idx] = 1.0
                    else:
                        raise IndexError(f"node_global_idx {node_global_idx} exceeds total_nodes {total_nodes}")

        node_idx_offset += len(batch_gates)

    return token_to_node_mapping


filtered_words = []

def filter_unmasked_words(hard_gate_, data_ptr_clean, raw_words, idx, mask=False):
    filtered_words = {}

    
    for batch_idx in range(len(data_ptr_clean) - 1):
        start_idx = data_ptr_clean[batch_idx].item()
        end_idx = data_ptr_clean[batch_idx + 1].item()

        
        batch_hard_gate = hard_gate_[start_idx:end_idx, 0]  # [batch_size]
        batch_words = raw_words[batch_idx]  # [batch_size]
        sample_id = idx[batch_idx].item()

        if mask:
            
            unmasked_words = [
                word if gate == 1 else "[MASK]"
                for word, gate in zip(batch_words, batch_hard_gate)
            ]
        else:
            
            unmasked_words= []
            for node_idx, gate in enumerate(batch_hard_gate):
                if gate == 1:
                    unmasked_words = batch_words[node_idx]


        filtered_words[sample_id] = unmasked_words 

    return filtered_words

def create_masked_token_dict(input_tokens, modified_attention_mask, valid_token_counts, meta_data_idx):
    
    batch_size = input_tokens.shape[0]
    
    masked_token = (input_tokens * modified_attention_mask[-batch_size:])

    
    masked_token_dict = {}
    for batch_idx, sample_id in enumerate(meta_data_idx):
        valid_count = valid_token_counts[batch_idx].item()
        masked_token_dict[sample_id.item()] = masked_token[batch_idx, :valid_count].tolist()

    return masked_token_dict

def create_sample_dict(gate_inputs, data_ptr_clean, meta_data):

    sample_dict = {}

    
    for batch_idx in range(len(data_ptr_clean) - 1):
        start_idx = data_ptr_clean[batch_idx].item()
        end_idx = data_ptr_clean[batch_idx + 1].item()

        
        batch_probs = gate_inputs[start_idx:end_idx].squeeze(1).tolist()  
        batch_words = meta_data['raw_word'][batch_idx] 
        sample_id = meta_data['idx'][batch_idx].item()

   
        sample_dict[sample_id] = {
            'words': batch_words,
            'prob': batch_probs,
        }

    return sample_dict

def create_grouped_sample_dict(gate_inputs, data_ptr_clean, split_matrix, meta_data):
    
    grouped_sample_dict = {}

    
    for batch_idx in range(len(split_matrix)):
        start_idx = data_ptr_clean[batch_idx].item()
        end_idx = data_ptr_clean[batch_idx + 1].item()

        
        batch_gate_inputs = gate_inputs[start_idx:end_idx].squeeze(1) 
        batch_raw_words = meta_data['raw_word'][batch_idx] 
        sample_id = meta_data['idx'][batch_idx].item() 

        adj_np = split_matrix[batch_idx].cpu().numpy()
        G = nx.from_numpy_array(adj_np)
        components = list(nx.connected_components(G)) 

        groups = []
        probs = []
        for group in components:
            group = list(group)
            group_words = [batch_raw_words[node] for node in group]
            group_probs = batch_gate_inputs[group].tolist()

            groups.append(group_words)
            probs.append(group_probs)

        grouped_sample_dict[sample_id] = {
            'groups': groups,
            'probs': probs
        }

    return grouped_sample_dict

def group_adj_matrices_by_sample(adj_matrix_tuple, meta_data):
    adjacency_matrix_list, semantic_adj_list, split_matrix_list = adj_matrix_tuple
    sample_ids = meta_data['idx']

    grouped_adj_dict = {}

    for i, sample_id in enumerate(sample_ids):
        sample_id = sample_id.item() if isinstance(sample_id, torch.Tensor) else sample_id

        grouped_adj_dict[sample_id] = {
            'adj': adjacency_matrix_list[i],
            'semantic_adj': semantic_adj_list[i],
            'split_matrix': split_matrix_list[i],
        }

    return grouped_adj_dict

def create_score_matrices_dict(score_matrices, batch_size, meta_data):

    score_matrices_dict = {}

    for batch_idx in range(batch_size):
        sample_id = meta_data['idx'][batch_idx].item()

        score_matrices_dict[sample_id] = {
            'relu': score_matrices[0][batch_idx],
            'not_relu': score_matrices[1][batch_idx],
            'softmax': score_matrices[2][batch_idx],
        }

    return score_matrices_dict

def apply_special_token_mask(modified_attention_mask, input_ids, special_token_ids, num_samples):

    special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in special_token_ids:
        special_token_mask |= (input_ids == token_id)

    special_token_mask = special_token_mask.repeat(num_samples, 1)
    modified_attention_mask[special_token_mask] = 1

    return modified_attention_mask
class Model_Align2(nn.Module):
    def __init__(self, tokenizer, predictor_tokenizer, predictor, selected_model, args, mask_embedding=None, **kargs):
        super(Model_Align2, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.predictor = predictor.to(args.device)
        self.selected_model = selected_model
        if args.target_model == 'xlnet':
            self.word_embedding = self.predictor.transformer.word_embedding
        elif args.target_model == 'deberta':
            self.word_embedding = self.predictor.deberta.embeddings.word_embeddings
        elif args.target_model in ['gpt2', 'BioMedLM']:
            self.word_embedding = self.predictor.transformer.wte
        elif args.target_model == 'roberta':
            self.word_embedding = self.predictor.roberta.embeddings.word_embeddings
        elif args.target_model == 'biolinkBert':
            
            self.word_embedding = self.predictor.bert.embeddings.word_embeddings
        self.embedding_output = None
        self.selected_model_type = args.selected_model_type

        if args.tokenizer_type == 'xlnet':
            mask_token_id = self.tokenizer.mask_token_id
            if mask_token_id is None:
                mask_token_id = self.tokenizer.unk_token_id
            self.mask_text_id = self.tokenizer.convert_tokens_to_ids("▁")
            
        elif args.tokenizer_type in ['gpt2', 'BioMedLM']:
            self.mask_text_id = self.tokenizer.convert_tokens_to_ids("Ġ")

        elif args.tokenizer_type in ['deberta', 'roberta', 'BioMedLM', 'biolinkBert', 'deberta_large', 'deberta_small']:
            self.mask_text_id = self.tokenizer.convert_tokens_to_ids("Ġ")

        special_tokens = self.tokenizer.special_tokens_map
        self.special_token_ids = [self.tokenizer.convert_tokens_to_ids(v) for k, v in special_tokens.items()]
        if self.args.data_name == 'bioasq':
            if self.args.target_model == 'BioMedLM':
                self.special_token_ids.extend(self.tokenizer.special_tokens_map['additional_special_tokens'])
        self.predictor_tokenizer = predictor_tokenizer
        self.mask_embedding = mask_embedding
        # self.mask_embedding.requires_grad_(False)
        
        
    def forward(self, data, meta_data, test=False, baseline = False, train_threshold=False):

        texts = meta_data['text']


        if baseline:
            if self.args.target_model == 'BioMedLM':
                
                context_prompt = [f"[CONTEXT] {x['context'].strip()}" for x in texts]
                question_prompt = [f"[QUESTION] {x['question'].strip()} [ANSWER]" for x in texts]
                
                # context_prompt = f"[CONTEXT] {context} [QUESTION]"
                # question_prompt = f"{question} [ANSWER]"

                baseline_encoded = self.predictor_tokenizer(
                    context_prompt,
                    text_pair=question_prompt,
                    padding=True,
                    truncation="only_first",
                    max_length=512,
                    # max_length=1024,
                    return_tensors="pt"
                ).to(self.args.device)

                outputs = self.predictor(
                    input_ids=baseline_encoded['input_ids'],
                    attention_mask=baseline_encoded['attention_mask']
                )
                return outputs

            else:
                baseline_encoded = self.predictor_tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=512 ,
                        return_tensors='pt'
                        ).to(self.args.device)
            
                if self.args.target_model in ['deberta', 'xlnet']:
                    outputs = self.predictor(
                        input_ids=baseline_encoded['input_ids'],
                        attention_mask=baseline_encoded['attention_mask'],
                        token_type_ids=baseline_encoded['token_type_ids']
                    )
                elif self.args.target_model in ['roberta', 'gpt2', 'BioMedLM', 'biolinkBert']:
                    outputs = self.predictor(
                        input_ids=baseline_encoded['input_ids'],
                        attention_mask=baseline_encoded['attention_mask']
                    )
                return outputs

        if self.args.target_model == 'BioMedLM':

            context_prompt = texts['context']
            question_prompt = texts['question']
            # context_prompt = f"[CONTEXT] {context} [QUESTION]"
            # question_prompt = f"{question} [ANSWER]"

            encoded = self.tokenizer(
                context_prompt,
                text_pair=question_prompt,
                padding=True,
                truncation="only_first",
                max_length=512,
                # max_length=1024,
                return_tensors="pt"
            ).to(self.args.device)
        else:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.args.device)
        
        original_embeddings = self.word_embedding(encoded['input_ids']).to(self.args.device) # torch.Size([32, 26, 768]) batch size 32 기준.
        
        selector_out = self.selected_model(data, model_type=self.selected_model_type)
        gate_inputs = selector_out['gate_inputs']
        test_gate_inputs = selector_out['test_gate_inputs']
        regularizer = selector_out['regularizer']
        loss_smo = selector_out['loss_smo']
        data_ptr_clean = selector_out['data_ptr_clean']

        batch_size = encoded['attention_mask'].size(0)
        seq_length = encoded['attention_mask'].size(1)
        
        #########################################
        
        if not test:
            if self.selected_model_type in ['STE', 'STGS']:  
                hard_gate = STEFunction.apply(
                    gate_inputs,
                )
                
                hard_gate_=hard_gate
                soft_gate=gate_inputs
            if self.selected_model_type in ['STE_S']:  
                
                hard_gate_list = []
                soft_gate_list = []
                for _ in range(self.args.num_samples):
                    hard_gate_=STE_S_Function.apply(
                        gate_inputs
                    )
                    hard_gate_list.append(hard_gate_)
                    soft_gate_list.append(gate_inputs)
                
                hard_gate = torch.stack(hard_gate_list,dim=1) # total_node, num_samples
                soft_gate = torch.stack(soft_gate_list,dim=1) # total_node, num_samples

            elif self.selected_model_type == 'RL':
                hard_gate_list = []
                soft_gate_list = []
                for _ in range(self.args.num_samples):
                    hard_gate_ = bernoulli_sampling(gate_inputs)
                    hard_gate_list.append(hard_gate_)
                    soft_gate_list.append(gate_inputs)
                
                hard_gate = torch.stack(hard_gate_list,dim=1) # total_node, num_samples
                soft_gate = torch.stack(soft_gate_list,dim=1) # total_node, num_samples
            else:
                hard_gate = gate_inputs > self.args.gate_threshold
                hard_gate_ = gate_inputs > self.args.gate_threshold
                soft_gate = gate_inputs

        else: 
            if self.selected_model_type in ['STE', 'STGS']:
                hard_gate = STEFunction.apply(
                    test_gate_inputs, self.args.gate_threshold 
                )
                hard_gate_= hard_gate
                soft_gate = test_gate_inputs
            else:
                hard_gate_ = test_gate_inputs > self.args.gate_threshold
                hard_gate_list = []
                soft_gate_list = []
                for _ in range(self.args.num_samples):
                    hard_gate_list.append(hard_gate_)
                    soft_gate_list.append(test_gate_inputs)
                
                hard_gate = torch.stack(hard_gate_list,dim=1) # total_node, num_samples
                soft_gate = torch.stack(soft_gate_list,dim=1) # total_node, num_samples



        b, s, h = original_embeddings.shape

        token_start_idx = find_first_one_argmax(encoded['attention_mask']) 
        token_end_idx = find_last_one_argmax(encoded['attention_mask']) 

        token_to_node_mapping = create_token_to_node_mapping(hard_gate_, data_ptr_clean, meta_data, batch_size, seq_length, token_start_idx, token_end_idx, self.args.device) 
        # # total_nodes, batch_size, seq_length
        token_to_node_mapping = token_to_node_mapping.unsqueeze(1)

        # for train thresholding
        if train_threshold:
            hard_gate_ = gate_inputs > self.args.gate_threshold
            hard_gate_list_th = []
            soft_gate_list_th = []
            for _ in range(self.args.num_samples):
                hard_gate_list_th.append(hard_gate_)
                soft_gate_list_th.append(gate_inputs)
            
            hard_gate_th = torch.stack(hard_gate_list_th,dim=1) # total_node, num_samples
            soft_gate_th = torch.stack(soft_gate_list_th,dim=1) # total_node, num_samples


            hard_gate_th = hard_gate_th.view(hard_gate.shape[0], -1, 1, 1)  # [total_nodes, num_samples , 1, 1]
            soft_gate_th = soft_gate_th.view(soft_gate.shape[0], -1, 1, 1)

            
            token_hard_gate_th = (token_to_node_mapping * hard_gate_th).sum(dim=0).view(b * self.args.num_samples, -1)
            token_soft_gate_th = (token_to_node_mapping * soft_gate_th).sum(dim=0).view(b * self.args.num_samples, -1)
            
            modified_attention_mask = token_hard_gate_th*(encoded['attention_mask'].repeat(self.args.num_samples,1)) # [batch_size*num_samples, seq_length]

            if self.args.target_model in ['deberta', 'xlnet']:
                outputs = self.predictor(
                    inputs_embeds=original_embeddings.repeat(self.args.num_samples,1, 1),
                    # inputs_embeds=gated_embeddings,
                    # inputs_embeds=mask_embeddings,
                    token_type_ids=encoded['token_type_ids'].repeat(self.args.num_samples,1),
                    # attention_mask=encoded['attention_mask'],
                    attention_mask=modified_attention_mask.long(),
                )
            elif self.args.target_model in ['roberta', 'gpt2', 'BioMedLM', 'biolinkBert']:
                outputs = self.predictor(
                    # inputs_embeds=gated_embeddings,
                    inputs_embeds=original_embeddings.repeat(self.args.num_samples,1, 1),
                    # attention_mask=encoded['attention_mask'].repeat(self.args.num_samples,1),
                    attention_mask=modified_attention_mask.long(),
                )
            total_token = encoded['attention_mask'].repeat(self.args.num_samples,1).sum()
            
            if self.args.mix_up_rate != 0.:
                out={'outputs': outputs, 'regularizer': regularizer, 'loss_smo': loss_smo, 'token_soft_gate': token_soft_gate_th, 'token_hard_gate': token_hard_gate_th, 'total_token': total_token, 'mixup_reg': selector_out['mixup_reg']}
            else:
                out={'outputs': outputs, 'regularizer': regularizer, 'loss_smo': loss_smo, 'token_soft_gate': token_soft_gate_th, 'token_hard_gate': token_hard_gate_th, 'total_token': total_token}
            return out

        hard_gate = hard_gate.view(hard_gate.shape[0], -1, 1, 1)  # [total_nodes, num_samples , 1, 1]
        soft_gate = soft_gate.view(soft_gate.shape[0], -1, 1, 1)

        token_hard_gate = (token_to_node_mapping * hard_gate).sum(dim=0).view(b * self.args.num_samples, -1)
        token_soft_gate = (token_to_node_mapping * soft_gate).sum(dim=0).view(b * self.args.num_samples, -1)
        

        if self.args.modified_method == 'word':

            input_ids = encoded['input_ids'].repeat(self.args.num_samples, 1)  # [batch_size * num_samples, seq_length]
            special_token_ids_tensor = torch.tensor(self.special_token_ids, device=input_ids.device)
            is_special_token = (input_ids.unsqueeze(-1) == special_token_ids_tensor).any(dim=-1)

            if self.args.data_name == 'cose':
                sep_token_id = self.tokenizer.sep_token_id
                token_hard_gate = self.force_gate_mask_up_to_sep(token_hard_gate, input_ids, sep_token_id)
                
            elif self.args.data_name == 'bioasq':
                if self.args.tokenizer_type == 'BioMedLM':
                    question_token_id = self.tokenizer.question_token_id
                    token_hard_gate = self.force_gate_mask_up_to_question(token_hard_gate, input_ids, question_token_id)
                elif self.args.tokenizer_type == 'biolinkBert':
                    sep_token_id = self.tokenizer.sep_token_id
                    token_hard_gate = self.force_gate_mask_up_to_sep(token_hard_gate, input_ids, sep_token_id)

            preserve_mask = (token_hard_gate > 0) | is_special_token 

            
            if self.args.tokenizer_type in ['deberta', 'deberta_large', 'deberta_small']:
                if self.args.replace_token == 'mask':
                    replace_token_id= 128000
                elif self.args.replace_token == 'blank':
                    replace_token_id = 507
                elif self.args.replace_token == 'the':
                    replace_token_id = 724
                elif self.args.replace_token == 'unk':
                    replace_token_id = 3
                elif self.args.replace_token == '_':
                    replace_token_id = 616
                elif self.args.replace_token == ',':
                    replace_token_id = 366
                elif self.args.replace_token == 'pad':
                    replace_token_id = 0
                    
                masked_input_ids = torch.where(preserve_mask> 0, input_ids, replace_token_id)

            elif self.args.tokenizer_type in ['roberta']:
                
                if self.args.replace_token == 'pad':
                    replace_token_id = 1
                elif self.args.replace_token == 'unk':
                    replace_token_id = 3
                elif self.args.replace_token == 'the':
                    replace_token_id = 627
                elif self.args.replace_token == '_':
                    replace_token_id = 616
                elif self.args.replace_token == ',':
                    replace_token_id = 6
                if self.args.replace_token == 'blank':
                    masked_input_ids = drop_and_pad_tokens_gpu(
                        input_ids=input_ids,
                        preserve_mask=preserve_mask,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                else:
                    masked_input_ids = torch.where(preserve_mask > 0, input_ids, replace_token_id)
                

            elif self.args.tokenizer_type in ['bert']:
                if self.args.replace_token == 'the':
                    replace_token_id = 1996
                elif self.args.replace_token == 'pad':
                    replace_token_id = 0
                elif self.args.replace_token == 'unk':
                    replace_token_id = 100
                elif self.args.replace_token == '_':
                    replace_token_id = 1035
                elif self.args.replace_token == ',':
                    replace_token_id = 1010

                if self.args.replace_token == 'blank':
                    masked_input_ids = drop_and_pad_tokens_gpu(
                        input_ids=input_ids,
                        preserve_mask=preserve_mask,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                else:
                    masked_input_ids = torch.where(preserve_mask > 0, input_ids, replace_token_id)
                
            elif self.args.tokenizer_type in ['gpt2', 'BioMedLM']:
                if self.args.replace_token == 'the':
                    replace_token_id = 3785
                elif self.args.replace_token == '_':
                    replace_token_id = 62
                elif self.args.replace_token == ',':
                    replace_token_id = 11
                if self.args.replace_token == 'blank':
                    masked_input_ids = drop_and_pad_tokens_gpu(
                        input_ids=input_ids,
                        preserve_mask=preserve_mask,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                else:
                    masked_input_ids = torch.where(preserve_mask > 0, input_ids, replace_token_id)

            elif self.args.tokenizer_type in ['biolinkBert']:
                if self.args.replace_token == 'the':
                    replace_token_id = 1680
                elif self.args.replace_token == '_':
                    replace_token_id = 40
                elif self.args.replace_token == ',':
                    replace_token_id = 15
                elif self.args.replace_token == 'unk':
                    replace_token_id = 1

                if self.args.replace_token == 'blank':
                    masked_input_ids = drop_and_pad_tokens_gpu(
                        input_ids=input_ids,
                        preserve_mask=preserve_mask,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                else:
                    masked_input_ids = torch.where(preserve_mask > 0, input_ids, replace_token_id)
                    
            masked_text_list = self.tokenizer.batch_decode(masked_input_ids, skip_special_tokens=True)

            new_encoded = self.predictor_tokenizer(
                masked_text_list,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.args.device)
        

            if self.args.target_model in ['deberta', 'xlnet']:
                outputs = self.predictor(
                    input_ids=new_encoded['input_ids'],
                    attention_mask=new_encoded['attention_mask'],
                    token_type_ids=new_encoded['token_type_ids']
                )
            elif self.args.target_model in ['roberta', 'gpt2', 'BioMedLM', 'biolinkBert']:
                outputs = self.predictor(
                    input_ids=new_encoded['input_ids'],
                    attention_mask=new_encoded['attention_mask']
                ) 

        elif self.args.modified_method == 'attention_mask':
            modified_attention_mask = token_hard_gate*(encoded['attention_mask'].repeat(self.args.num_samples,1)) # [batch_size*num_samples, seq_length]

            modified_attention_mask = apply_special_token_mask(
                modified_attention_mask,
                encoded['input_ids'],
                self.special_token_ids,
                self.args.num_samples
            )
            if self.args.data_name == 'cose':

                modified_attention_mask = self.force_mask_up_to_sep(
                    modified_attention_mask,
                    encoded['input_ids'],
                    self.tokenizer.sep_token_id,
                    self.args.num_samples
                )
            elif self.args.data_name == 'bioasq':
                question_token_id = self.tokenizer.convert_tokens_to_ids("[QUESTION]")
                token_hard_gate = self.force_mask_from_question(token_hard_gate, input_ids, question_token_id, encoded['attention_mask'].repeat(self.args.num_samples,1))


            if self.args.tokenizer_type in ['deberta', 'xlnet']:
                outputs = self.predictor(
                    inputs_embeds=original_embeddings.repeat(self.args.num_samples,1, 1),
                    token_type_ids=encoded['token_type_ids'].repeat(self.args.num_samples,1),
                    attention_mask=modified_attention_mask.long(),
                )
            elif self.args.tokenizer_type in ['roberta', 'gpt2', 'BioMedLM', 'biolinkBert']:
                outputs = self.predictor(
                    inputs_embeds=original_embeddings.repeat(self.args.num_samples,1, 1),
                    attention_mask=modified_attention_mask.long(),
                )
    
        elif self.args.modified_method == 'embedding':
            modified_attention_mask = token_hard_gate*(encoded['attention_mask'].repeat(self.args.num_samples,1)) # [batch_size*num_samples, seq_length]
            input_ids = encoded['input_ids'].repeat(self.args.num_samples, 1)

            if test:
                gate = token_hard_gate
            else:
                if self.args.selected_model_type in ['STE', 'STGS', 'STE_S']:
                    gate = token_hard_gate
                else:
                    gate = token_soft_gate
            
            total_tokens_num = encoded['attention_mask'].sum().item()
            pad_id = self.tokenizer.pad_token_id
            special_ids = set(self.tokenizer.all_special_ids)
            special_ids.discard(pad_id)

            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for sid in special_ids:
                special_token_mask |= (input_ids == sid)
            if self.args.data_name == 'cose':

                modified_attention_mask = self.force_mask_up_to_sep(
                    modified_attention_mask,
                    encoded['input_ids'],
                    self.tokenizer.sep_token_id,
                    self.args.num_samples
                )
            elif self.args.data_name == 'bioasq':
                question_token_id = self.tokenizer.convert_tokens_to_ids("[QUESTION]")
                gate = self.force_mask_from_question(gate, input_ids, question_token_id, encoded['attention_mask'].repeat(self.args.num_samples,1))

            gate[special_token_mask] = 1.0
            
            b, s, h = original_embeddings.shape
            if self.mask_embedding != None:
                mask_embeddings = self.mask_embedding.expand(b, s, -1).to(self.args.device)
            else:
                mask_embeddings = torch.zeros_like(original_embeddings).to(self.args.device)
            # pdb.set_trace()
            gated_embeddings = (original_embeddings.repeat(self.args.num_samples,1, 1) * gate.unsqueeze(-1) + mask_embeddings.repeat(self.args.num_samples,1, 1) * (1 - gate).unsqueeze(-1))

            if self.args.target_model in ['deberta', 'xlnet', 'deberta_large', 'deberta_small']:
                outputs = self.predictor(
                    inputs_embeds=gated_embeddings,
                    token_type_ids=encoded['token_type_ids'].repeat(self.args.num_samples,1),
                    attention_mask=encoded['attention_mask'].repeat(self.args.num_samples,1),
                )
            elif self.args.target_model in ['roberta', 'gpt2', 'BioMedLM', 'biolinkBert']:
                outputs = self.predictor(
                    inputs_embeds=gated_embeddings,
                    attention_mask=encoded['attention_mask'].repeat(self.args.num_samples,1),
                )

        total_token = encoded['attention_mask'].repeat(self.args.num_samples,1).sum()
        
        ######################## words prob ########################
        score_matrices_dict = create_score_matrices_dict(selector_out['score_matrices'], len(data_ptr_clean)-1, meta_data)
        ####################################################################
        if test:

            ######################## masked_tokens check ########################

            if self.args.modified_method == 'word':
                masked_input_ids = new_encoded['input_ids']
                valid_token_counts = encoded['attention_mask'].sum(dim=1)

                filtered_tokens = {
                    sample_id.item(): masked_input_ids[batch_idx, :valid_token_counts[batch_idx].item()].tolist()
                    for batch_idx, sample_id in enumerate(meta_data['idx'])
                }
            elif self.args.modified_method == 'attention_mask':
                valid_token_counts = encoded['attention_mask'].sum(dim=1)
                filtered_tokens = create_masked_token_dict(encoded['input_ids'], modified_attention_mask, valid_token_counts, meta_data['idx'])
                
            elif self.args.modified_method == 'embedding':
                valid_token_counts = encoded['attention_mask'].sum(dim=1)
                filtered_tokens = create_masked_token_dict(encoded['input_ids'], modified_attention_mask, valid_token_counts, meta_data['idx'])
            ####################################################################
            
            ######################## maskeds_words check ########################
            filtered_words = filter_unmasked_words(hard_gate_, data_ptr_clean, meta_data['raw_word'], meta_data['idx'], mask=True) # mask=True: masking된 단어가 [MASK]로 replace / mask=False: masking되지 않은 단어만 필터링
            ####################################################################
            ######################## words prob ########################
            sample_prob_dict = create_sample_dict(gate_inputs, data_ptr_clean, meta_data)
            ####################################################################
            ######################## words prob ########################
            split_matrix = selector_out['split_matrix']
            grouped_prob_dict = create_grouped_sample_dict(gate_inputs, data_ptr_clean, split_matrix, meta_data)
            ####################################################################
            grouped_adj_dict = group_adj_matrices_by_sample(selector_out['adj_matrix'], meta_data)

            if self.args.mix_up_rate != 0.:
                out={'outputs': outputs, 'regularizer': regularizer, 'loss_smo': loss_smo, 'token_soft_gate': token_soft_gate, \
                    'token_hard_gate': token_hard_gate, 'total_token': total_token, 'filtered_tokens': filtered_tokens, 'filtered_words': filtered_words, 
                    'sample_prob_dict': sample_prob_dict, 'grouped_prob_dict': grouped_prob_dict,
                    'mixup_reg': selector_out['mixup_reg'],
                    'score_matrices': score_matrices_dict,
                    }
            else:
                out={'outputs': outputs, 'regularizer': regularizer, 'loss_smo': loss_smo, 'token_soft_gate': token_soft_gate, \
                    'token_hard_gate': token_hard_gate, 'total_token': total_token, 'filtered_tokens': filtered_tokens, 'filtered_words': filtered_words,
                    'sample_prob_dict': sample_prob_dict, 'grouped_prob_dict': grouped_prob_dict, 
                    'score_matrices': score_matrices_dict, 'adj_matrix' : grouped_adj_dict,
                    }
            return out
        if self.args.mix_up_rate != 0.:
            out={'outputs': outputs, 'regularizer': regularizer, 'loss_smo': loss_smo, 'token_soft_gate': token_soft_gate, \
                'token_hard_gate': token_hard_gate, 'total_token': total_token,
                'mixup_reg': selector_out['mixup_reg'], 
                'score_matrices': score_matrices_dict,
                }
        else:
            out={'outputs': outputs, 'regularizer': regularizer, 'loss_smo': loss_smo, 'token_soft_gate': token_soft_gate, \
                'token_hard_gate': token_hard_gate, 'total_token': total_token, 
                'score_matrices': score_matrices_dict,
                }
        return out


    def set_special_tokens_gate(self, token_gates, input_ids, special_token_ids):

        special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in special_token_ids:
            special_token_mask |= (input_ids == token_id)
        
        token_gates[special_token_mask] = 1.0
        return token_gates

    def get_token_gradients(self):
        if (self.embedding_output is not None and 
            hasattr(self.embedding_output, 'grad') and 
            self.embedding_output.grad is not None):
            return self.embedding_output.grad
        return None

    def force_mask_up_to_sep(self, modified_mask, input_ids, sep_token_id, num_samples):

        B, L = input_ids.shape
        forced_mask = torch.zeros_like(modified_mask)

        for i in range(B):
            sep_idx = (input_ids[i] == sep_token_id).nonzero(as_tuple=False)
            if len(sep_idx) > 0:
                sep_pos = sep_idx[0].item() 
                forced_mask[i, :sep_pos + 1] = 1
                forced_mask[i, sep_pos + 1:] = modified_mask[i, sep_pos + 1:] 
            else:
                forced_mask[i] = modified_mask[i] 

        return forced_mask

    def force_mask_from_question(self, modified_mask, input_ids, question_token_id, attention_mask):

        B, L = input_ids.shape
        forced_mask = modified_mask.clone()

        for i in range(B):
            q_idx = (input_ids[i] == question_token_id).nonzero(as_tuple=False)
            if len(q_idx) > 0:
                q_pos = q_idx[0].item()

                seq_end = attention_mask[i].sum().item()

                forced_mask[i, q_pos:seq_end] = 1  
            else:
                forced_mask[i] = modified_mask[i]  # fallback

        return forced_mask

    def force_gate_mask_up_to_sep(self, token_hard_gate, input_ids, sep_token_id):

        B, L = input_ids.shape
        forced_gate = token_hard_gate.clone()

        for i in range(B):
            sep_idx = (input_ids[i] == sep_token_id).nonzero(as_tuple=False)
            if len(sep_idx) > 0:
                sep_pos = sep_idx[0].item() 
                forced_gate[i, :sep_pos + 1] = 1  
        return forced_gate

    def force_gate_mask_up_to_question(self, token_hard_gate, input_ids, question_token_id):

        B, L = input_ids.shape
        forced_gate = token_hard_gate.clone()

        for i in range(B):
            question_idx = (input_ids[i] == question_token_id).nonzero(as_tuple=False)
            if len(question_idx) > 0:
                q_pos = question_idx[0].item() 
                forced_gate[i, q_pos:] = 1 

        return forced_gate

    def fix_test(self, data, meta_data, fix_mask, baseline = False):

        texts = meta_data['text']

        if baseline:
            if self.args.target_model == 'BioMedLM':
                context_prompt = texts['context']
                question_prompt = texts['question']
                # context_prompt = f"[CONTEXT] {context} [QUESTION]"
                # question_prompt = f"{question} [ANSWER]"

                baseline_encoded = self.predictor_tokenizer(
                    f"[CONTEXT] {context_prompt}",
                    text_pair=f"[QUESTION] {question_prompt} [ANSWER]",
                    padding=True,
                    truncation="only_first", 
                    max_length=512,
                    return_tensors="pt"
                ).to(self.args.device)

                outputs = self.predictor(
                    input_ids=baseline_encoded['input_ids'],
                    attention_mask=baseline_encoded['attention_mask']
                )
                return outputs

            else:
                baseline_encoded = self.predictor_tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=512 ,
                        return_tensors='pt'
                        ).to(self.args.device)
            
                if self.args.target_model in ['deberta', 'xlnet']:
                    outputs = self.predictor(
                        input_ids=baseline_encoded['input_ids'],
                        attention_mask=baseline_encoded['attention_mask'],
                        token_type_ids=baseline_encoded['token_type_ids']
                    )
                elif self.args.target_model in ['roberta', 'gpt2', 'BioMedLM', 'biolinkBert']:
                    outputs = self.predictor(
                        input_ids=baseline_encoded['input_ids'],
                        attention_mask=baseline_encoded['attention_mask']
                    )
                return outputs

        if self.args.target_model == 'BioMedLM':
            context_prompt = texts['context']
            question_prompt = texts['question']
            # context_prompt = f"[CONTEXT] {context} [QUESTION]"
            # question_prompt = f"{question} [ANSWER]"

            encoded = self.tokenizer(
                f"[CONTEXT] {context_prompt}",
                text_pair=f"[QUESTION] {question_prompt} [ANSWER]",
                padding=True,
                truncation="only_first", 
                max_length=512,
                return_tensors="pt"
            ).to(self.args.device)
        else:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.args.device)
        
        safe_input_ids = encoded['input_ids'].detach().cpu()
        original_embeddings = self.word_embedding(encoded['input_ids']).to(self.args.device) 
        
        
        selector_out = self.selected_model(data, model_type=self.selected_model_type)

        data_ptr_clean = selector_out['data_ptr_clean']

        batch_size = encoded['attention_mask'].size(0)
        seq_length = encoded['attention_mask'].size(1)

        
        b, s, h = original_embeddings.shape
        hard_gate = fix_mask
        
        token_start_idx = find_first_one_argmax(encoded['attention_mask'])
        token_end_idx = find_last_one_argmax(encoded['attention_mask']) 
        
        
        token_to_node_mapping = create_token_to_node_mapping(hard_gate, data_ptr_clean, meta_data, batch_size, seq_length, token_start_idx, token_end_idx, self.args.device) 

        token_to_node_mapping = token_to_node_mapping.unsqueeze(1)

        hard_gate = hard_gate.view(hard_gate.shape[0], -1, 1, 1)  # [total_nodes, num_samples , 1, 1]
        
        
        token_hard_gate = (token_to_node_mapping * hard_gate).sum(dim=0).view(b, -1)

        if self.args.modified_method == 'word':
            input_ids = encoded['input_ids'].repeat(self.args.num_samples, 1)  # [batch_size * num_samples, seq_length]
            special_token_ids_tensor = torch.tensor(self.special_token_ids, device=input_ids.device)
            is_special_token = (input_ids.unsqueeze(-1) == special_token_ids_tensor).any(dim=-1)

            if self.args.data_name == 'cose':
                sep_token_id = self.tokenizer.sep_token_id
                token_hard_gate = self.force_gate_mask_up_to_sep(token_hard_gate, input_ids, sep_token_id)
                
            elif self.args.data_name == 'bioasq':
                if self.args.tokenizer_type == 'BioMedLM':
                    question_token_id = self.tokenizer.question_token_id
                    token_hard_gate = self.force_gate_mask_up_to_question(token_hard_gate, input_ids, question_token_id)
                elif self.args.tokenizer_type == 'biolinkBert':
                    sep_token_id = self.tokenizer.sep_token_id
                    token_hard_gate = self.force_gate_mask_up_to_sep(token_hard_gate, input_ids, sep_token_id)

            preserve_mask = (token_hard_gate > 0) | is_special_token 
            
            if self.args.tokenizer_type in ['deberta', 'deberta_large', 'deberta_small']:
                if self.args.replace_token == 'mask':
                    replace_token_id= 128000
                elif self.args.replace_token == 'blank':
                    replace_token_id = 507
                elif self.args.replace_token == 'the':
                    replace_token_id = 724
                elif self.args.replace_token == 'unk':
                    replace_token_id = 3
                elif self.args.replace_token == '_':
                    replace_token_id = 616
                elif self.args.replace_token == ',':
                    replace_token_id = 366
                masked_input_ids = torch.where(preserve_mask> 0, input_ids, replace_token_id)
                
            elif self.args.tokenizer_type in ['roberta']:
                if self.args.replace_token == 'blank':
                    replace_token_id = 1437
                elif self.args.replace_token == 'pad':
                    replace_token_id = 1
                elif self.args.replace_token == 'unk':
                    replace_token_id = 3
                elif self.args.replace_token == '_':
                    replace_token_id = 616
                elif self.args.replace_token == ',':
                    replace_token_id = 6
                masked_input_ids = torch.where(preserve_mask> 0, input_ids, replace_token_id) 

            elif self.args.tokenizer_type in ['bert']:
                if self.args.replace_token == 'the':
                    replace_token_id = 1996
                elif self.args.replace_token == 'pad':
                    replace_token_id = 0
                elif self.args.replace_token == 'unk':
                    replace_token_id = 100
                elif self.args.replace_token == '_':
                    replace_token_id = 1035
                elif self.args.replace_token == ',':
                    replace_token_id = 1010

                if self.args.replace_token == 'blank':
                    masked_input_ids = drop_and_pad_tokens_gpu(
                        input_ids=input_ids,
                        preserve_mask=preserve_mask,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                else:
                    masked_input_ids = torch.where(preserve_mask > 0, input_ids, replace_token_id)
                
            elif self.args.tokenizer_type in ['gpt2', 'BioMedLM']:
                if self.args.replace_token == 'blank':
                    masked_input_ids = drop_and_pad_tokens_gpu(
                        input_ids=input_ids,
                        preserve_mask=preserve_mask,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                else:
                    if self.args.replace_token == 'the':
                        replace_token_id = 3785
                    elif self.args.replace_token == '_':
                        replace_token_id = 62
                    elif self.args.replace_token == ',':
                        replace_token_id = 11
                    masked_input_ids = torch.where(preserve_mask > 0, input_ids, replace_token_id)
            elif self.args.tokenizer_type in ['biolinkBert']:
                if self.args.replace_token == 'the':
                    replace_token_id = 1680
                elif self.args.replace_token == '_':
                    replace_token_id = 40
                elif self.args.replace_token == ',':
                    replace_token_id = 15
                elif self.args.replace_token == 'unk':
                    replace_token_id = 1

                if self.args.replace_token == 'blank':
                    masked_input_ids = drop_and_pad_tokens_gpu(
                        input_ids=input_ids,
                        preserve_mask=preserve_mask,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                else:
                    masked_input_ids = torch.where(preserve_mask > 0, input_ids, replace_token_id)
            masked_text_list = self.tokenizer.batch_decode(masked_input_ids, skip_special_tokens=True)

            new_encoded = self.predictor_tokenizer(
                masked_text_list,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.args.device)
        

            # predictor forward
            if self.args.target_model in ['deberta', 'xlnet']:
                outputs = self.predictor(
                    input_ids=new_encoded['input_ids'],
                    attention_mask=new_encoded['attention_mask'],
                    token_type_ids=new_encoded['token_type_ids']
                )
            elif self.args.target_model in ['roberta', 'gpt2', 'BioMedLM', 'biolinkBert']:
                outputs = self.predictor(
                    input_ids=new_encoded['input_ids'],
                    attention_mask=new_encoded['attention_mask']
                ) 

        elif self.args.modified_method == 'attention_mask':
            modified_attention_mask = token_hard_gate*(encoded['attention_mask'])
            modified_attention_mask = apply_special_token_mask(
                modified_attention_mask,
                encoded['input_ids'],
                self.special_token_ids,
                self.args.num_samples
            )
            if self.args.data_name == 'cose':

                modified_attention_mask = self.force_mask_up_to_sep(
                    modified_attention_mask,
                    encoded['input_ids'],
                    self.tokenizer.sep_token_id,
                    self.args.num_samples
                )
            elif self.args.data_name == 'bioasq':
                question_token_id = self.tokenizer.convert_tokens_to_ids("[QUESTION]")
                token_hard_gate = self.force_mask_from_question(token_hard_gate, input_ids, question_token_id, encoded['attention_mask'])

            if self.args.target_model in ['deberta', 'xlnet']:
                outputs = self.predictor(
                    inputs_embeds=original_embeddings,
                    token_type_ids=encoded['token_type_ids'],
                    attention_mask=modified_attention_mask.long(),
                )
            elif self.args.target_model in ['roberta', 'gpt2', 'BioMedLM', 'biolinkBert']:
                outputs = self.predictor(
                    inputs_embeds=original_embeddings,
                    attention_mask=modified_attention_mask.long(),
                )

        elif self.args.modified_method == 'embedding':
            modified_attention_mask = token_hard_gate*(encoded['attention_mask'].repeat(self.args.num_samples,1)) # [batch_size*num_samples, seq_length]
            input_ids = encoded['input_ids'].repeat(self.args.num_samples, 1)


            gate = token_hard_gate
            total_tokens_num = encoded['attention_mask'].sum().item()
            pad_id = self.tokenizer.pad_token_id
            special_ids = set(self.tokenizer.all_special_ids)
            special_ids.discard(pad_id)

            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for sid in special_ids:
                special_token_mask |= (input_ids == sid)
            if self.args.data_name == 'cose':

                modified_attention_mask = self.force_mask_up_to_sep(
                    modified_attention_mask,
                    encoded['input_ids'],
                    self.tokenizer.sep_token_id,
                    self.args.num_samples
                )
            elif self.args.data_name == 'bioasq':
                question_token_id = self.tokenizer.convert_tokens_to_ids("[QUESTION]")
                gate = self.force_mask_from_question(gate, input_ids, question_token_id, encoded['attention_mask'].repeat(self.args.num_samples,1))

            gate[special_token_mask] = 1.0
            
            b, s, h = original_embeddings.shape
            if self.mask_embedding != None:
                mask_embeddings = self.mask_embedding.expand(b, s, -1).to(self.args.device)
            else:
                mask_embeddings = torch.zeros_like(original_embeddings).to(self.args.device)
            # pdb.set_trace()
            gated_embeddings = (original_embeddings.repeat(self.args.num_samples,1, 1) * gate.unsqueeze(-1) + mask_embeddings.repeat(self.args.num_samples,1, 1) * (1 - gate).unsqueeze(-1))

            if self.args.target_model in ['deberta', 'xlnet']:
                outputs = self.predictor(
                    inputs_embeds=gated_embeddings,
                    token_type_ids=encoded['token_type_ids'].repeat(self.args.num_samples,1),
                    attention_mask=encoded['attention_mask'].repeat(self.args.num_samples,1),
                )
            elif self.args.target_model in ['roberta', 'gpt2', 'BioMedLM', 'biolinkBert']:
                outputs = self.predictor(
                    inputs_embeds=gated_embeddings,
                    attention_mask=encoded['attention_mask'].repeat(self.args.num_samples,1),
                )

        total_token = encoded['attention_mask'].sum()
        
        if self.args.modified_method == 'word':
            masked_input_ids = new_encoded['input_ids']
            valid_token_counts = encoded['attention_mask'].sum(dim=1)

            filtered_tokens = {
                sample_id.item(): masked_input_ids[batch_idx, :valid_token_counts[batch_idx].item()].tolist()
                for batch_idx, sample_id in enumerate(meta_data['idx'])
            }
        elif self.args.modified_method == 'attention_mask':
            valid_token_counts = encoded['attention_mask'].sum(dim=1)
            filtered_tokens = create_masked_token_dict(encoded['input_ids'], modified_attention_mask, valid_token_counts, meta_data['idx'])
            
        elif self.args.modified_method == 'embedding':
            valid_token_counts = encoded['attention_mask'].sum(dim=1)
            filtered_tokens = create_masked_token_dict(encoded['input_ids'], modified_attention_mask, valid_token_counts, meta_data['idx'])

        filtered_words = filter_unmasked_words(hard_gate, data_ptr_clean, meta_data['raw_word'], meta_data['idx'], mask=True) # mask=True: masking된 단어가 [MASK]로 replace / mask=False: masking되지 않은 단어만 필터링

        grouped_adj_dict = group_adj_matrices_by_sample(selector_out['adj_matrix'], meta_data)
        

        out={'outputs': outputs, \
            'token_hard_gate': token_hard_gate, 'total_token': total_token, 'filtered_tokens': filtered_tokens, 'filtered_words': filtered_words, \
            'adj_matrix' : grouped_adj_dict,
            }
        return out

    def mask_with_random_replacement(self, input_ids, token_hard_gate, candidates=["the", "_", ","]):

        candidate_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tok) for tok in candidates], device=self.args.device)

        mask = token_hard_gate <= 0  # shape: [batch_size, seq_len]

        random_ids = candidate_ids[torch.randint(0, len(candidate_ids), mask.shape, device=self.args.device)]

        masked_input_ids = torch.where(mask, random_ids, input_ids)

        return masked_input_ids




class GatherModel(nn.Module):
    """
    MPNN from
    `Neural Message Passing for Quantum Chemistry <https://arxiv.org/abs/1704.01212>`
    Parameters
    ----------
    node_input_dim : int
        Dimension of input node feature, default to be 42.
    edge_input_dim : int
        Dimension of input edge feature, default to be 10.
    node_hidden_dim : int
        Dimension of node feature in hidden layers, default to be 42.
    edge_hidden_dim : int
        Dimension of edge feature in hidden layers, default to be 128.
    num_step_message_passing : int
        Number of message passing steps, default to be 6.
    """

    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_step_message_passing=3,
                 dropout = 0.0,
                 ):
        super(GatherModel, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.set2set = Set2Set(node_hidden_dim, processing_steps=2, num_layers=1)
        self.message_layer = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim), nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
        self.conv = NNConv(in_channels=node_hidden_dim,
                           out_channels=node_hidden_dim,
                           nn=edge_network,
                           aggr='add',
                           root_weight=True
                           )
        self.dropout = dropout

    def forward(self, g):
        """Returns the node embeddings after message passing phase.
        Parameters
        ----------
        g : Torch geometric batch data
            Input batch data for molecule(s)
        Returns
        -------
        res : node features
        """
        init = g.x.clone()
        out = F.relu(self.lin0(g.x))
        for i in range(self.num_step_message_passing):
            if len(g.edge_attr) != 0:
                m = torch.relu(self.conv(out, g.edge_index, g.edge_attr))
            else:
                m = torch.relu(self.conv.bias + out)
            out = self.message_layer(torch.cat([m, out], dim=1))
        return out + init

class CustomBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(CustomBatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        
        # input: [batch_size, num_features]
        mask = ~torch.isnan(input)
        masked_input = torch.where(mask, input, torch.zeros_like(input))
        
        # Compute mean and variance excluding NaNs
        mean = masked_input.sum(dim=0) / mask.sum(dim=0).clamp(min=1)
        variance = ((masked_input - mean) ** 2).sum(dim=0) / mask.sum(dim=0).clamp(min=1)
        
        # Normalize
        normalized_input = (masked_input - mean) / torch.sqrt(variance + self.bn.eps)
        
        # Apply scale and shift
        if self.bn.affine:
            normalized_input = normalized_input * self.bn.weight + self.bn.bias
        
        # Restore NaNs
        normalized_input = torch.where(mask, normalized_input, torch.full_like(normalized_input, float('nan')))
        
        return normalized_input
    
class Our_Selector_V1(nn.Module):

    def __init__(self,
                args,
                device,
                node_hidden_dim=768,
                ):
        super(Our_Selector_V1, self).__init__()

        self.args = args
        self.device = device
        self.node_hidden_dim = node_hidden_dim        

        # Attention weight 계산
        self.weight_attetion = SelfAttentionWeightedAdjacency(self.node_hidden_dim)
        self.similarity_matrix = CosineWeightedAdjacency()
        # 확률 생성하는 함수
        self.compressor = nn.Sequential(
            nn.Linear(self.node_hidden_dim, self.node_hidden_dim),
            CustomBatchNorm1d(self.node_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.node_hidden_dim, 1)
            ).to(device)
        
        self.mse_loss = torch.nn.MSELoss()
        self.init_model()


    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def mask_adjust_graph(self, node_features, data_ptr, edge_index): 

        mask_ = ~torch.isnan(node_features).any(dim=1)


        last_true_indices = self.find_last_true_indices(mask_, data_ptr)

        mask = torch.zeros_like(mask_, dtype=torch.bool)
        for i, last_idx in enumerate(last_true_indices):
            if last_idx >= 0:
                mask[data_ptr[i]:min(last_idx + 1, data_ptr[i + 1].item())] = True  
        
        node_features_clean = node_features[mask]

        nan_mask = torch.isnan(node_features_clean)
        if nan_mask.any():
            mask_embedding = self.args.enc_mask_embedding.expand_as(node_features_clean).to(node_features_clean.device)
            node_features_clean = torch.where(nan_mask, mask_embedding, node_features_clean)
        ########################################################
        data_ptr_clean = self.adjust_data_ptr(data_ptr, mask) 
        edge_index_clean = self.adjust_edge_index(edge_index, mask)

        return node_features_clean, data_ptr_clean, edge_index_clean
    
    def compress(self, node_features_clean, model_type = 'STE'):
        p = self.compressor(node_features_clean)
        
        if model_type == 'concrete':
            temperature = 1.0
            bias = 0.0 + 0.0001
            eps = bias + (1 - 2 * bias) * torch.rand(p.size())
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(self.device)
            gate_inputs = (gate_inputs + p) / temperature
            gate_inputs = torch.sigmoid(gate_inputs).squeeze()

            regularizer = torch.sigmoid(p)

            test_gate_inputs = torch.sigmoid(p)

        elif model_type == 'hard_concrete':
            limit_a = torch.tensor(-0.1, device=p.device) # Lower limit
            limit_b = torch.tensor(1.1, device=p.device) # Upper limit
            temperature = 0.2
            eps = 1e-6

            u = torch.rand_like(p, requires_grad=True)
            u = eps + u * (1 - 2 * eps)
            s = torch.sigmoid((p + torch.log(u) - torch.log(1-u)) / temperature) 
            s_bar = limit_a + s * (limit_b - limit_a)

            gate_inputs = torch.clamp(s_bar, 0, 1)

            regularizer = torch.sigmoid(p - temperature * torch.log(-limit_a/limit_b))
            
            test_gate_inputs = torch.sigmoid(p) 
            test_gate_inputs = limit_a + test_gate_inputs * (limit_b - limit_a)
            test_gate_inputs = torch.clamp(test_gate_inputs, 0, 1)

        elif model_type == 'stg':
            sigma = 1.0
            
            eps = torch.randn_like(p)
            z = p + sigma * eps
            gate_inputs = torch.clamp(z + 0.5, 0, 1)

            # regularizer
            x = p / sigma
            regularizer = 0.5 * (1 + torch.erf(x /torch.sqrt(torch.tensor(2.0, device=p.device))))
            test_gate_inputs = torch.clamp(p + 0.5, 0, 1)

        elif model_type == 'gumbel_soft':
            temperature = 0.2
            k = self.args.k  
            eps = 1e-8

            B, T = p.size()  # p: (B, T), logits

            # Gumbel noise
            uniform_noise = torch.rand(B, k, T).to(p.device)
            gumbel_noise = -torch.log(-torch.log(uniform_noise + eps) + eps)

            # Expand logits for k samples
            p_exp = p.unsqueeze(1).expand(-1, k, -1)  # (B, k, T)

            # Add noise and temperature scale
            noisy_logits = (p_exp + gumbel_noise) / temperature  # (B, k, T)
            soft_samples = F.softmax(noisy_logits, dim=-1)       # (B, k, T)

            # Aggregate into one vector via max over k samples
            gate_inputs = torch.max(soft_samples, dim=1).values  # (B, T)

            # Regularizer (optional): encourage sparsity on logits
            regularizer = torch.sigmoid(p)  # or use entropy of gate_inputs

            # Hard gate at test time (if needed)
            if not self.training:
                # Top-k thresholding
                threshold = torch.topk(p, k, dim=-1, largest=True, sorted=True).values[:, -1].unsqueeze(1)  # (B, 1)
                test_gate_inputs = (p > threshold).float()
            else:
                test_gate_inputs = gate_inputs  # during training, keep soft gate

        elif model_type in ['STE', 'STE_S']:
            
            gate_inputs = torch.sigmoid(p)
            test_gate_inputs = gate_inputs
            regularizer = gate_inputs

        elif model_type == 'STGS':
            
            temperature = 0.2  

            gumbel_noise = sample_gumbel(p.shape, device=p.device)
            gumbel_p = (p + gumbel_noise) / temperature

            gate_inputs = torch.sigmoid(gumbel_p)
            test_gate_inputs = torch.sigmoid(p)
            regularizer = torch.sigmoid(p)

        elif model_type == 'RL':
            gate_inputs = torch.sigmoid(p)
            test_gate_inputs = gate_inputs
            regularizer = gate_inputs

        return gate_inputs, test_gate_inputs, regularizer
    
    def adjust_edge_index(self, edge_index, mask):

        num_nodes = mask.size(0)
        if edge_index.numel() == 0:
            valid_node_indices = torch.nonzero(mask, as_tuple=False).view(-1)
            num_valid = valid_node_indices.size(0)

            if num_valid <= 1:
                return torch.empty((2, 0), dtype=torch.long, device=mask.device)

            row = valid_node_indices.repeat_interleave(num_valid)
            col = valid_node_indices.repeat(num_valid)
            mask_no_self = row != col
            edge_index = torch.stack([row[mask_no_self], col[mask_no_self]], dim=0)
        else:
            edge_index = edge_index[:, (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)]

        if edge_index.max() >= num_nodes:
            
            raise ValueError(f"[❗] edge_index includes invalid node indices: max = {edge_index.max().item()}, mask size = {num_nodes}")

        if edge_index.device != mask.device:
            mask = mask.to(edge_index.device)

        valid_edges = mask[edge_index[0]] & mask[edge_index[1]]
        filtered_edge_index = edge_index[:, valid_edges]

        new_indices = torch.zeros_like(mask, dtype=torch.long)
        new_indices[mask] = torch.arange(mask.sum(), device=edge_index.device)

        remapped_edge_index = new_indices[filtered_edge_index]

        return remapped_edge_index


    def adjust_data_ptr(self, data_ptr, mask):
        batch_size = data_ptr.size(0) - 1
        counts = []
        for i in range(batch_size):
            start = data_ptr[i].item()
            end = data_ptr[i + 1].item()
            valid = mask[start:end].sum().item()
            counts.append(valid)
        # 누적합을 통해 data_ptr_clean 생성
        data_ptr_clean = [0]
        for count in counts:
            data_ptr_clean.append(data_ptr_clean[-1] + count)
        return torch.tensor(data_ptr_clean, dtype=torch.long, device=data_ptr.device)

    def find_last_true_indices(self, mask, data_ptr):
        batch_size = data_ptr.size(0) - 1
        last_true_indices = torch.full((batch_size,), -1, dtype=torch.long, device=mask.device)

        for i in range(batch_size):
            start = data_ptr[i].item()
            end = data_ptr[i + 1].item()
            sample_mask = mask[start:end]
            if sample_mask.any():
                last_true_indices[i] = torch.where(sample_mask)[0][-1] + start

        return last_true_indices

    def propagate_mask_probs(node_probs: torch.Tensor, edge_index: torch.Tensor, edge_weights: torch.Tensor, n_hops: int = 1, eps: float = 1e-8) -> torch.Tensor:

        original_shape = node_probs.shape
        if node_probs.dim() == 1:
            node_probs = node_probs.unsqueeze(-1)  # shape [N, 1]

        propagated = node_probs
        src, dst = edge_index  

        for _ in range(n_hops):
            numerator = scatter_add(edge_weights.unsqueeze(-1) * propagated[src], dst, dim=0, dim_size=propagated.size(0))
            denominator = scatter_add(edge_weights, dst, dim=0, dim_size=propagated.size(0)).unsqueeze(-1)
            propagated = numerator / (denominator + eps)
        
        if original_shape == node_probs.shape:
            return propagated
        else:
            return propagated.squeeze(-1)

    def propagate_gate_inputs_list(self, gate_inputs, adjacency_matrix, ptr, num_hops=1, eps=1e-8):
        batch_size = len(adjacency_matrix)
        propagated_gate_inputs = torch.zeros_like(gate_inputs) 

        for batch_idx in range(batch_size):
            start_idx = ptr[batch_idx].item()
            end_idx = ptr[batch_idx + 1].item()

            batch_gate_inputs = gate_inputs[start_idx:end_idx]  # [word_num, 1]
            batch_adjacency_matrix = adjacency_matrix[batch_idx]  # [word_num, word_num]

            row_sum = batch_adjacency_matrix.sum(dim=1, keepdim=True) + eps
            normalized_adjacency = batch_adjacency_matrix / row_sum

            propagated = batch_gate_inputs.clone()
            for _ in range(num_hops):
                propagated = torch.matmul(normalized_adjacency, propagated)

            propagated_gate_inputs[start_idx:end_idx] = propagated

        return propagated_gate_inputs

    def compute_smoothness_loss(self, node_values, adjacency_matrix, ptr):
        
        smoothness_losses = []
        for i, adj in enumerate(adjacency_matrix):
            start = ptr[i]
            end = ptr[i + 1]
            num_nodes_i = end - start
            p_i = node_values[start:end]

            if p_i.dim() == 1:
                p_i = p_i.unsqueeze(1)

            D = torch.diag(adj.sum(dim=1))
            L = D - adj

            reg_i = (p_i.T @ L @ p_i).squeeze()
            reg_i = reg_i / (num_nodes_i ** 2)  
            smoothness_losses.append(reg_i)

        return torch.stack(smoothness_losses).mean()


    def compute_group_sparsity_loss(self, gate_inputs, split_matrix, ptr,
                                    mode="gaussian",        # 'exact' or 'gaussian'
                                    normalize=True       # whether to normalize by group size
        ):
        group_losses = []

        for i, adj in enumerate(split_matrix):
            start = ptr[i].item()
            end = ptr[i + 1].item()
            gate_i = gate_inputs[start:end].squeeze(1)

            adj_np = adj.cpu().numpy()
            G = nx.from_numpy_array(adj_np)
            components = list(nx.connected_components(G))

            for group in components:
                group = list(group)
                if len(group) == 0:
                    continue

                probs = gate_i[group]
                group_size = len(group)

                if mode == "exact":
                    p_inactive = torch.clamp(1.0 - probs, min=1e-6, max=1.0)
                    p_zero = torch.prod(p_inactive)
                    p_active = 1.0 - p_zero

                    if normalize:
                        p_active = p_active / group_size  # or / sqrt(group_size)

                    group_losses.append(p_active)

                elif mode == "gaussian":
                    mu = probs.sum()
                    var = (probs * (1 - probs)).sum()
                    std = torch.sqrt(var + 1e-6)

                    if normalize:
                        denom = std * torch.sqrt(torch.tensor(group_size, dtype=torch.float32, device=probs.device))
                    else:
                        denom = std
                    shift = (mu - 0.5) / (denom + 1e-6)
                    cdf = 0.5 * (1 + erf(shift))

                    group_losses.append(cdf)

                else:
                    raise ValueError(f"Unknown mode: {mode}")

        if len(group_losses) == 0:
            return torch.tensor(0.0, device=gate_inputs.device)

        return torch.stack(group_losses).mean()

    def permute_within_samples(self, node_feature_clean, data_ptr_clean, gate_inputs):
        permuted_node_features = []
        permuted_gate_inputs = []

        for i in range(data_ptr_clean.size(0) - 1):
            start_idx = data_ptr_clean[i].item()
            end_idx = data_ptr_clean[i + 1].item()

            sample_permutation = torch.randperm(end_idx - start_idx)

            permuted_node_features.append(node_feature_clean[start_idx:end_idx][sample_permutation])
            permuted_gate_inputs.append(gate_inputs[start_idx:end_idx][sample_permutation])

        permuted_node_features = torch.cat(permuted_node_features, dim=0)
        permuted_gate_inputs = torch.cat(permuted_gate_inputs, dim=0)

        return permuted_node_features, permuted_gate_inputs

    def create_split_matrix(self, score_matrices_not_relu, data_ptr_clean, num=1):
        split_matrices = []
        softmax_scores_matrices = []

        for batch_idx in range(len(data_ptr_clean) - 1):

            start_idx = data_ptr_clean[batch_idx].item()
            end_idx = data_ptr_clean[batch_idx + 1].item()
            num_nodes = end_idx - start_idx

            score_matrix = score_matrices_not_relu[batch_idx]

            score_matrix = score_matrix * (1 - torch.eye(score_matrix.size(0), device=score_matrix.device))
            score_matrix_flat = score_matrix.view(-1)  
            softmax_flat = F.softmax(score_matrix_flat, dim=0)  
            softmax_scores = softmax_flat.view_as(score_matrix)  

            threshold = (num / (num_nodes*(num_nodes-1))) if num_nodes > 1 else 0.0

            split_matrix = (softmax_scores > threshold).float()

            split_matrices.append(split_matrix)
            softmax_scores_matrices.append(softmax_scores)

        return split_matrices, softmax_scores_matrices

    def forward(self, data, model_type = 'STE'):


        if self.args.data_name == 'graph_sst2' and self.args.adj_type =='cross':
            syn_data=data[0]
            sem_data=data[1]
            node_feature_clean, data_ptr_clean, edge_index_clean = self.mask_adjust_graph(syn_data.x, syn_data.ptr, syn_data.edge_index) 
            
            sem_node_feature_clean, sem_data_ptr_clean, semantic_edge_index_clean = self.mask_adjust_graph(sem_data.x, sem_data.ptr, sem_data.edge_index) 
            adjacency_matrix= batch_to_adj_matrices(edge_index_clean, data_ptr_clean) 
            score_matrices, score_matrices_not_relu = self.similarity_matrix(node_feature_clean, data_ptr_clean) 

            semantic_adj= batch_to_adj_matrices(semantic_edge_index_clean, sem_data_ptr_clean) 
            _, softmax_scores_matrices = self.create_split_matrix(score_matrices_not_relu, data_ptr_clean, num=self.args.sem_num_threshold)

        else:

            node_feature_clean, data_ptr_clean, edge_index_clean = self.mask_adjust_graph(data.x, data.ptr, data.edge_index) 
            adjacency_matrix= batch_to_adj_matrices(edge_index_clean, data_ptr_clean) 
            score_matrices, score_matrices_not_relu = self.similarity_matrix(node_feature_clean, data_ptr_clean) 
            
            semantic_adj, softmax_scores_matrices = self.create_split_matrix(score_matrices_not_relu, data_ptr_clean, num=self.args.sem_num_threshold)

            if self.args.adj_type == 'syntactic':
                main_adj = adjacency_matrix
            elif self.args.adj_type == 'semantic':
                semantic_edge_index_clean = self.convert_split_matrices_to_edge_indices(semantic_adj, data_ptr_clean)
                main_adj = semantic_adj
            else:
                fully_connected_edge_index = self.create_fully_connected_edge_index(data_ptr_clean, node_feature_clean.size(0))
                main_adj = batch_to_adj_matrices(fully_connected_edge_index, data_ptr_clean)

        gate_inputs, test_gate_inputs, regularizer = self.compress(node_feature_clean, model_type)

        if self.args.mix_up_rate == -1:
            
            permuted_node_features, permuted_gate_inputs = self.permute_within_samples(
                node_feature_clean, data_ptr_clean, gate_inputs
            )

            mixup_rates = torch.distributions.Beta(2.0, 2.0).sample((gate_inputs.shape[0],)).to(gate_inputs.device)

            mixup_rates = mixup_rates.unsqueeze(1)

            mix_gate_node_feature = (1 - mixup_rates) * node_feature_clean + mixup_rates * permuted_node_features
            mix_gate_GT = (1 - mixup_rates) * gate_inputs + mixup_rates * permuted_gate_inputs

            mix_gate_pred, _, _ = self.compress(mix_gate_node_feature, model_type)
            mixup_reg=torch.mean((mix_gate_GT - mix_gate_pred) ** 2)

        elif self.args.mix_up_rate != 0.:
            permuted_node_features, permuted_gate_inputs = self.permute_within_samples(
                node_feature_clean, data_ptr_clean, gate_inputs
            )
            mix_gate_node_feature = (1-self.args.mix_up_rate)*node_feature_clean + self.args.mix_up_rate * permuted_node_features
            mix_gate_GT = (1-self.args.mix_up_rate)*gate_inputs + self.args.mix_up_rate * permuted_gate_inputs

            mix_gate_pred, _, _ = self.compress(mix_gate_node_feature, model_type)
            mixup_reg=torch.mean((mix_gate_GT - mix_gate_pred) ** 2)

        weighted_adj_matrix = [adj * score for adj, score in zip(adjacency_matrix, score_matrices)] # 추가 2025/04/02 myh

        if self.args.group_matric == "cos_nagative_cut":
            softmax_scores_matrices =None
            binary_scores = [(score > 0).float() for score in score_matrices] # cos negative 자르기
            
        elif self.args.group_matric == "cos_softmax_node_cut":
            binary_scores = semantic_adj
        
        split_matrix = [adj * mask for adj, mask in zip(adjacency_matrix, binary_scores)]

        if self.args.adj_type == 'cross':
            main_adj = self.compute_average_adj(semantic_adj, adjacency_matrix)
        if not self.args.use_weighted_adjacency:
            loss_smo= self.compute_smoothness_loss(gate_inputs, main_adj, data_ptr_clean)
        
            gate_inputs = self.propagate_gate_inputs_list(gate_inputs, main_adj, data_ptr_clean, num_hops=self.args.num_hops)
        
        else:
            loss_smo= self.compute_smoothness_loss(gate_inputs, weighted_adj_matrix, data_ptr_clean)

            gate_inputs = self.propagate_gate_inputs_list(gate_inputs, weighted_adj_matrix, data_ptr_clean, num_hops=self.args.num_hops)

        if self.args.reg_type == 'group_sparsity_G_N':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=True)
        elif self.args.reg_type == 'group_sparsity_E_N':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=True)
        elif self.args.reg_type == 'group_sparsity_G':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=False)
        elif self.args.reg_type == 'group_sparsity_E':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=False)

        elif self.args.reg_type == 'L0_group_sparsity_G_N':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=True)
        elif self.args.reg_type == 'L0_group_sparsity_E_N':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=True)
        elif self.args.reg_type == 'L0_group_sparsity_G':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=False)
        elif self.args.reg_type == 'L0_group_sparsity_E':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=False)
        gate_inputs = gate_inputs.reshape(-1, 1)

        if self.args.mix_up_rate != 0.:
            out={'gate_inputs': gate_inputs, 'test_gate_inputs': test_gate_inputs, 'regularizer': regularizer, 'loss_smo': loss_smo, 'data_ptr_clean': data_ptr_clean, 'split_matrix':split_matrix, 'mixup_reg': mixup_reg, 'score_matrices': (score_matrices, score_matrices_not_relu, softmax_scores_matrices), 'adj_matrix':(adjacency_matrix, binary_scores, split_matrix)}
            return out
        out={'gate_inputs': gate_inputs, 'test_gate_inputs': test_gate_inputs, 'regularizer': regularizer, 'loss_smo': loss_smo, 'data_ptr_clean': data_ptr_clean, 'split_matrix': split_matrix, 'score_matrices': (score_matrices, score_matrices_not_relu, softmax_scores_matrices), 'adj_matrix':(adjacency_matrix, binary_scores, split_matrix)}
        return out
    def create_fully_connected_edge_index(self, ptr, total_num_nodes):

        edge_indices = []
        
        for i in range(len(ptr) - 1):
            start, end = ptr[i].item(), ptr[i + 1].item()
            nodes = torch.arange(start, end, device=ptr.device)

            row = nodes.repeat_interleave(len(nodes) - 1)
            col_list = []
            
            for n in nodes:
                others = nodes[nodes != n]
                col_list.append(others)
            
            col = torch.cat(col_list, dim=0)
            edge = torch.stack([row, col], dim=0)  
            
            edge_indices.append(edge)
        
        full_edge_index = torch.cat(edge_indices, dim=1) 
        return full_edge_index
    def convert_split_matrices_to_edge_indices(self, split_matrices, data_ptr_clean):

        edge_indices = []

        for batch_idx, adj in enumerate(split_matrices):
            num_nodes = adj.size(0)

            src, dst = torch.nonzero(adj, as_tuple=True)

            start_idx = data_ptr_clean[batch_idx].item()
            src += start_idx
            dst += start_idx

            edge_indices.append(torch.stack([src, dst], dim=0))  # [2, num_edges_sample]

        edge_index = torch.cat(edge_indices, dim=1)  # [2, total_num_edges]
        return edge_index

    def compute_average_adj(self, semantic_adj, adjacency_matrix):

        assert len(semantic_adj) == len(adjacency_matrix), "Different lengths of adjacency matrices"

        average_adj = [
            (sem_adj + adj) / 2 for sem_adj, adj in zip(semantic_adj, adjacency_matrix)
        ]
        return average_adj


class Our_Selector_V1_WithGNN(Our_Selector_V1):
    def __init__(self, args, device
        , gnn_in_channels, gnn_hidden_channels, gnn_out_channels, gnn_num_layers=3, gat_heads=4, dropout=0.5
        , model_type='GCN'
        , adj_type='syntactic'
        , **kwargs):
        super(Our_Selector_V1_WithGNN, self).__init__(args, device, **kwargs)
        
        if model_type == 'GAT':
            self.gnn = MultiLayerGAT(
                in_channels=gnn_in_channels,
                hidden_channels=gnn_hidden_channels,
                out_channels=gnn_out_channels,
                num_layers=gnn_num_layers,
                heads=gat_heads,
            )
        elif model_type == 'GCN':
            self.gnn = MultiLayerGCN(
                in_channels=gnn_in_channels,
                hidden_channels=gnn_hidden_channels,
                out_channels=gnn_out_channels,
                num_layers=gnn_num_layers,
                dropout = dropout
            ).to(device)
        elif model_type == 'graphSAGE':
            if args.sage_agg == 'pool':
                aggr = PoolAggregation(gnn_hidden_channels, gnn_hidden_channels)
            else:
                aggr=args.sage_agg

            self.gnn = GraphSAGE(
                in_channels=gnn_in_channels,         
                hidden_channels=gnn_hidden_channels,     
                out_channels=gnn_out_channels,        
                num_layers=gnn_num_layers,           
                dropout=dropout,             
                act='relu',              #
                norm='layernorm',        
                jk='last',              
                aggr=aggr              
            ).to(device)
        self.compressor = nn.Identity().to(device)
        self.adj_type = adj_type

    def forward(self, data, model_type='STE'):

        
        node_feature_clean, data_ptr_clean, edge_index_clean = self.mask_adjust_graph(data.x, data.ptr, data.edge_index) 
        node_feature_clean.requires_grad = True

        adjacency_matrix= batch_to_adj_matrices(edge_index_clean, data_ptr_clean)         
        score_matrices, score_matrices_not_relu = self.similarity_matrix(node_feature_clean, data_ptr_clean) 
        
        semantic_adj, softmax_scores_matrices = self.create_split_matrix(score_matrices_not_relu, data_ptr_clean, num=self.args.sem_num_threshold)

        if self.adj_type == 'syntactic':
            gnn_embeddings = self.gnn(node_feature_clean, edge_index_clean)
            main_adj = adjacency_matrix
        elif self.adj_type == 'semantic':
            semantic_edge_index_clean = self.convert_split_matrices_to_edge_indices(semantic_adj, data_ptr_clean)
            gnn_embeddings = self.gnn(node_feature_clean, semantic_edge_index_clean)
            main_adj = semantic_adj
        else:
            fully_connected_edge_index = self.create_fully_connected_edge_index(data_ptr_clean, node_feature_clean.size(0))
            gnn_embeddings = self.gnn(node_feature_clean, fully_connected_edge_index)
            main_adj = batch_to_adj_matrices(fully_connected_edge_index, data_ptr_clean)
        
        gate_inputs, test_gate_inputs, regularizer = self.compress(gnn_embeddings, model_type)

        if self.args.mix_up_rate == -1:
            
            permuted_node_features, permuted_gate_inputs = self.permute_within_samples(
                node_feature_clean, data_ptr_clean, gate_inputs
            )
            mixup_rates = torch.distributions.Beta(2.0, 2.0).sample((gate_inputs.shape[0],)).to(gate_inputs.device)

            mixup_rates = mixup_rates.unsqueeze(1)

            mix_gate_node_feature = (1 - mixup_rates) * node_feature_clean + mixup_rates * permuted_node_features
            mix_gate_GT = (1 - mixup_rates) * gate_inputs + mixup_rates * permuted_gate_inputs
        
            mix_gate_pred, _, _ = self.compress(mix_gate_node_feature, model_type)
            mixup_reg=torch.mean((mix_gate_GT - mix_gate_pred) ** 2)

        elif self.args.mix_up_rate != 0.:
            permuted_node_features, permuted_gate_inputs = self.permute_within_samples(
                node_feature_clean, data_ptr_clean, gate_inputs
            )
            mix_gate_node_feature = (1-self.args.mix_up_rate)*node_feature_clean + self.args.mix_up_rate * permuted_node_features
            mix_gate_GT = (1-self.args.mix_up_rate)*gate_inputs + self.args.mix_up_rate * permuted_gate_inputs

            mix_gate_pred, _, _ = self.compress(mix_gate_node_feature, model_type)
            mixup_reg=torch.mean((mix_gate_GT - mix_gate_pred) ** 2)

        weighted_adj_matrix = [adj * score for adj, score in zip(adjacency_matrix, score_matrices)]

        if self.args.group_matric == "cos_nagative_cut":
            softmax_scores_matrices =None
            binary_scores = [(score > 0).float() for score in score_matrices]
            
        elif self.args.group_matric == "cos_softmax_node_cut":
            binary_scores = semantic_adj
        
        split_matrix = [adj * mask for adj, mask in zip(adjacency_matrix, binary_scores)]
        
        average_adj = self.compute_average_adj(semantic_adj, adjacency_matrix)

        if not self.args.use_weighted_adjacency:
            
            loss_smo= self.compute_smoothness_loss(gate_inputs, adjacency_matrix, data_ptr_clean) 

            gate_inputs = self.propagate_gate_inputs_list(gate_inputs, adjacency_matrix, data_ptr_clean, num_hops=self.args.num_hops)
        
        else:
            weighted_adj_matrix = [adj * score for adj, score in zip(average_adj, score_matrices)] 
            loss_smo= self.compute_smoothness_loss(gate_inputs, weighted_adj_matrix, data_ptr_clean)

            gate_inputs = self.propagate_gate_inputs_list(gate_inputs, weighted_adj_matrix, data_ptr_clean, num_hops=self.args.num_hops)

        if self.args.reg_type == 'group_sparsity_G_N':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=True)
        elif self.args.reg_type == 'group_sparsity_E_N':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=True)
        elif self.args.reg_type == 'group_sparsity_G':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=False)
        elif self.args.reg_type == 'group_sparsity_E':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=False)

        elif self.args.reg_type == 'L0_group_sparsity_G_N':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=True)
        elif self.args.reg_type == 'L0_group_sparsity_E_N':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=True)
        elif self.args.reg_type == 'L0_group_sparsity_G':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=False)
        elif self.args.reg_type == 'L0_group_sparsity_E':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=False)
        gate_inputs = gate_inputs.reshape(-1, 1)

        if self.args.mix_up_rate != 0.:
            out={'gate_inputs': gate_inputs, 'test_gate_inputs': test_gate_inputs, 'regularizer': regularizer, 'loss_smo': loss_smo, 'data_ptr_clean': data_ptr_clean, 'split_matrix':split_matrix, 'mixup_reg': mixup_reg, 'score_matrices': (score_matrices, score_matrices_not_relu, softmax_scores_matrices), 'adj_matrix':(adjacency_matrix, semantic_adj, split_matrix)}
            return out
        out={'gate_inputs': gate_inputs, 'test_gate_inputs': test_gate_inputs, 'regularizer': regularizer, 'loss_smo': loss_smo, 'data_ptr_clean': data_ptr_clean, 'split_matrix': split_matrix, 'score_matrices': (score_matrices, score_matrices_not_relu, softmax_scores_matrices), 'adj_matrix':(adjacency_matrix, semantic_adj, split_matrix)}
        return out


class Our_Selector_V1_With_Attetnion(Our_Selector_V1):
    def __init__(self, args, device, num_heads=4, num_layers=1,
                **kwargs):
        super(Our_Selector_V1_With_Attetnion, self).__init__(args, device, **kwargs)

        self.attention_layer = MultiLayerSelfAttentionWithAdjacency(self.node_hidden_dim, num_heads, num_layers)
    def forward(self, data, model_type = 'STE'):

        if self.args.data_name == 'graph_sst2' and self.args.adj_type =='cross':
            syn_data=data[0]
            sem_data=data[1]
            node_feature_clean, data_ptr_clean, edge_index_clean = self.mask_adjust_graph(syn_data.x, syn_data.ptr, syn_data.edge_index) # 추가 2025/03/25 myh
            
            sem_node_feature_clean, sem_data_ptr_clean, semantic_edge_index_clean = self.mask_adjust_graph(sem_data.x, sem_data.ptr, sem_data.edge_index) # 추가 2025/03/25 myh

            adjacency_matrix= batch_to_adj_matrices(edge_index_clean, data_ptr_clean) 
            score_matrices, score_matrices_not_relu = self.similarity_matrix(node_feature_clean, data_ptr_clean) 

            semantic_adj= batch_to_adj_matrices(semantic_edge_index_clean, sem_data_ptr_clean) 
            _, softmax_scores_matrices = self.create_split_matrix(score_matrices_not_relu, data_ptr_clean, num=self.args.sem_num_threshold)

        else:

            node_feature_clean, data_ptr_clean, edge_index_clean = self.mask_adjust_graph(data.x, data.ptr, data.edge_index) # 추가 2025/03/25 myh
            # padding 제거하는 부분을 따로 함수로 만들어서 처리
            adjacency_matrix= batch_to_adj_matrices(edge_index_clean, data_ptr_clean) # list type len(batch_size) 각 원소는 해당 배치의 (node_num, node_num) size 추가 2025/03/25 myh
            score_matrices, score_matrices_not_relu = self.similarity_matrix(node_feature_clean, data_ptr_clean) 
            
            semantic_adj, softmax_scores_matrices = self.create_split_matrix(score_matrices_not_relu, data_ptr_clean, num=self.args.sem_num_threshold)
            # edge_index_clean과 data_ptr_clean을 사용하여 adjacency_matrix 생성 list type으로 len(adjacency_matrix) -> batch_size 각 원소는 각 샘플의 (word_num, word_num) 형태

            if self.args.adj_type == 'syntactic':
                main_adj = adjacency_matrix
            elif self.args.adj_type == 'semantic':
                semantic_edge_index_clean = self.convert_split_matrices_to_edge_indices(semantic_adj, data_ptr_clean)
                main_adj = semantic_adj
            else:
                fully_connected_edge_index = self.create_fully_connected_edge_index(data_ptr_clean, node_feature_clean.size(0))
                main_adj = batch_to_adj_matrices(fully_connected_edge_index, data_ptr_clean)
        node_feature_clean = self.attention_layer(node_feature_clean, data_ptr_clean)

        gate_inputs, test_gate_inputs, regularizer = self.compress(node_feature_clean, model_type)

        if self.args.mix_up_rate == -1:
            
            permuted_node_features, permuted_gate_inputs = self.permute_within_samples(
                node_feature_clean, data_ptr_clean, gate_inputs
            )
            # 노드마다 1개의 mixup rate 샘플 (Beta(2,2) 분포에서)
            mixup_rates = torch.distributions.Beta(2.0, 2.0).sample((gate_inputs.shape[0],)).to(gate_inputs.device)

            # [num_nodes] -> [num_nodes, 1] 로 확장 (broadcast 용)
            mixup_rates = mixup_rates.unsqueeze(1)

            # mix-up 적용
            mix_gate_node_feature = (1 - mixup_rates) * node_feature_clean + mixup_rates * permuted_node_features
            mix_gate_GT = (1 - mixup_rates) * gate_inputs + mixup_rates * permuted_gate_inputs

            mix_gate_pred, _, _ = self.compress(mix_gate_node_feature, model_type)
            mixup_reg=torch.mean((mix_gate_GT - mix_gate_pred) ** 2)

        elif self.args.mix_up_rate != 0.:
            permuted_node_features, permuted_gate_inputs = self.permute_within_samples(
                node_feature_clean, data_ptr_clean, gate_inputs
            )
            mix_gate_node_feature = (1-self.args.mix_up_rate)*node_feature_clean + self.args.mix_up_rate * permuted_node_features
            mix_gate_GT = (1-self.args.mix_up_rate)*gate_inputs + self.args.mix_up_rate * permuted_gate_inputs

            mix_gate_pred, _, _ = self.compress(mix_gate_node_feature, model_type)
            mixup_reg=torch.mean((mix_gate_GT - mix_gate_pred) ** 2)


        # smoothness_loss 계산
        
        # weight_attention 계산
        # x_input = torch.nan_to_num(data.x, 0.0)
        # weight_attention = self.weight_attetion(x_input, filtered_edge_index) # (E,1)
        
        # weight_attention = self.weight_attetion(node_feature_clean, data_ptr_clean)
        # attention_adj_matrix = [adj * score for adj, score in zip(adjacency_matrix, weight_attention)]
        # score_matrix 계산
        
        weighted_adj_matrix = [adj * score for adj, score in zip(adjacency_matrix, score_matrices)] # 추가 2025/04/02 myh

        if self.args.group_matric == "cos_nagative_cut":
            softmax_scores_matrices =None
            binary_scores = [(score > 0).float() for score in score_matrices] # cos negative 자르기
            
        elif self.args.group_matric == "cos_softmax_node_cut":
            binary_scores = semantic_adj
        
        split_matrix = [adj * mask for adj, mask in zip(adjacency_matrix, binary_scores)]

        

        if self.args.adj_type == 'cross':
            # main_adj = split_matrix
            main_adj = self.compute_average_adj(semantic_adj, adjacency_matrix)
        if not self.args.use_weighted_adjacency:
            loss_smo= self.compute_smoothness_loss(gate_inputs, main_adj, data_ptr_clean)
        
            # propagation 수행
            gate_inputs = self.propagate_gate_inputs_list(gate_inputs, main_adj, data_ptr_clean, num_hops=self.args.num_hops)
        
        else:
            loss_smo= self.compute_smoothness_loss(gate_inputs, weighted_adj_matrix, data_ptr_clean)

            # propagation 수행
            gate_inputs = self.propagate_gate_inputs_list(gate_inputs, weighted_adj_matrix, data_ptr_clean, num_hops=self.args.num_hops)

        if self.args.reg_type == 'group_sparsity_G_N':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=True)
        elif self.args.reg_type == 'group_sparsity_E_N':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=True)
        elif self.args.reg_type == 'group_sparsity_G':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=False)
        elif self.args.reg_type == 'group_sparsity_E':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=False)

        elif self.args.reg_type == 'L0_group_sparsity_G_N':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=True)
        elif self.args.reg_type == 'L0_group_sparsity_E_N':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=True)
        elif self.args.reg_type == 'L0_group_sparsity_G':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=False)
        elif self.args.reg_type == 'L0_group_sparsity_E':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=False)
        gate_inputs = gate_inputs.reshape(-1, 1)

        if self.args.mix_up_rate != 0.:
            out={'gate_inputs': gate_inputs, 'test_gate_inputs': test_gate_inputs, 'regularizer': regularizer, 'loss_smo': loss_smo, 'data_ptr_clean': data_ptr_clean, 'split_matrix':split_matrix, 'mixup_reg': mixup_reg, 'score_matrices': (score_matrices, score_matrices_not_relu, softmax_scores_matrices), 'adj_matrix':(adjacency_matrix, binary_scores, split_matrix)}
            return out
        out={'gate_inputs': gate_inputs, 'test_gate_inputs': test_gate_inputs, 'regularizer': regularizer, 'loss_smo': loss_smo, 'data_ptr_clean': data_ptr_clean, 'split_matrix': split_matrix, 'score_matrices': (score_matrices, score_matrices_not_relu, softmax_scores_matrices), 'adj_matrix':(adjacency_matrix, binary_scores, split_matrix)}
        return out

class Our_Selector_V1_WithGNN(Our_Selector_V1):
    def __init__(self, args, device
        , gnn_in_channels, gnn_hidden_channels, gnn_out_channels, gnn_num_layers=3, gat_heads=4, dropout=0.5
        , model_type='GCN'
        , adj_type='syntactic'
        , **kwargs):
        super(Our_Selector_V1_WithGNN, self).__init__(args, device, **kwargs)
        
        # GAT 모듈 추가
        if model_type == 'GAT':
            self.gnn = MultiLayerGAT(
                in_channels=gnn_in_channels,
                hidden_channels=gnn_hidden_channels,
                out_channels=gnn_out_channels,
                num_layers=gnn_num_layers,
                heads=gat_heads,
            )
        elif model_type == 'GCN':
            self.gnn = MultiLayerGCN(
                in_channels=gnn_in_channels,
                hidden_channels=gnn_hidden_channels,
                out_channels=gnn_out_channels,
                num_layers=gnn_num_layers,
                dropout = dropout
            ).to(device)
        elif model_type == 'graphSAGE':
            if args.sage_agg == 'pool':
                aggr = PoolAggregation(gnn_hidden_channels, gnn_hidden_channels)
            else:
                aggr=args.sage_agg

            self.gnn = GraphSAGE(
                in_channels=gnn_in_channels,         # 노드 feature dim
                hidden_channels=gnn_hidden_channels,      # 중간 hidden dim
                out_channels=gnn_out_channels,         # 예측 dim (클래스 수 or embedding dim)
                num_layers=gnn_num_layers,            # SAGEConv 층 수
                dropout=dropout,             # 드롭아웃 비율
                act='relu',              # 활성화 함수
                norm='layernorm',        # 정규화 방식
                jk='last',               # Jumping Knowledge
                aggr=aggr              # SAGEConv aggregator ('mean', 'pool', 'lstm', etc.)
            ).to(device)
        self.compressor = nn.Identity().to(device)
        # self.compressor = nn.Linear()
        self.adj_type = adj_type

    def forward(self, data, model_type='STE'):
        # gnn을 사용하여 노드 임베딩 계산
        # # 입력 데이터의 requires_grad 설정
        # if not data.x.requires_grad:
        #     data.x = data.x.requires_grad_(True)

        # valid_nodes = ~torch.isnan(data.x).any(dim=1)
        
        # # edge_index 필터링 - nan 노드와 연결된 엣지 제거
        # valid_edges = valid_nodes[data.edge_index[0]] & valid_nodes[data.edge_index[1]]
        # filtered_edge_index = data.edge_index[:, valid_edges]

        
        node_feature_clean, data_ptr_clean, edge_index_clean = self.mask_adjust_graph(data.x, data.ptr, data.edge_index) # 추가 2025/03/25 myh
        node_feature_clean.requires_grad = True

        adjacency_matrix= batch_to_adj_matrices(edge_index_clean, data_ptr_clean) # list type len(batch_size) 각 원소는 해당 배치의 (node_num, node_num) size 추가 2025/03/25 myh
        # padding 제거하는 부분을 따로 함수로 만들어서 처리
        
        # edge_index_clean과 data_ptr_clean을 사용하여 adjacency_matrix 생성 list type으로 len(adjacency_matrix) -> batch_size 각 원소는 각 샘플의 (word_num, word_num) 형태
        score_matrices, score_matrices_not_relu = self.similarity_matrix(node_feature_clean, data_ptr_clean) 
        
        semantic_adj, softmax_scores_matrices = self.create_split_matrix(score_matrices_not_relu, data_ptr_clean, num=self.args.sem_num_threshold)

        if self.adj_type == 'syntactic':
            gnn_embeddings = self.gnn(node_feature_clean, edge_index_clean)
            main_adj = adjacency_matrix
        elif self.adj_type == 'semantic':
            semantic_edge_index_clean = self.convert_split_matrices_to_edge_indices(semantic_adj, data_ptr_clean)
            gnn_embeddings = self.gnn(node_feature_clean, semantic_edge_index_clean)
            main_adj = semantic_adj
        else:
            fully_connected_edge_index = self.create_fully_connected_edge_index(data_ptr_clean, node_feature_clean.size(0))
            gnn_embeddings = self.gnn(node_feature_clean, fully_connected_edge_index)
            main_adj = batch_to_adj_matrices(fully_connected_edge_index, data_ptr_clean)
        
        gate_inputs, test_gate_inputs, regularizer = self.compress(gnn_embeddings, model_type)

        if self.args.mix_up_rate == -1:
            
            permuted_node_features, permuted_gate_inputs = self.permute_within_samples(
                node_feature_clean, data_ptr_clean, gate_inputs
            )
            # 노드마다 1개의 mixup rate 샘플 (Beta(2,2) 분포에서)
            mixup_rates = torch.distributions.Beta(2.0, 2.0).sample((gate_inputs.shape[0],)).to(gate_inputs.device)

            # [num_nodes] -> [num_nodes, 1] 로 확장 (broadcast 용)
            mixup_rates = mixup_rates.unsqueeze(1)

            # mix-up 적용
            mix_gate_node_feature = (1 - mixup_rates) * node_feature_clean + mixup_rates * permuted_node_features
            mix_gate_GT = (1 - mixup_rates) * gate_inputs + mixup_rates * permuted_gate_inputs
        
            mix_gate_pred, _, _ = self.compress(mix_gate_node_feature, model_type)
            mixup_reg=torch.mean((mix_gate_GT - mix_gate_pred) ** 2)

        elif self.args.mix_up_rate != 0.:
            permuted_node_features, permuted_gate_inputs = self.permute_within_samples(
                node_feature_clean, data_ptr_clean, gate_inputs
            )
            mix_gate_node_feature = (1-self.args.mix_up_rate)*node_feature_clean + self.args.mix_up_rate * permuted_node_features
            mix_gate_GT = (1-self.args.mix_up_rate)*gate_inputs + self.args.mix_up_rate * permuted_gate_inputs

            mix_gate_pred, _, _ = self.compress(mix_gate_node_feature, model_type)
            mixup_reg=torch.mean((mix_gate_GT - mix_gate_pred) ** 2)


        # smoothness_loss 계산
        
        # weight_attention 계산
        # x_input = torch.nan_to_num(data.x, 0.0)
        # weight_attention = self.weight_attetion(x_input, filtered_edge_index) # (E,1)
        
        # weight_attention = self.weight_attetion(node_feature_clean, data_ptr_clean)
        # attention_adj_matrix = [adj * score for adj, score in zip(adjacency_matrix, weight_attention)]
        # score_matrix 계산
        # score_matrices, score_matrices_not_relu = self.similarity_matrix(node_feature_clean, data_ptr_clean) # list type len(batch_size) 각 원소는 해당 배치의 (node_num, node_num) size 추가 2025/04/02 myh
        
        weighted_adj_matrix = [adj * score for adj, score in zip(adjacency_matrix, score_matrices)] # 추가 2025/04/02 myh

        if self.args.group_matric == "cos_nagative_cut":
            softmax_scores_matrices =None
            binary_scores = [(score > 0).float() for score in score_matrices] # cos negative 자르기
            
        elif self.args.group_matric == "cos_softmax_node_cut":
            binary_scores = semantic_adj
            # binary_scores, softmax_scores_matrices = self.create_split_matrix(score_matrices_not_relu, data_ptr_clean)
        
        split_matrix = [adj * mask for adj, mask in zip(adjacency_matrix, binary_scores)]
        
        average_adj = self.compute_average_adj(semantic_adj, adjacency_matrix)

        if not self.args.use_weighted_adjacency:
            
            loss_smo= self.compute_smoothness_loss(gate_inputs, adjacency_matrix, data_ptr_clean) 
        
            # propagation 수행
            gate_inputs = self.propagate_gate_inputs_list(gate_inputs, adjacency_matrix, data_ptr_clean, num_hops=self.args.num_hops)
        
        else:
            weighted_adj_matrix = [adj * score for adj, score in zip(average_adj, score_matrices)] 
            loss_smo= self.compute_smoothness_loss(gate_inputs, weighted_adj_matrix, data_ptr_clean)

            # propagation 수행
            gate_inputs = self.propagate_gate_inputs_list(gate_inputs, weighted_adj_matrix, data_ptr_clean, num_hops=self.args.num_hops)

        if self.args.reg_type == 'group_sparsity_G_N':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=True)
        elif self.args.reg_type == 'group_sparsity_E_N':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=True)
        elif self.args.reg_type == 'group_sparsity_G':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=False)
        elif self.args.reg_type == 'group_sparsity_E':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=False)

        elif self.args.reg_type == 'L0_group_sparsity_G_N':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=True)
        elif self.args.reg_type == 'L0_group_sparsity_E_N':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=True)
        elif self.args.reg_type == 'L0_group_sparsity_G':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=False)
        elif self.args.reg_type == 'L0_group_sparsity_E':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=False)
        gate_inputs = gate_inputs.reshape(-1, 1)

        if self.args.mix_up_rate != 0.:
            out={'gate_inputs': gate_inputs, 'test_gate_inputs': test_gate_inputs, 'regularizer': regularizer, 'loss_smo': loss_smo, 'data_ptr_clean': data_ptr_clean, 'split_matrix':split_matrix, 'mixup_reg': mixup_reg, 'score_matrices': (score_matrices, score_matrices_not_relu, softmax_scores_matrices), 'adj_matrix':(adjacency_matrix, semantic_adj, split_matrix)}
            return out
        out={'gate_inputs': gate_inputs, 'test_gate_inputs': test_gate_inputs, 'regularizer': regularizer, 'loss_smo': loss_smo, 'data_ptr_clean': data_ptr_clean, 'split_matrix': split_matrix, 'score_matrices': (score_matrices, score_matrices_not_relu, softmax_scores_matrices), 'adj_matrix':(adjacency_matrix, semantic_adj, split_matrix)}
        return out


class Our_Selector_V1_WithGNN_Double(Our_Selector_V1):
    def __init__(self, args, device
        , syn_gnn_in_channels, syn_gnn_hidden_channels, syn_gnn_out_channels
        , sem_gnn_in_channels, sem_gnn_hidden_channels, sem_gnn_out_channels
        , syn_gnn_num_layers=3, syn_gat_heads=4
        , sem_gnn_num_layers=3, sem_gat_heads=4
        ,model_type='GCN'
        ,dropout=0.5
        , **kwargs):
        super(Our_Selector_V1_WithGNN_Double, self).__init__(args, device, **kwargs)

        # GAT 모듈 추가
        if model_type == 'GAT':
            # GAT 모듈 추가
            self.syntetic_gnn = MultiLayerGAT(
                in_channels=syn_gnn_in_channels,
                hidden_channels=syn_gnn_hidden_channels,
                out_channels=syn_gnn_out_channels,
                num_layers=syn_gnn_num_layers,
                heads=syn_gat_heads
            )
            
            self.semantic_gnn = MultiLayerGAT(
                in_channels=sem_gnn_in_channels,
                hidden_channels=sem_gnn_hidden_channels,
                out_channels=sem_gnn_out_channels,
                num_layers=sem_gnn_num_layers,
                heads=sem_gat_heads
            )
        elif model_type == 'GCN':
            self.syntetic_gnn = MultiLayerGCN(
                in_channels=syn_gnn_in_channels,
                hidden_channels=syn_gnn_hidden_channels,
                out_channels=syn_gnn_out_channels,
                num_layers=syn_gnn_num_layers,
                dropout=dropout,
            )
            
            self.semantic_gnn = MultiLayerGCN(
                in_channels=sem_gnn_in_channels,
                hidden_channels=sem_gnn_hidden_channels,
                out_channels=sem_gnn_out_channels,
                num_layers=sem_gnn_num_layers,
                dropout=dropout,
            )
        elif model_type == 'graphSAGE':

            if args.sage_agg == 'pool':
                syn_aggr = PoolAggregation(syn_gnn_hidden_channels, syn_gnn_hidden_channels)
                sem_aggr = PoolAggregation(sem_gnn_hidden_channels, sem_gnn_hidden_channels)


            else:
                syn_aggr=args.sage_agg
                sem_aggr=args.sage_agg

            self.syntetic_gnn = GraphSAGE(
                in_channels=syn_gnn_in_channels,         # 노드 feature dim
                hidden_channels=syn_gnn_hidden_channels,      # 중간 hidden dim
                out_channels=syn_gnn_out_channels,         # 예측 dim (클래스 수 or embedding dim)
                num_layers=syn_gnn_num_layers,            # SAGEConv 층 수
                dropout=dropout,             # 드롭아웃 비율
                act='relu',              # 활성화 함수
                norm='layernorm',        # 정규화 방식
                jk='last',               # Jumping Knowledge
                aggr=syn_aggr                # SAGEConv aggregator ('mean', 'pool', 'lstm', etc.)
                    
                )


            self.semantic_gnn = GraphSAGE(
                in_channels=sem_gnn_in_channels,         # 노드 feature dim
                hidden_channels=sem_gnn_hidden_channels,      # 중간 hidden dim
                out_channels=sem_gnn_out_channels,         # 예측 dim (클래스 수 or embedding dim)
                num_layers=sem_gnn_num_layers,            # SAGEConv 층 수
                dropout=dropout,             # 드롭아웃 비율
                act='relu',              # 활성화 함수
                norm='layernorm',        # 정규화 방식
                jk='last',               # Jumping Knowledge
                aggr=sem_aggr               # SAGEConv aggregator ('mean', 'pool', 'lstm', etc.)
                )

        if syn_gnn_out_channels == 1:
            self.compressor = nn.Identity() # 채널 polling
        else:
            self.compressor = nn.Linear(syn_gnn_out_channels, 1) # 채널 polling
        # self.compressor = nn.Linear()

    def forward(self, data, model_type='STE'):
        # gnn을 사용하여 노드 임베딩 계산
        # # 입력 데이터의 requires_grad 설정
        # if not data.x.requires_grad:
        #     data.x = data.x.requires_grad_(True)
        
        if self.args.data_name == 'graph_sst2' and self.args.adj_type =='cross':
            syn_data=data[0]
            sem_data=data[1]
            # valid_nodes = ~torch.isnan(data.x).any(dim=1)
            
            # # edge_index 필터링 - nan 노드와 연결된 엣지 제거
            # valid_edges = valid_nodes[data.edge_index[0]] & valid_nodes[data.edge_index[1]]
            # filtered_edge_index = data.edge_index[:, valid_edges]
            node_feature_clean, data_ptr_clean, edge_index_clean = self.mask_adjust_graph(syn_data.x, syn_data.ptr, syn_data.edge_index) # 추가 2025/03/25 myh
            
            sem_node_feature_clean, sem_data_ptr_clean, semantic_edge_index_clean = self.mask_adjust_graph(sem_data.x, sem_data.ptr, sem_data.edge_index) # 추가 2025/03/25 myh
            # padding 제거하는 부분을 따로 함수로 만들어서 처리
            adjacency_matrix= batch_to_adj_matrices(edge_index_clean, data_ptr_clean) # list type len(batch_size) 각 원소는 해당 배치의 (node_num, node_num) size 추가 2025/03/25 myh
            score_matrices, score_matrices_not_relu = self.similarity_matrix(node_feature_clean, data_ptr_clean) 

            semantic_adj= batch_to_adj_matrices(semantic_edge_index_clean, sem_data_ptr_clean) # list type len(batch_size) 각 원소는 해당 배치의 (node_num, node_num) size 추가 2025/03/25 myh
            # sem_score_matrices, sem_score_matrices_not_relu = self.similarity_matrix(sem_node_feature_clean, sem_data_ptr_clean) 
            _, softmax_scores_matrices = self.create_split_matrix(score_matrices_not_relu, data_ptr_clean, num=self.args.sem_num_threshold)

        else:
            # valid_nodes = ~torch.isnan(data.x).any(dim=1)
            
            # # edge_index 필터링 - nan 노드와 연결된 엣지 제거
            # valid_edges = valid_nodes[data.edge_index[0]] & valid_nodes[data.edge_index[1]]
            # filtered_edge_index = data.edge_index[:, valid_edges]

            node_feature_clean, data_ptr_clean, edge_index_clean = self.mask_adjust_graph(data.x, data.ptr, data.edge_index) # 추가 2025/03/25 myh
            
            # padding 제거하는 부분을 따로 함수로 만들어서 처리
            adjacency_matrix= batch_to_adj_matrices(edge_index_clean, data_ptr_clean) # list type len(batch_size) 각 원소는 해당 배치의 (node_num, node_num) size 추가 2025/03/25 myh
            
            # edge_index_clean과 data_ptr_clean을 사용하여 adjacency_matrix 생성 list type으로 len(adjacency_matrix) -> batch_size 각 원소는 각 샘플의 (word_num, word_num) 형태
            
            score_matrices, score_matrices_not_relu = self.similarity_matrix(node_feature_clean, data_ptr_clean) 
            
            semantic_adj, softmax_scores_matrices = self.create_split_matrix(score_matrices_not_relu, data_ptr_clean, num=self.args.sem_num_threshold)

            semantic_edge_index_clean = self.convert_split_matrices_to_edge_indices(semantic_adj, data_ptr_clean)

        syntetic_gnn_embeddings = self.syntetic_gnn(node_feature_clean, edge_index_clean)
        
        semantic_gnn_embeddings = self.semantic_gnn(node_feature_clean, semantic_edge_index_clean)
        gnn_embeddings = torch.mean(torch.stack([syntetic_gnn_embeddings, semantic_gnn_embeddings], dim=0),dim=0)
        
        gate_inputs, test_gate_inputs, regularizer = self.compress(gnn_embeddings, model_type)

        if self.args.mix_up_rate == -1:
            
            permuted_node_features, permuted_gate_inputs = self.permute_within_samples(
                node_feature_clean, data_ptr_clean, gate_inputs
            )
            # 노드마다 1개의 mixup rate 샘플 (Beta(2,2) 분포에서)
            mixup_rates = torch.distributions.Beta(2.0, 2.0).sample((gate_inputs.shape[0],)).to(gate_inputs.device)

            # [num_nodes] -> [num_nodes, 1] 로 확장 (broadcast 용)
            mixup_rates = mixup_rates.unsqueeze(1)

            # mix-up 적용
            mix_gate_node_feature = (1 - mixup_rates) * node_feature_clean + mixup_rates * permuted_node_features
            mix_gate_GT = (1 - mixup_rates) * gate_inputs + mixup_rates * permuted_gate_inputs
        
            mix_gate_pred, _, _ = self.compress(mix_gate_node_feature, model_type)
            mixup_reg=torch.mean((mix_gate_GT - mix_gate_pred) ** 2)

        elif self.args.mix_up_rate != 0.:
            permuted_node_features, permuted_gate_inputs = self.permute_within_samples(
                node_feature_clean, data_ptr_clean, gate_inputs
            )
            mix_gate_node_feature = (1-self.args.mix_up_rate)*node_feature_clean + self.args.mix_up_rate * permuted_node_features
            mix_gate_GT = (1-self.args.mix_up_rate)*gate_inputs + self.args.mix_up_rate * permuted_gate_inputs

            mix_gate_pred, _, _ = self.compress(mix_gate_node_feature, model_type)
            mixup_reg=torch.mean((mix_gate_GT - mix_gate_pred) ** 2)


        # smoothness_loss 계산
        
        # weight_attention 계산
        # x_input = torch.nan_to_num(data.x, 0.0)
        # weight_attention = self.weight_attetion(x_input, filtered_edge_index) # (E,1)
        
        # weight_attention = self.weight_attetion(node_feature_clean, data_ptr_clean)
        # attention_adj_matrix = [adj * score for adj, score in zip(adjacency_matrix, weight_attention)]
        # score_matrix 계산
        # score_matrices, score_matrices_not_relu = self.similarity_matrix(node_feature_clean, data_ptr_clean) # list type len(batch_size) 각 원소는 해당 배치의 (node_num, node_num) size 추가 2025/04/02 myh
        
        # weighted_adj_matrix = [adj * score for adj, score in zip(adjacency_matrix, score_matrices)] # 추가 2025/04/02 myh


        if self.args.group_matric == "cos_nagative_cut":
            softmax_scores_matrices =None
            binary_scores = [(score > 0).float() for score in score_matrices] # cos negative 자르기
            
        elif self.args.group_matric == "cos_softmax_node_cut":
            binary_scores = semantic_adj
            # binary_scores, softmax_scores_matrices = self.create_split_matrix(score_matrices_not_relu, data_ptr_clean)
        
        split_matrix = [adj * mask for adj, mask in zip(adjacency_matrix, binary_scores)]
        
        average_adj = self.compute_average_adj(semantic_adj, adjacency_matrix) # syn , sem 둘의 평균

        if not self.args.use_weighted_adjacency:
            loss_smo= self.compute_smoothness_loss(gate_inputs, average_adj, data_ptr_clean)
            # loss_smo_syn= self.compute_smoothness_loss(gate_inputs, adjacency_matrix, data_ptr_clean)
            # loss_smo_sem= self.compute_smoothness_loss(gate_inputs, semantic_adj, data_ptr_clean)
            
            # pdb.set_trace()
            # propagation 수행
            gate_inputs = self.propagate_gate_inputs_list(gate_inputs, average_adj, data_ptr_clean, num_hops=self.args.num_hops)
        
        else:
            weighted_adj_matrix = [adj * score for adj, score in zip(average_adj, score_matrices)] 
            loss_smo= self.compute_smoothness_loss(gate_inputs, weighted_adj_matrix, data_ptr_clean)

            # propagation 수행
            gate_inputs = self.propagate_gate_inputs_list(gate_inputs, weighted_adj_matrix, data_ptr_clean, num_hops=self.args.num_hops)

        if self.args.reg_type == 'group_sparsity_G_N':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=True)
        elif self.args.reg_type == 'group_sparsity_E_N':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=True)
        elif self.args.reg_type == 'group_sparsity_G':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=False)
        elif self.args.reg_type == 'group_sparsity_E':
            regularizer = self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=False)

        elif self.args.reg_type == 'L0_group_sparsity_G_N':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=True)
        elif self.args.reg_type == 'L0_group_sparsity_E_N':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=True)
        elif self.args.reg_type == 'L0_group_sparsity_G':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="gaussian", normalize=False)
        elif self.args.reg_type == 'L0_group_sparsity_E':
            regularizer = self.args.reg_rate * regularizer + (1-self.args.reg_rate) * self.compute_group_sparsity_loss(gate_inputs, split_matrix, data_ptr_clean, mode="extract", normalize=False)
        gate_inputs = gate_inputs.reshape(-1, 1)

        if self.args.mix_up_rate != 0.:
            out={'gate_inputs': gate_inputs, 'test_gate_inputs': test_gate_inputs, 'regularizer': regularizer, 'loss_smo': loss_smo, 'data_ptr_clean': data_ptr_clean, 'split_matrix':split_matrix, 'mixup_reg': mixup_reg, 'score_matrices': (score_matrices, score_matrices_not_relu, softmax_scores_matrices), 'adj_matrix':(adjacency_matrix, semantic_adj, split_matrix)}
            return out
        out={'gate_inputs': gate_inputs, 'test_gate_inputs': test_gate_inputs, 'regularizer': regularizer, 'loss_smo': loss_smo, 'data_ptr_clean': data_ptr_clean, 'split_matrix': split_matrix, 'score_matrices': (score_matrices, score_matrices_not_relu, softmax_scores_matrices), 'adj_matrix':(adjacency_matrix, semantic_adj, split_matrix)}
        return out

    def convert_split_matrices_to_edge_indices(self, split_matrices, data_ptr_clean):
        """
        split_matrices (list of torch.Tensor): 각 샘플의 (num_nodes, num_nodes) 이진 인접 행렬
        data_ptr_clean (torch.Tensor): [batch_size + 1] 형태로 각 샘플 시작/끝 인덱스
        반환:
            torch.Tensor: 전체 배치에 대한 edge_index [2, num_edges]
        """
        edge_indices = []

        for batch_idx, adj in enumerate(split_matrices):
            num_nodes = adj.size(0)

            # 1이 있는 위치를 (i,j) 좌표로 얻음
            src, dst = torch.nonzero(adj, as_tuple=True)

            # 전역 인덱스로 변환 (샘플별 시작 인덱스를 더함)
            start_idx = data_ptr_clean[batch_idx].item()
            src += start_idx
            dst += start_idx

            edge_indices.append(torch.stack([src, dst], dim=0))  # [2, num_edges_sample]

        # 모든 샘플을 이어붙임
        edge_index = torch.cat(edge_indices, dim=1)  # [2, total_num_edges]
        return edge_index

    def compute_average_adj(self, semantic_adj, adjacency_matrix):
        """
        semantic_adj와 adjacency_matrix의 평균 adjacency matrix를 계산합니다.

        Args:
            semantic_adj (list of torch.Tensor): semantic adjacency matrices 리스트.
            adjacency_matrix (list of torch.Tensor): 일반 adjacency matrices 리스트.

        Returns:
            list of torch.Tensor: 각 샘플의 평균 adjacency matrix 리스트.
        """
        # 두 리스트의 길이가 동일한지 확인
        assert len(semantic_adj) == len(adjacency_matrix), "두 리스트의 길이가 다릅니다."

        # 각 샘플에 대해 평균 계산
        average_adj = [
            (sem_adj + adj) / 2 for sem_adj, adj in zip(semantic_adj, adjacency_matrix)
        ]
        return average_adj

class SelfAttentionWeightedAdjacency(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** 0.5

    def forward(self, x, ptr):
        """
        x: (total_nodes, D)
        ptr: (batch_size + 1,)
        """
        attn_matrices = []
        num_graphs = ptr.size(0) - 1

        for i in range(num_graphs):
            start, end = ptr[i].item(), ptr[i + 1].item()
            x_i = x[start:end]  # (Nᵢ, D)

            q = self.query_proj(x_i)  # (Nᵢ, D)
            k = self.key_proj(x_i)    # (Nᵢ, D)

            attn_score = torch.matmul(q, k.T) / self.scale  # (Nᵢ, Nᵢ)
            attn_score = F.softmax(attn_score, dim=-1)      # Row-wise softmax

            attn_matrices.append(attn_score)

        return attn_matrices
class CosineWeightedAdjacency(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x, ptr):
        """
        x: (total_nodes, D)  전체 배치의 노드 임베딩
        ptr: (batch_size + 1,)  그래프별 노드 시작/끝 인덱스
        """
        similarity_matrices = []
        similarity_matrices_not_relu = []

        num_graphs = ptr.size(0) - 1
        for i in range(num_graphs):
            start, end = ptr[i].item(), ptr[i + 1].item()
            x_i = x[start:end]  # 해당 그래프의 노드 임베딩 (num_nodes_i, D)

            normalized = F.normalize(x_i, p=2, dim=1)
            sim_i_ = torch.matmul(normalized, normalized.T)  # (num_nodes_i, num_nodes_i)
            sim_i = self.relu(sim_i_)  # [-1,1] -> [0,1]

            similarity_matrices.append(sim_i)
            similarity_matrices_not_relu.append(sim_i_)

        return similarity_matrices, similarity_matrices_not_relu



import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerSelfAttentionWithAdjacency(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_layers = num_layers

        self.attn_layers = nn.ModuleList([
            SelfAttentionBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x, ptr):
        """
        Args:
            x: (total_nodes, D)
            ptr: (batch_size + 1,)

        Returns:
            out_x: (total_nodes, D)
            attn_all_layers: list of list of attention matrices per graph
            shape: [num_layers][batch_size] = (H, Ni, Ni)
        """
        for layer in self.attn_layers:
            x = layer(x, ptr)

        return x  # (total_nodes, D), list of (num_layers × num_graphs × (H, Ni, Ni))


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.scale = self.head_dim ** 0.5

    def forward(self, x, ptr):
        """
        Args:
            x: (total_nodes, D)
            ptr: (batch_size + 1,)

        Returns:
            out_x: (total_nodes, D)
            attn_matrices: list of (num_heads, Ni, Ni)
        """
        out_list = []
        num_graphs = ptr.size(0) - 1

        for i in range(num_graphs):
            start, end = ptr[i].item(), ptr[i + 1].item()
            x_i = x[start:end]  # (Ni, D)

            q = self.q_proj(x_i).view(-1, self.num_heads, self.head_dim)  # (Ni, H, Dh)
            k = self.k_proj(x_i).view(-1, self.num_heads, self.head_dim)
            v = self.v_proj(x_i).view(-1, self.num_heads, self.head_dim)

            # (H, Ni, Dh)
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (H, Ni, Ni)
            attn_weights = F.softmax(attn_scores, dim=-1)

            out_i = torch.matmul(attn_weights, v)  # (H, Ni, Dh)
            out_i = out_i.transpose(0, 1).contiguous().view(end - start, self.embed_dim)  # (Ni, D)

            out_list.append(self.out_proj(out_i))

        out_x = torch.cat(out_list, dim=0)  # (total_nodes, D)
        return out_x

class SelfAttentionWeightedAdjacencyWithOutput(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = self.head_dim ** 0.5

    def forward(self, x, ptr):
        """
        Args:
            x: (total_nodes, D)
            ptr: (batch_size + 1,)

        Returns:
            out_x: (total_nodes, D)
            attn_matrices: list of (num_heads, Ni, Ni) attention weights per graph
        """
        out_list = []

        num_graphs = ptr.size(0) - 1

        for i in range(num_graphs):
            start, end = ptr[i].item(), ptr[i + 1].item()
            x_i = x[start:end]  # (Ni, D)

            # Project to Q, K, V and split heads
            q = self.q_proj(x_i).view(-1, self.num_heads, self.head_dim)  # (Ni, H, Dh)
            k = self.k_proj(x_i).view(-1, self.num_heads, self.head_dim)  # (Ni, H, Dh)
            v = self.v_proj(x_i).view(-1, self.num_heads, self.head_dim)  # (Ni, H, Dh)

            # (H, Ni, Dh)
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

            # Compute attention weights: (H, Ni, Ni)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Weighted sum of V: (H, Ni, Dh)
            out_i = torch.matmul(attn_weights, v)

            # (Ni, H, Dh) -> (Ni, D)
            out_i = out_i.transpose(0, 1).contiguous().view(end - start, self.embed_dim)

            out_list.append(self.out_proj(out_i))  # Optional output projection

        out_x = torch.cat(out_list, dim=0)  # (total_nodes, D)
        return out_x  # list of (H, Ni, Ni)

from transformers import GPT2PreTrainedModel, GPT2Model
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
class CustomGPT2Classifier(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # GPT-2 모델 로드
        self.transformer = GPT2Model.from_pretrained("gpt2")

        # 분류 헤드 추가
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)


        # 가중치 초기화
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # GPT-2 모델의 출력
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                if attention_mask is not None:
                    # 마지막으로 유효한 토큰의 위치를 계산
                    sequence_lengths = attention_mask.sum(dim=-1) - 1
                else:
                    sequence_lengths = -1
                    logger.warning(
                        f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                        "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                    )
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )



def print_grad(grad):
    print("Gradient:", grad)        

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.norm import LayerNorm, GraphNorm


class MultiLayerGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=4, norm_type='layer'):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GATv2Conv(in_channels, out_channels, heads=1))
        else:
            self.convs.append(GATv2Conv(in_channels, hidden_channels * heads, heads=heads))
            self.norms.append(self._get_norm_layer(hidden_channels * heads, norm_type))

            for _ in range(num_layers - 2):
                self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads))
                self.norms.append(self._get_norm_layer(hidden_channels * heads, norm_type))

            # 마지막 레이어
            self.convs.append(GATv2Conv(hidden_channels * heads, out_channels, heads=1))

        self.hidden_states = []

    def _get_norm_layer(self, channels, norm_type):
        if norm_type == 'layer':
            return LayerNorm(channels)
        elif norm_type == 'graph':
            return GraphNorm(channels)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def forward(self, x, edge_index):
        self.hidden_states = []
        # pdb.set_trace()
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            self.hidden_states.append(x.detach())
            if i != self.num_layers - 1:
                x = self.norms[i](x)
                x = F.elu(x)
        return x        


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.norm import LayerNorm, GraphNorm


class MultiLayerGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, norm_type='layer', dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout

        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            self.norms.append(self._get_norm_layer(hidden_channels, norm_type))

            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.norms.append(self._get_norm_layer(hidden_channels, norm_type))

            self.convs.append(GCNConv(hidden_channels, out_channels))

        self.hidden_states = []

    def _get_norm_layer(self, channels, norm_type):
        if norm_type == 'layer':
            return LayerNorm(channels)
        elif norm_type == 'graph':
            return GraphNorm(channels)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def forward(self, x, edge_index):
        self.hidden_states = []
        # pdb.set_trace()
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            self.hidden_states.append(x.detach())
            if i != self.num_layers - 1:
                x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

from torch_geometric.nn import GINConv
import torch.nn.functional as F

class GINNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

from torch_geometric.nn import GraphSAGE

class GraphSAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x        

from torch_geometric.nn import SAGEConv
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn.aggr import MaxAggregation

class PoolAggregation(MaxAggregation):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.mlp = Sequential(
            Linear(in_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        )

    def forward(self, x, index, ptr=None, dim_size=None, dim=-2):
        x = self.mlp(x)  # 각 이웃에 MLP 적용
        return super().forward(x, index, ptr, dim_size, dim) # element-wise max        
    








# ##########################################################

import torch
from torch import nn
from transformers import GPT2Model, GPT2PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

class BioMedLMForSequenceClassification(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

        # GPT2는 pad_token이 없을 수 있으므로 eos_token으로 설정
        if config.pad_token_id is None:
            # config.pad_token_id = 28895
            config.pad_token_id = config.eos_token_id

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.transformer(input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs.last_hidden_state

        # 각 배치에서 유효한 마지막 토큰의 인덱스를 계산
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1  # 실제 토큰 수 - 1
        else:
            # attention_mask가 없을 경우, pad_token 기준 계산
            seq_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1

        # safe indexing
        batch_size = input_ids.size(0)
        device = input_ids.device
        pooled_output = hidden_states[torch.arange(batch_size, device=device), seq_lengths]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )


class Predictor_only(nn.Module):
    def __init__(self, tokenizer, predictor, args):
        super(Predictor_only, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.predictor = predictor

    def forward(self, data, meta_data):

        if self.args.data_name == 'bioasq':
            # pdb.set_trace()
            if self.args.predictor == 'BioMedLM':
                texts = meta_data['text']
                main_sentence = [f"[CONTEXT] {x['context'].strip()}" for x in texts]
                query = [f"[QUESTION] {x['question'].strip()} [ANSWER]" for x in texts]
                # pdb.set_trace()
                encoded = self.tokenizer(
                    main_sentence,
                    text_pair=query,
                    padding=True,
                    truncation="only_first",  # context만 자르기
                    
                    # truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
            elif self.args.predictor == 'biolinkBert':
                texts = meta_data['text']  # 배치의 모든 텍스트 리스트
                encoded = self.tokenizer(
                    texts,  # 텍스트 리스트 전체 전달
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )

        else:
            texts = meta_data['text']  # 배치의 모든 텍스트 리스트
            encoded = self.tokenizer(
                texts,  # 텍스트 리스트 전체 전달
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
        encoded = encoded.to(self.args.device)

        outputs = self.predictor(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )

        return outputs



##########################L2X###########################

import torch
import torch.nn as nn
import torch.nn.functional as F




class GumbelSelectorCNN(nn.Module):
    def __init__(self, num_words, embedding_dim, hidden_dim, maxlen):
        super(GumbelSelectorCNN, self).__init__()
        self.embedding = nn.Embedding(num_words, embedding_dim, padding_idx=0)

        self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)  # conv1_gumbel

        # Global info path
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.global_fc = nn.Linear(hidden_dim, hidden_dim)  # new_dense_1

        # Local info path
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)  # conv2_gumbel
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)  # conv3_gumbel

        # Combined features
        self.dropout = nn.Dropout(0.2)
        self.conv_last = nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=1, padding=0)  # conv_last_gumbel

        # Output logits per token
        self.conv_out = nn.Conv1d(hidden_dim, 1, kernel_size=1, padding=0)  # conv4_gumbel

    def forward(self, x):
        """
        Args:
            x: LongTensor (B, T) – token indices
        Returns:
            logits: FloatTensor (B, T, 1) – token-level selection logits
        """
        emb = self.embedding(x)  # (B, T, D)
        emb = self.dropout(emb)

        # Prepare for conv1d: (B, D, T)
        x_conv = emb.permute(0, 2, 1)

        # First conv layer
        first_layer = F.relu(self.conv1(x_conv))  # (B, 100, T)

        # Global info
        global_feat = self.global_pool(first_layer)  # (B, 100, 1)
        global_feat = global_feat.squeeze(-1)       # (B, 100)
        global_feat = F.relu(self.global_fc(global_feat))  # (B, 100)

        # Local info
        local_feat = F.relu(self.conv2(first_layer))        # (B, 100, T)
        local_feat = F.relu(self.conv3(local_feat))         # (B, 100, T)

        # Concatenate global info at each position
        B, C, T = local_feat.shape
        global_feat_exp = global_feat.unsqueeze(2).expand(-1, -1, T)  # (B, 100, T)
        combined = torch.cat([local_feat, global_feat_exp], dim=1)    # (B, 200, T)

        # Final conv and output
        net = self.dropout(combined)
        net = F.relu(self.conv_last(net))            # (B, 100, T)
        logits = self.conv_out(net)                  # (B, 1, T)
        logits = logits.permute(0, 2, 1)             # (B, T, 1) to match Keras

        return logits
    

class SampleConcrete(nn.Module):
    def __init__(self, hidden_dim, k: int, tau0: float = 0.5):
        """
        Args:
            k: 선택할 top-k feature 개수
            tau0: softmax temperature (낮을수록 hard해짐)
        """
        self.hidden_dim = hidden_dim
        super(SampleConcrete, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            CustomBatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.k = k
        self.tau0 = tau0

    def forward(self, data, threshold = -1):
        """
        Args:
            logits: Tensor of shape (B, d) - raw logits for each feature
        Returns:
            Tensor of shape (B, d) - soft mask during training, hard top-k mask during eval
        """
        logits = self.encoder(data)
        B, d, _ = logits.shape
        if self.training:
            # (1) Gumbel noise
            eps = 1e-20
            
            uniform = torch.rand(B, self.k, d, device=logits.device)
            gumbel = -torch.log(-torch.log(uniform + eps) + eps)

            # (2) Noisy logits
            logits = logits.permute(0, 2, 1)
            logits_expanded = logits.expand(-1, self.k, -1)  # (B, k, d)
            noisy_logits = (logits_expanded + gumbel) / self.tau0

            # (3) Softmax over d
            soft_samples = F.softmax(noisy_logits, dim=-1)  # (B, k, d)

            # (4) Max over k samples → soft top-k approximation
            sample = torch.max(soft_samples, dim=1).values  # (B, d)

        else:
            # print(threshold)
            # pdb.set_trace()
            softmax_logits = F.softmax(logits.squeeze(-1), dim=-1)  # (B, d, 1) -> (B, d)
            if threshold == -1:
                # Hard top-k selection
                threshold = torch.topk(softmax_logits.squeeze(-1), self.k, dim=-1, sorted=True).values[:, -1].unsqueeze(1)  # (B, 1)
            # print("softmax_logits", softmax_logits)
            sample = (softmax_logits > threshold).float()  # (B, d)
            return sample.squeeze(), softmax_logits

        return sample, None


class SampleHardConcrete(nn.Module):
    def __init__(self, hidden_dim, k: int, tau0: float = 1.0):
        """
        Args:
            k: 선택할 top-k feature 개수
            tau0: softmax temperature (낮을수록 hard해짐)
        """
        self.hidden_dim = hidden_dim
        super(SampleHardConcrete, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            CustomBatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.k = k
        self.tau0 = tau0
        self.l = -0.1
        self.r = 1.1

    def forward(self, data, threshold = -1):
        """
        Args:
            logits: Tensor of shape (B, d) - raw logits for each feature
        Returns:
            Tensor of shape (B, d) - soft mask during training, hard top-k mask during eval
        """
        logits = self.encoder(data)
        B, d, _ = logits.shape
        if self.training:
            # (1) Gumbel noise
            eps = 1e-20
            u = torch.rand_like(logits)
            gumbel_noise = torch.log(u + eps) - torch.log(1 - u + eps)  # logistic noise
            s = torch.sigmoid((logits + gumbel_noise) / self.tau)  # (B, d)
            stretched = s * (self.r - self.l) + self.l
            gate = torch.clamp(stretched, 0, 1)  # (B, d)
            return gate, None

        else:

            
            eps = 1e-20
            u = torch.rand_like(logits)
            gumbel_noise = torch.log(u + eps) - torch.log(1 - u + eps)  # logistic noise
            s = torch.sigmoid((logits + gumbel_noise) / self.tau)  # (B, d)
            stretched = s * (self.r - self.l) + self.l
            probs = torch.clamp(stretched, 0, 1)  # (B, d)

            # # deterministic path
            # probs = torch.sigmoid(logits) * (self.r - self.l) + self.l  # (B, d)
            # probs = torch.clamp(probs, 0, 1)

            if self.k is not None and threshold == -1:
                threshold = torch.topk(probs, self.k, dim=-1).values[:, -1].unsqueeze(1)  # (B, 1)

            if threshold != -1:
                gate = (probs >= threshold).float()  # hard top-k
            else:
                gate = (probs >= 0.5).float()  # standard hard concrete

            return gate, probs


from torch.nn.utils.rnn import pad_sequence
class L2X_Align(nn.Module):
    def __init__(self, predictor_tokenizer, predictor, hidden_dim, num_heads, num_layers, args, mask_embedding=None , **kargs):
        super(L2X_Align, self).__init__()
        self.args = args
        self.predictor = predictor.to(args.device)
        self.hidden_dim = hidden_dim
        self.tokenizer= predictor_tokenizer
        self.attention_layer = MultiLayerSelfAttentionWithAdjacency_L2X(embed_dim=hidden_dim, num_heads=4, num_layers=2)
        
        self.selected_model = SampleConcrete(hidden_dim=hidden_dim, k=self.args.num_samples)  
        if args.target_model == 'xlnet':
            self.word_embedding = self.predictor.transformer.word_embedding
        elif args.target_model == 'deberta':
            self.word_embedding = self.predictor.deberta.embeddings.word_embeddings
        elif args.target_model in ['gpt2', 'BioMedLM']:
            self.word_embedding = self.predictor.transformer.wte
        elif args.target_model == 'roberta':
            self.word_embedding = self.predictor.roberta.embeddings.word_embeddings
        elif args.target_model == 'biolinkBert':
            self.word_embedding = self.predictor.bert.embeddings.word_embeddings

        # self.special_token_ids = [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]  # CLS, SEP, [MASK] 토큰 ID
        special_tokens = self.tokenizer.special_tokens_map
        self.special_token_ids = [self.tokenizer.convert_tokens_to_ids(v) for k, v in special_tokens.items()]
        if self.args.data_name == 'bioasq':
            if self.args.target_model == 'BioMedLM':
                self.special_token_ids.extend(self.tokenizer.special_tokens_map['additional_special_tokens'])
        self.mask_embedding = mask_embedding
            
        
    def forward(self, data, threshold =-1, baseline=False):    
        
        # 토큰화 및 인코딩
        texts = data['text']
        if self.args.target_model == 'BioMedLM':
            context_prompt = texts['context']
            question_prompt = texts['question']

            # 토크나이징
            encoded = self.tokenizer(
                context_prompt,
                text_pair=question_prompt,
                padding=True,
                truncation="only_first",  # context만 자르기!
                max_length=512,
                # max_length=1024,
                return_tensors="pt"
            ).to(self.args.device)
        else:
            encoded = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.args.device)

        if baseline:
            if self.args.target_model in ['deberta', 'xlnet']:
                outputs = self.predictor(
                    input_ids=encoded['input_ids'],
                    token_type_ids=encoded['token_type_ids'],
                    attention_mask=encoded['attention_mask'],
                )
            elif self.args.target_model in ['roberta', 'gpt2', 'BioMedLM', 'biolinkBert']:
                outputs = self.predictor(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                )
            return outputs
        
        

        input_embeding = pad_sequence(data['embedding_matrix_full'],  # list of [num_nodes_i, hidden_dim]
                                      batch_first=True,               # shape: [B, max_len, hidden_dim]
                                      padding_value=0.0
                                    ).to(self.args.device)
        attention_mask = (input_embeding.abs().sum(dim=-1) != 0)
        
        input_embeding=self.attention_layer(input_embeding, attention_mask)
        # pdb.set_trace()
        gate, softmax_logits = self.selected_model(input_embeding, threshold)

            
        input_ids = encoded['input_ids']
        total_tokens_num = encoded['attention_mask'].sum().item()
        original_embeddings = self.word_embedding(encoded['input_ids']).to(self.args.device)
        pad_id = self.tokenizer.pad_token_id
        special_ids = set(self.tokenizer.all_special_ids)
        special_ids.discard(pad_id)

        special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for sid in special_ids:
            special_token_mask |= (input_ids == sid)

        # print(gate.shape)
        # print(special_token_mask.shape)

        if gate.dim() == 1:
            gate = gate.view(input_ids.shape[0], -1)

        gate[special_token_mask] = 1.0

        if self.training:
            filtered_gate_list = [
                sg[am == 1].tolist()     # 각 배치별 soft_gate, attention_mask
                for sg, am in zip(gate, encoded['attention_mask'])
            ]
        # else:
        #     # pdb.set_trace()
        #     filtered_gate_list = [
        #         sg[am == 1].tolist()     # 각 배치별 soft_gate, attention_mask
        #         for sg, am in zip(gate, encoded['attention_mask'])
        #     ]
        else:
            filtered_gate_list = [
                sg[am == 1].tolist()     # 각 배치별 soft_gate, attention_mask
                for sg, am in zip(softmax_logits, encoded['attention_mask'])
            ]

        b, s, h = original_embeddings.shape
        if self.mask_embedding != None:
            mask_embeddings = self.mask_embedding.expand(b, s, -1).to(self.args.device)
        else:
            mask_embeddings = torch.zeros_like(original_embeddings).to(self.args.device)
        
        gated_embeddings = (original_embeddings * gate.unsqueeze(-1) + mask_embeddings * (1 - gate).unsqueeze(-1))

        if self.args.target_model in ['deberta', 'xlnet']:
            outputs = self.predictor(
                inputs_embeds=gated_embeddings,
                token_type_ids=encoded['token_type_ids'],
                attention_mask=encoded['attention_mask'],
            )
        elif self.args.target_model in ['roberta', 'gpt2', 'BioMedLM', 'biolinkBert']:
            outputs = self.predictor(
                inputs_embeds=gated_embeddings,
                attention_mask=encoded['attention_mask'],
            )

            
            return outputs, softmax_logits, total_tokens_num, filtered_gate_list
                
        return outputs, gate, total_tokens_num, filtered_gate_list
            
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** 0.5

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, D)  # Batch x SeqLen x EmbeddingDim
            mask: (B, N) or (B, N, N), optional

        Returns:
            out: (B, N, D)
            attn_weights: (B, H, N, N)
        """
        B, N, D = x.size()

        # Project to QKV
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, Dh)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, N, N)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, N, N)
        attn_output = torch.matmul(attn_weights, v)    # (B, H, N, Dh)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)  # (B, N, D)

        out = self.out_proj(attn_output)  # (B, N, D)

        return out, attn_weights

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionBlock_L2X(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.scale = self.head_dim ** 0.5

    def forward(self, x, mask=None):
        B, N, D = x.size()

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, Dh)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, N, N)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # → (16, 1, 1, 512)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (B, H, N, Dh)

        out = attn_output.transpose(1, 2).contiguous().view(B, N, D)  # (B, N, D)
        out_x = self.out_proj(out)  # (B, N, D)
        return out_x

class MultiLayerSelfAttentionWithAdjacency_L2X(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_layers = num_layers

        self.attn_layers = nn.ModuleList([
            SelfAttentionBlock_L2X(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        """
        Args:
            x: (total_nodes, D)
            ptr: (batch_size + 1,)

        Returns:
            out_x: (total_nodes, D)
            attn_all_layers: list of list of attention matrices per graph
            shape: [num_layers][batch_size] = (H, Ni, Ni)
        """
        for layer in self.attn_layers:
            x = layer(x, mask)

        return x  # (total_nodes, D), list of (num_layers × num_graphs × (H, Ni, Ni))

def sample_gumbel(shape, device='cpu', eps=1e-10):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)