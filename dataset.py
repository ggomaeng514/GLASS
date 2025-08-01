import torch
import pdb

import os
import re
import pdb
import string
import pickle
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt

import spacy
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    XLNetTokenizer, XLNetModel,
    GPT2Tokenizer, GPT2Model, GPT2ForSequenceClassification, GPT2LMHeadModel, GPT2Config,
    OPTForCausalLM,
    RobertaTokenizer, RobertaModel, RobertaForSequenceClassification,
    DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DebertaV2Config,
    BertTokenizer, BertModel,
    AutoModelForCausalLM,
)

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import networkx as nx
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx

def normalize(text):
    return unicodedata.normalize("NFKC", text)

def clean_basic_latin_only(s):
    s = unicodedata.normalize("NFKC", s.lower())
    return ''.join(c for c in s if c.isalnum() and 'LATIN' in unicodedata.name(c, ''))
def clean_strict_ascii(s):
    s = unicodedata.normalize("NFKC", s.lower())
    return re.sub(r'[^a-z0-9]', '', s)
def ascii_only(text):
    return ''.join(c for c in text if ord(c) < 128)

def split_final_punctuation(word_list):
    if not word_list:
        return word_list

    last_word = word_list[-1]
    if len(last_word) >= 2 and last_word[-1] in '.!?':
        return word_list[:-1] + [last_word[:-1], last_word[-1]]
    else:
        return word_list


class DependencyGraphDataset(Dataset):
    def __init__(self, texts, labels=None, graph_process_type = 'direct_root', tokenizer_type = 'xlnet', data_name='glue_sst2' ,encoder_layer_depth=-1, max_token_length = 512, **kargs):

        
        self.nlp = spacy.load("en_core_web_sm")
        if tokenizer_type == 'xlnet':
            self.tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
            self.model = XLNetModel.from_pretrained("xlnet-base-cased")
        elif tokenizer_type == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2Model.from_pretrained("gpt2")
        elif tokenizer_type == 'deberta':
            model_name = "microsoft/deberta-v3-base"
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
            self.model = DebertaV2ForSequenceClassification.from_pretrained(model_name)
        elif tokenizer_type == 'deberta_small':
            model_name = "microsoft/deberta-v3-small"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        elif tokenizer_type == 'deberta_large':
            model_name = "microsoft/deberta-v3-large"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        elif tokenizer_type == 'bert':
            model_name = "bert-base-uncased"
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
        elif tokenizer_type == 'roberta':
            model_name = "FacebookAI/roberta-base"
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaModel.from_pretrained(model_name)
        elif tokenizer_type == 'BioMedLM':
            model_name = "stanford-crfm/BioMedLM"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)   
            special_tokens_dict = {
                "additional_special_tokens": ["[CONTEXT]", "[QUESTION]", "[ANSWER]"]
            }
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        elif tokenizer_type == 'biolinkBert':
            self.tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-large')
            self.model = AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-large')
        elif tokenizer_type == 'galactica':
            model_name = "facebook/galactica-6.7b"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.add_special_tokens({
                'bos_token': '<s>',
                'eos_token': '</s>',
                'unk_token': '<unk>',
                'pad_token': '<pad>', 
            })    
            self.tokenizer.padding_side = "left"
            self.model = OPTForCausalLM.from_pretrained(model_name)
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.max_token_length = max_token_length

        self.tokenizer_type = tokenizer_type
        self.texts = texts
        self.labels = labels

        self.encoder_layer_depth = encoder_layer_depth
        if self.encoder_layer_depth != -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = device
            self.model = self.model.to(self.device)
        if data_name == 'cose':
            if graph_process_type == 'direct_root':
                self.processed_data = [self._process_text_direct_root_word_query(text) for text in texts]

        elif data_name == 'graph_sst2':
            embed_path = "./data/Graph-SST2/raw/Graph-SST2_node_features.pkl"
            edge_path = "/storage/personal/myhwang/NLP_FS/data/Graph-SST2/raw/Graph-SST2_edge_index.txt"
            
            node_indicator_path = "./data/Graph-SST2/raw/Graph-SST2_node_indicator.txt"
            
            with open(embed_path, "rb") as f:
                all_node_embeddings = pickle.load(f)
            self.sample_idx=kargs['idx']
            
            self.sem_embedding_matrix_full = torch.tensor(np.array(all_node_embeddings), dtype=torch.float)

            self.sem_edge_index = np.loadtxt(edge_path, dtype=int).T
            self.node_indicator = np.loadtxt(node_indicator_path, dtype=int)
            self.processed_data = [self._process_text_direct_root_word_semantic(text, self.sample_idx[i], load=False) for i , text in enumerate(texts)]
        elif data_name == 'bioasq':
            if graph_process_type == 'direct_root':
                
                if tokenizer_type == 'BioMedLM':
                    self.processed_data = [self._process_text_direct_root_word_query_last(text) for text in texts]
                elif tokenizer_type == 'biolinkBert':
                    self.processed_data = [self._process_text_direct_root_word_query(text) for text in texts]
                elif tokenizer_type == 'galactica':
                    self.processed_data = [self._process_text_direct_root_word_prompt_query_last(text) for text in texts]
        else:
            if graph_process_type == 'direct_root':
                self.processed_data = [self._process_text_direct_root_word(text) for text in texts]


    def _process_text_direct_root_word_semantic(self, text, sample_idx=None, load=True):
        
        doc = self.nlp(text)
        all_word_nodes = []
        raw_word_list = []
        offset = 0

        for sent in doc.sents:
            norm_sent_text = normalize(sent.text)
            word_spans = norm_sent_text.split()
            word_spans = split_final_punctuation(word_spans)
            raw_word_list.extend(word_spans)
            span_to_token_idxs = defaultdict(list)
            running_offset = 0

            for i, word in enumerate(word_spans):
                norm_word = normalize(word)
                word_start_in_sent = norm_sent_text.find(norm_word, running_offset)
                word_start = sent.start_char + word_start_in_sent
                word_end = word_start + len(word)
                running_offset = word_start_in_sent + len(word)

                for j, token in enumerate(sent):
                    token_start = token.idx
                    token_end = token.idx + len(token)

                    if token_start >= word_start and token_end <= word_end:
                        span_to_token_idxs[i].append(j)

            word_nodes = []
            for span_idx, token_idxs in span_to_token_idxs.items():
                words = [sent[i].text for i in token_idxs]
                deps = [sent[i].dep_ for i in token_idxs]
                head_tokens = [sent[i].head for i in token_idxs]

                head_span = None
                for head in head_tokens:
                    for k, v in span_to_token_idxs.items():
                        if head.i == sent[v[0]].i:
                            head_span = k
                            break
                    if head_span is not None:
                        break

                word_nodes.append({
                    'text': ' '.join(words),
                    'dep': deps,
                    'head_pos': head_span + offset if head_span is not None else None,
                    'head_text': word_spans[head_span] if head_span is not None else 'ROOT',
                    'position': span_idx + offset
                })

            all_word_nodes.extend(word_nodes)
            offset += len(word_spans)
        print('raw_word_lis:',raw_word_list)
        G_full = nx.Graph()

        for node in all_word_nodes:
            G_full.add_node(node['position'], word=node['text'], dep=node['dep'])

        for node in all_word_nodes:
            if node['head_pos'] is not None and node['head_pos'] != node['position']:
                G_full.add_edge(
                    node['position'],
                    node['head_pos'],
                    label=node['dep'] 
                )
        root_positions = [node['position'] for node in all_word_nodes if 'ROOT' in node['dep']]

        for i in range(len(root_positions)-1):
            G_full.add_edge(
                root_positions[i],
                root_positions[i+1],
                label='root_connection'
            )
            expected_nodes = set(range(len(all_word_nodes)))
            actual_nodes = set(G_full.nodes)

        expected_nodes = set(range(len(all_word_nodes)))
        actual_nodes = set(G_full.nodes)

        missing_nodes = expected_nodes - actual_nodes
        if missing_nodes:
            print(f"[❗] Missing nodes in G_full: {missing_nodes}")
            for node in missing_nodes:
                G_full.add_node(
                    node,
                    word=raw_word_list[node] if node < len(raw_word_list) else 'MISSING',
                    xlnet_tokens=[],
                    dep=[]
                )

        token_ids = self.tokenizer(text, return_tensors='pt',truncation=True, max_length=self.max_token_length,)

        tokenized_tokens = self.tokenizer.convert_ids_to_tokens(token_ids['input_ids'][0])


        max_shift_attempts = 5
        node2token = {}
        char_pointer = 0
        subword_pointer = 0
        
        special_tokens = set(self.tokenizer.all_special_tokens)
        print(all_word_nodes)
        print(raw_word_list)

        for node in all_word_nodes:
            word = node['text']
            word_start = text.lower().find(word.lower(), char_pointer)
            word_end = word_start + len(word)
            char_pointer = word_end
            print('node word:', word)

            compare_word = ''.join(c for c in ascii_only(word.lower()) if c.isalnum())
            matched = False

            attempt_count = 0
            start_pointer = subword_pointer

            while start_pointer < len(tokenized_tokens) and attempt_count < max_shift_attempts:
                
                if tokenized_tokens[start_pointer] in special_tokens:
                    start_pointer += 1
                    continue

                temp_token_idxs = []
                token_text_acc = ''

                for i in range(start_pointer, len(tokenized_tokens)):
                    token = tokenized_tokens[i]
                    normalized = token.lstrip("Ġ▁#").lower()

                    token_text_acc += ascii_only(normalized)
                    temp_token_idxs.append(i)

                    compare_token = ''.join(c for c in ascii_only(token_text_acc) if c.isalnum())

                    if compare_token == compare_word:
                        node2token[node['position']] = temp_token_idxs
                        subword_pointer = i + 1 
                        matched = True
                        break
                    elif len(compare_token) > len(compare_word):
                        break

                if matched:
                    break
                else:
                    start_pointer += 1
                    attempt_count += 1

            if not matched:
                node2token[node['position']] = []
                print(f"[WARN] Failed to match word '{word}' (node {node['position']})")



        print(f"node2token: {node2token}")

        embedding_matrix_full = None
        G_external = None
        graph_label = None
        
        mask = torch.tensor(self.node_indicator == (sample_idx + 1))
        total_nodes = mask.sum()
        sample_node_indices = torch.where(mask)[0]

        src_nodes = self.sem_edge_index[0]
        dst_nodes = self.sem_edge_index[1]

        valid_src = mask[src_nodes]
        valid_dst = mask[dst_nodes]
        sample_mask = valid_src & valid_dst
        selected_edges = self.sem_edge_index[:, sample_mask]

        sample_node_indices = torch.where(mask)[0]  # tensor of positions
        index_remap = {old.item(): new for new, old in enumerate(sample_node_indices)}

        src_selected = selected_edges[0]
        dst_selected = selected_edges[1]

        try:
            src_remapped = torch.tensor([index_remap[int(s)] for s in src_selected])
            dst_remapped = torch.tensor([index_remap[d.item()] for d in dst_selected])
        except KeyError as e:
            raise ValueError(f"[❗] remapping error: node {e}")

        edge_index_remapped = torch.stack([src_remapped, dst_remapped], dim=0)

        embedding_matrix_full = torch.tensor(
            np.array(self.sem_embedding_matrix_full[mask]), dtype=torch.float
        )

        G_external = nx.Graph()
        for i in range(edge_index_remapped.shape[1]):
            src = int(edge_index_remapped[0, i])
            dst = int(edge_index_remapped[1, i])
            G_external.add_edge(src, dst)

        if not load:
            if self.encoder_layer_depth == -1:
                with torch.no_grad():
                    input_embeddings = self.model.get_input_embeddings()(token_ids['input_ids'])
                    token_embeddings = input_embeddings.squeeze(0)  # [seq_len, hidden_dim]
            else:
                
                self.model.eval()
                special_token_ids = set(self.tokenizer.all_special_ids)
                with torch.no_grad():
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_token_length)
                    inputs = inputs.to(self.device)
                    outputs = self.model(**inputs, output_hidden_states=True)
                    token_embeddings = outputs.hidden_states[self.encoder_layer_depth].squeeze(0)
                    

            embedding_dim = token_embeddings.shape[1]
            total_nodes = max(node2token.keys())+1
            embedding_matrix_full = torch.full((total_nodes, embedding_dim), np.nan)

            max_len = token_embeddings.shape[0]
            for node_pos in range(total_nodes):
                token_indices = node2token.get(node_pos, [])
                token_indices = [i for i in token_indices if i < max_len]
                if token_indices:
                    node_embedding = torch.mean(token_embeddings[token_indices], dim=0)
                    embedding_matrix_full[node_pos] = node_embedding
                else:
                    print(f"[WARN] No valid subword tokens for node {node['position']}")

        return {
            'text': text,
            'token_info': all_word_nodes,
            'graph': G_full,
            'external_graph': G_external,
            'tokens': tokenized_tokens,
            'token_ids': token_ids,
            'node2token': node2token,
            'max_sent_len': total_nodes,
            'root_positions': root_positions,
            'embedding_matrix_full': embedding_matrix_full,
            'raw_word': raw_word_list,
        }


    def _process_text_direct_root_word_prompt_query_last(self, text):

        main_sentence = text["context"]
        query = text["question"]

        doc = self.nlp(main_sentence)

        all_word_nodes = []
        raw_word_list=[]
        offset = 0 
        for sent in doc.sents:
            sent_text = sent.text
            norm_sent_text = normalize(sent_text)
            word_spans = norm_sent_text.split() 
            word_spans = split_final_punctuation(word_spans)
            raw_word_list.extend(word_spans) 
            span_to_token_idxs = defaultdict(list)

            running_offset = 0

            for i, word in enumerate(word_spans):
                norm_word = normalize(word)
                word_start_in_sent = norm_sent_text.find(norm_word, running_offset)
                word_start = sent.start_char + word_start_in_sent
                word_end = word_start + len(word)
                running_offset = word_start_in_sent + len(word)

                for j, token in enumerate(sent):
                    token_start = token.idx
                    token_end = token.idx + len(token)

                    if token_start >= word_start and token_end <= word_end:
                        span_to_token_idxs[i].append(j)

            word_nodes = []
            for span_idx, token_idxs in span_to_token_idxs.items():
                words = [sent[i].text for i in token_idxs]
                deps = [sent[i].dep_ for i in token_idxs]
                head_tokens = [sent[i].head for i in token_idxs]

                head_span = None
                for head in head_tokens:
                    for k, v in span_to_token_idxs.items():
                        if head.i == sent[v[0]].i:
                            head_span = k
                            break
                    if head_span is not None:
                        break

                word_nodes.append({
                    'text': ' '.join(words),
                    'dep': deps,
                    'head_pos': head_span + offset if head_span is not None else None,
                    'head_text': word_spans[head_span] if head_span is not None else 'ROOT',
                    'position': span_idx + offset
                })

            all_word_nodes.extend(word_nodes)
            offset += len(word_spans)
        print('raw_word_lis:',raw_word_list)
        G_full = nx.Graph()

        for node in all_word_nodes:
            G_full.add_node(node['position'], word=node['text'], dep=node['dep'])

        for node in all_word_nodes:
            if node['head_pos'] is not None and node['head_pos'] != node['position']:
                G_full.add_edge(
                    node['position'],
                    node['head_pos'],
                    label=node['dep'] 
                )
        root_positions = [node['position'] for node in all_word_nodes if 'ROOT' in node['dep']]

        for i in range(len(root_positions)-1):
            G_full.add_edge(
                root_positions[i],
                root_positions[i+1],
                label='root_connection'
            )
        expected_nodes = set(range(len(all_word_nodes)))
        actual_nodes = set(G_full.nodes)

        missing_nodes = expected_nodes - actual_nodes
        if missing_nodes:
            print(f"[❗] Missing nodes in G_full: {missing_nodes}")
            for node in missing_nodes:
                G_full.add_node(
                    node,
                    word=raw_word_list[node] if node < len(raw_word_list) else 'MISSING',
                    xlnet_tokens=[],
                    dep=[]
                )

        instructions = f"You are a biomedical question‑answering agent. Output exactly 'yes' or 'no'. Q: {query.strip()} A:"
        
        encodings = self.tokenizer(
            main_sentence,
            text_pair=instructions,
            padding='max_length',
            truncation="only_first", 
            max_length=512,
            return_tensors=None,
            add_special_tokens=True
        )

        def pad_to_fixed_length(sequences, padding_value, max_length, padding_side='right'):
            """
            sequences: list of 1D torch tensors
            padding_value: int (tokenizer.pad_token_id)
            max_length: int
            padding_side: 'right' or 'left'
            """
            result = []
            for seq in sequences:
                seq_len = len(seq)
                if seq_len > max_length:
                    if padding_side == 'right':
                        seq = seq[:max_length]
                    else:  # left
                        seq = seq[-max_length:]
                pad_len = max_length - len(seq)
                pad_tensor = torch.full((pad_len,), padding_value, dtype=seq.dtype)
                if padding_side == 'right':
                    padded_seq = torch.cat([seq, pad_tensor])
                else:  # left
                    padded_seq = torch.cat([pad_tensor, seq])
                result.append(padded_seq)
            return torch.stack(result)
        

        input_ids = torch.tensor(encodings["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encodings["attention_mask"], dtype=torch.long)

        input_ids_padded = pad_to_fixed_length([input_ids], padding_value=self.tokenizer.pad_token_id, max_length=512, padding_side='left')
        attention_mask_padded = pad_to_fixed_length([attention_mask], padding_value=0, max_length=512, padding_side='left')

        token_ids = {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded
        }

        tokenized_tokens = self.tokenizer.convert_ids_to_tokens(token_ids['input_ids'][0])

        max_shift_attempts = 5
        node2token = {}
        char_pointer = 0

        subword_pointer =0
        special_tokens = set(self.tokenizer.all_special_tokens)

        for node in all_word_nodes:
            word = node['text']
            word_start = text['context'].lower().find(word.lower(), char_pointer)
            word_end = word_start + len(word)
            char_pointer = word_end
            print('node word:', word)

            compare_word = ''.join(c for c in ascii_only(word.lower()) if c.isalnum())
            matched = False

            attempt_count = 0
            start_pointer = subword_pointer

            while start_pointer < len(tokenized_tokens) and attempt_count < max_shift_attempts:
                
                if tokenized_tokens[start_pointer] in special_tokens:
                    start_pointer += 1
                    continue

                temp_token_idxs = []
                token_text_acc = ''

                for i in range(start_pointer, len(tokenized_tokens)):
                    token = tokenized_tokens[i]
                    normalized = token.lstrip("Ġ▁#").lower()

                    token_text_acc += ascii_only(normalized)
                    temp_token_idxs.append(i)

                    compare_token = ''.join(c for c in ascii_only(token_text_acc) if c.isalnum())

                    if compare_token == compare_word:
                        node2token[node['position']] = temp_token_idxs
                        subword_pointer = i + 1 
                        matched = True
                        break
                    elif len(compare_token) > len(compare_word):
                        break

                if matched:
                    break
                else:
                    start_pointer += 1
                    attempt_count += 1

            if not matched:
                node2token[node['position']] = []
                print(f"[WARN] Failed to match word '{word}' (node {node['position']})")

        print(f"node2token: {node2token}")

        if self.encoder_layer_depth == -1:
            with torch.no_grad():
                input_embeddings = self.model.get_input_embeddings()(token_ids['input_ids'])
                token_embeddings = input_embeddings.squeeze(0)  # [seq_len, hidden_dim]
        else:
            
            self.model.eval()
            with torch.no_grad():

        
                instructions = f"You are a biomedical question‑answering agent. Output exactly 'yes' or 'no'. Q: {query.strip()} A:"

                inputs = self.tokenizer(
                    main_sentence,
                    text_pair=instructions,
                    padding=True,
                    truncation="only_first", 
                    max_length=512,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
                outputs = self.model(**inputs, output_hidden_states=True)
                token_embeddings = outputs.hidden_states[self.encoder_layer_depth].squeeze(0)
                

        embedding_dim = token_embeddings.shape[1]
        total_nodes = max(node2token.keys())+1  
        embedding_matrix_full = torch.full((total_nodes, embedding_dim), np.nan)

        max_len = token_embeddings.shape[0]

        for node_pos in range(total_nodes):
            token_indices = node2token.get(node_pos, [])
            token_indices = [i for i in token_indices if i < max_len]
            if token_indices:
                node_embedding = torch.mean(token_embeddings[token_indices], dim=0)
                embedding_matrix_full[node_pos] = node_embedding
            else:
                print(f"[WARN] No valid subword tokens for node {node['position']}")

        return {
            'text': text,
            'token_info': all_word_nodes,
            'graph': G_full,
            # 'adj_matrix_full': adj_matrix_full,
            'tokens': tokenized_tokens,
            'token_ids': token_ids,
            'node2token': node2token,
            'max_sent_len': len(all_word_nodes),
            # 'sent_lengths': [len(sent) for sent in sentences_token_info],
            'root_positions': root_positions,
            'embedding_matrix_full': embedding_matrix_full,
            'raw_word': raw_word_list,
        }

    def _process_text_direct_root_word_query_last(self, text):

        main_sentence = text["context"]
        query = text["question"]

        doc = self.nlp(main_sentence)

        all_word_nodes = []
        raw_word_list=[]
        offset = 0  

        for sent in doc.sents:
            sent_text = sent.text
            norm_sent_text = normalize(sent_text)
            word_spans = norm_sent_text.split()
            word_spans = split_final_punctuation(word_spans)
            raw_word_list.extend(word_spans) 
            span_to_token_idxs = defaultdict(list)

            running_offset = 0 

            for i, word in enumerate(word_spans):
                norm_word = normalize(word)
                word_start_in_sent = norm_sent_text.find(norm_word, running_offset)
                word_start = sent.start_char + word_start_in_sent
                word_end = word_start + len(word)
                running_offset = word_start_in_sent + len(word)

                for j, token in enumerate(sent):
                    token_start = token.idx
                    token_end = token.idx + len(token)

                    if token_start >= word_start and token_end <= word_end:
                        span_to_token_idxs[i].append(j)

            word_nodes = []
            for span_idx, token_idxs in span_to_token_idxs.items():
                words = [sent[i].text for i in token_idxs]
                deps = [sent[i].dep_ for i in token_idxs]
                head_tokens = [sent[i].head for i in token_idxs]

                head_span = None
                for head in head_tokens:
                    for k, v in span_to_token_idxs.items():
                        if head.i == sent[v[0]].i:
                            head_span = k
                            break
                    if head_span is not None:
                        break

                word_nodes.append({
                    'text': ' '.join(words),
                    'dep': deps,
                    'head_pos': head_span + offset if head_span is not None else None,
                    'head_text': word_spans[head_span] if head_span is not None else 'ROOT',
                    'position': span_idx + offset
                })

            all_word_nodes.extend(word_nodes)
            offset += len(word_spans)
        print('raw_word_lis:',raw_word_list)
        G_full = nx.Graph()

        for node in all_word_nodes:
            G_full.add_node(node['position'], word=node['text'], dep=node['dep'])

        for node in all_word_nodes:
            if node['head_pos'] is not None and node['head_pos'] != node['position']:
                G_full.add_edge(
                    node['position'],
                    node['head_pos'],
                    label=node['dep'] 
                )
        root_positions = [node['position'] for node in all_word_nodes if 'ROOT' in node['dep']]

        for i in range(len(root_positions)-1):
            G_full.add_edge(
                root_positions[i],
                root_positions[i+1],
                label='root_connection'
            )
        expected_nodes = set(range(len(all_word_nodes)))
        actual_nodes = set(G_full.nodes)

        missing_nodes = expected_nodes - actual_nodes
        if missing_nodes:
            print(f"[❗] Missing nodes in G_full: {missing_nodes}")
            for node in missing_nodes:
                G_full.add_node(
                    node,
                    word=raw_word_list[node] if node < len(raw_word_list) else 'MISSING',
                    xlnet_tokens=[],
                    dep=[]
                )


        token_ids = self.tokenizer(
            f"[CONTEXT] {main_sentence.strip()}",
            text_pair=f"[QUESTION] {query} [ANSWER]",
            padding=True,
            truncation="only_first",
            max_length=512,
            return_tensors="pt"
        )

        tokenized_tokens = self.tokenizer.convert_ids_to_tokens(token_ids['input_ids'][0])  
        max_shift_attempts = 5  
        node2token = {}
        char_pointer = 0

        subword_pointer =1 # [CONTEXT]
        special_tokens = set(self.tokenizer.all_special_tokens)

        for node in all_word_nodes:
            word = node['text']
            word_start = text['context'].lower().find(word.lower(), char_pointer)
            word_end = word_start + len(word)
            char_pointer = word_end
            print('node word:', word)

            compare_word = ''.join(c for c in ascii_only(word.lower()) if c.isalnum())
            matched = False

            attempt_count = 0
            start_pointer = subword_pointer

            while start_pointer < len(tokenized_tokens) and attempt_count < max_shift_attempts:
                
                if tokenized_tokens[start_pointer] in special_tokens:
                    start_pointer += 1
                    continue

                temp_token_idxs = []
                token_text_acc = ''

                for i in range(start_pointer, len(tokenized_tokens)):
                    token = tokenized_tokens[i]
                    normalized = token.lstrip("Ġ▁#").lower()

                    token_text_acc += ascii_only(normalized)
                    temp_token_idxs.append(i)

                    compare_token = ''.join(c for c in ascii_only(token_text_acc) if c.isalnum())

                    if compare_token == compare_word:
                        node2token[node['position']] = temp_token_idxs
                        subword_pointer = i + 1 
                        matched = True
                        break
                    elif len(compare_token) > len(compare_word):
                        break  

                if matched:
                    break
                else:
                    start_pointer += 1
                    attempt_count += 1

            if not matched:
                node2token[node['position']] = []
                print(f"[WARN] Failed to match word '{word}' (node {node['position']})")

        print(f"node2token: {node2token}")


        if self.encoder_layer_depth == -1:

            with torch.no_grad():
                input_embeddings = self.model.get_input_embeddings()(token_ids['input_ids'])
                token_embeddings = input_embeddings.squeeze(0)  # [seq_len, hidden_dim]
        else:
            
            self.model.eval()
            with torch.no_grad():
                inputs = self.tokenizer(
                    f"[CONTEXT] {main_sentence.strip()}",
                    text_pair=f"[QUESTION] {query} [ANSWER]",
                    padding=True,
                    truncation="only_first",
                    max_length=512,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
                outputs = self.model(**inputs, output_hidden_states=True)
                token_embeddings = outputs.hidden_states[self.encoder_layer_depth].squeeze(0)
                

        embedding_dim = token_embeddings.shape[1]
        total_nodes = max(node2token.keys())+1  
        embedding_matrix_full = torch.full((total_nodes, embedding_dim), np.nan)

        max_len = token_embeddings.shape[0]

        for node_pos in range(total_nodes):
            token_indices = node2token.get(node_pos, [])
            token_indices = [i for i in token_indices if i < max_len]
            if token_indices:
                node_embedding = torch.mean(token_embeddings[token_indices], dim=0)
                embedding_matrix_full[node_pos] = node_embedding
            else:
                print(f"[WARN] No valid subword tokens for node {node['position']}")

        return {
            'text': text,
            'token_info': all_word_nodes,
            'graph': G_full,
            # 'adj_matrix_full': adj_matrix_full,
            'tokens': tokenized_tokens,
            'token_ids': token_ids,
            'node2token': node2token,
            'max_sent_len': len(all_word_nodes),
            # 'sent_lengths': [len(sent) for sent in sentences_token_info],
            'root_positions': root_positions,
            'embedding_matrix_full': embedding_matrix_full,
            'raw_word': raw_word_list,
        }

    def _process_text_direct_root_word_query(self, text):

        parts = text.split("[SEP]", 1)
        query = parts[0].strip()
        main_sentence = parts[1].strip()

        doc = self.nlp(main_sentence)

        all_word_nodes = []
        raw_word_list=[]
        offset = 0 

        for sent in doc.sents:
            sent_text = sent.text
            norm_sent_text = normalize(sent_text)
            word_spans = norm_sent_text.split()  
            word_spans = split_final_punctuation(word_spans)
            raw_word_list.extend(word_spans)
            span_to_token_idxs = defaultdict(list)

            running_offset = 0

            for i, word in enumerate(word_spans):
                norm_word = normalize(word)
                word_start_in_sent = norm_sent_text.find(norm_word, running_offset)
                word_start = sent.start_char + word_start_in_sent
                word_end = word_start + len(word)
                running_offset = word_start_in_sent + len(word)

                for j, token in enumerate(sent):
                    token_start = token.idx
                    token_end = token.idx + len(token)

                    if token_start >= word_start and token_end <= word_end:
                        span_to_token_idxs[i].append(j)

            word_nodes = []
            for span_idx, token_idxs in span_to_token_idxs.items():
                words = [sent[i].text for i in token_idxs]
                deps = [sent[i].dep_ for i in token_idxs]
                head_tokens = [sent[i].head for i in token_idxs]

                head_span = None
                for head in head_tokens:
                    for k, v in span_to_token_idxs.items():
                        if head.i == sent[v[0]].i:
                            head_span = k
                            break
                    if head_span is not None:
                        break

                word_nodes.append({
                    'text': ' '.join(words),
                    'dep': deps,
                    'head_pos': head_span + offset if head_span is not None else None,
                    'head_text': word_spans[head_span] if head_span is not None else 'ROOT',
                    'position': span_idx + offset
                })

            all_word_nodes.extend(word_nodes)
            offset += len(word_spans)
        print('raw_word_lis:',raw_word_list)
        G_full = nx.Graph()

        for node in all_word_nodes:
            G_full.add_node(node['position'], word=node['text'], dep=node['dep'])

        for node in all_word_nodes:
            if node['head_pos'] is not None and node['head_pos'] != node['position']:
                G_full.add_edge(
                    node['position'],
                    node['head_pos'],
                    label=node['dep'] 
                )
        root_positions = [node['position'] for node in all_word_nodes if 'ROOT' in node['dep']]

        for i in range(len(root_positions)-1):
            G_full.add_edge(
                root_positions[i],
                root_positions[i+1],
                label='root_connection'
            )
        expected_nodes = set(range(len(all_word_nodes)))
        actual_nodes = set(G_full.nodes)

        missing_nodes = expected_nodes - actual_nodes
        if missing_nodes:
            print(f"[❗] Missing nodes in G_full: {missing_nodes}")
            for node in missing_nodes:
                G_full.add_node(
                    node,
                    word=raw_word_list[node] if node < len(raw_word_list) else 'MISSING',
                    xlnet_tokens=[],
                    dep=[]
                )


        sep_token = self.tokenizer.sep_token 
        text = text.replace("[SEP]", sep_token)
        token_ids = self.tokenizer(text, return_tensors='pt',truncation=True, max_length=self.max_token_length,)

        tokenized_tokens = self.tokenizer.convert_ids_to_tokens(token_ids['input_ids'][0])  

        max_shift_attempts = 5 
        node2token = {}
        char_pointer = 0
        sep_token_id = self.tokenizer.sep_token_id
        sep_idx = token_ids['input_ids'][0].tolist().index(sep_token_id)
        subword_pointer = sep_idx + 1
        special_tokens = set(self.tokenizer.all_special_tokens)
        print(all_word_nodes)
        print(raw_word_list)

        for node in all_word_nodes:
            word = node['text']
            word_start = text.lower().find(word.lower(), char_pointer)
            word_end = word_start + len(word)
            char_pointer = word_end
            print('node word:', word)

            compare_word = ''.join(c for c in ascii_only(word.lower()) if c.isalnum())
            matched = False

            attempt_count = 0
            start_pointer = subword_pointer

            while start_pointer < len(tokenized_tokens) and attempt_count < max_shift_attempts:
                
                if tokenized_tokens[start_pointer] in special_tokens:
                    start_pointer += 1
                    continue

                temp_token_idxs = []
                token_text_acc = ''

                for i in range(start_pointer, len(tokenized_tokens)):
                    token = tokenized_tokens[i]
                    normalized = token.lstrip("Ġ▁#").lower()

                    token_text_acc += ascii_only(normalized)
                    temp_token_idxs.append(i)

                    compare_token = ''.join(c for c in ascii_only(token_text_acc) if c.isalnum())

                    if compare_token == compare_word:
                        node2token[node['position']] = temp_token_idxs
                        subword_pointer = i + 1 
                        matched = True
                        break
                    elif len(compare_token) > len(compare_word):
                        break 

                if matched:
                    break
                else:
                    start_pointer += 1
                    attempt_count += 1

            if not matched:
                node2token[node['position']] = []
                print(f"[WARN] Failed to match word '{word}' (node {node['position']})")

        print(f"node2token: {node2token}")


        if self.encoder_layer_depth == -1:

            with torch.no_grad():
                input_embeddings = self.model.get_input_embeddings()(token_ids['input_ids'])
                token_embeddings = input_embeddings.squeeze(0)  # [seq_len, hidden_dim]
        else:
            
            self.model.eval()
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_token_length)
                inputs = inputs.to(self.device)
                outputs = self.model(**inputs, output_hidden_states=True)
                token_embeddings = outputs.hidden_states[self.encoder_layer_depth].squeeze(0)

        embedding_dim = token_embeddings.shape[1]
        total_nodes = max(node2token.keys())+1
        embedding_matrix_full = torch.full((total_nodes, embedding_dim), np.nan)

        max_len = token_embeddings.shape[0]

        for node_pos in range(total_nodes):
            token_indices = node2token.get(node_pos, [])
            token_indices = [i for i in token_indices if i < max_len]
            if token_indices:
                node_embedding = torch.mean(token_embeddings[token_indices], dim=0)
                embedding_matrix_full[node_pos] = node_embedding
            else:
                print(f"[WARN] No valid subword tokens for node {node['position']}")

        return {
            'text': text,
            'token_info': all_word_nodes,
            'graph': G_full,
            # 'adj_matrix_full': adj_matrix_full,
            'tokens': tokenized_tokens,
            'token_ids': token_ids,
            'node2token': node2token,
            'max_sent_len': len(all_word_nodes),
            # 'sent_lengths': [len(sent) for sent in sentences_token_info],
            'root_positions': root_positions,
            'embedding_matrix_full': embedding_matrix_full,
            'raw_word': raw_word_list,
        }


    def _process_text_direct_root_word(self, text):

        doc = self.nlp(text)

        all_word_nodes = []
        raw_word_list=[]
        offset = 0 

        for sent in doc.sents:
            sent_text = sent.text
            norm_sent_text = normalize(sent_text)
            word_spans = norm_sent_text.split() 
            word_spans = split_final_punctuation(word_spans)
            raw_word_list.extend(word_spans) 
            span_to_token_idxs = defaultdict(list)

            running_offset = 0 

            for i, word in enumerate(word_spans):
                norm_word = normalize(word)
                word_start_in_sent = norm_sent_text.find(norm_word, running_offset)
                word_start = sent.start_char + word_start_in_sent
                word_end = word_start + len(word)
                running_offset = word_start_in_sent + len(word) 

                for j, token in enumerate(sent):
                    token_start = token.idx
                    token_end = token.idx + len(token)

                    if token_start >= word_start and token_end <= word_end:
                        span_to_token_idxs[i].append(j)

            word_nodes = []
            for span_idx, token_idxs in span_to_token_idxs.items():
                words = [sent[i].text for i in token_idxs]
                deps = [sent[i].dep_ for i in token_idxs]
                head_tokens = [sent[i].head for i in token_idxs]

                head_span = None
                for head in head_tokens:
                    for k, v in span_to_token_idxs.items():
                        if head.i == sent[v[0]].i:
                            head_span = k
                            break
                    if head_span is not None:
                        break

                word_nodes.append({
                    'text': ' '.join(words),
                    'dep': deps,
                    'head_pos': head_span + offset if head_span is not None else None,
                    'head_text': word_spans[head_span] if head_span is not None else 'ROOT',
                    'position': span_idx + offset
                })

            all_word_nodes.extend(word_nodes)
            offset += len(word_spans)
        print('raw_word_lis:',raw_word_list)
        G_full = nx.Graph()

        for node in all_word_nodes:
            G_full.add_node(node['position'], word=node['text'], dep=node['dep'])

        for node in all_word_nodes:
            if node['head_pos'] is not None and node['head_pos'] != node['position']:
                G_full.add_edge(
                    node['position'],
                    node['head_pos'],
                    label=node['dep'] 
                )
        root_positions = [node['position'] for node in all_word_nodes if 'ROOT' in node['dep']]

        for i in range(len(root_positions)-1):
            G_full.add_edge(
                root_positions[i],
                root_positions[i+1],
                label='root_connection'
            )
        expected_nodes = set(range(len(all_word_nodes)))
        actual_nodes = set(G_full.nodes)

        missing_nodes = expected_nodes - actual_nodes
        if missing_nodes:
            print(f"[❗] Missing nodes in G_full: {missing_nodes}")
            for node in missing_nodes:
                G_full.add_node(
                    node,
                    word=raw_word_list[node] if node < len(raw_word_list) else 'MISSING',
                    xlnet_tokens=[],
                    dep=[]
                )

        token_ids = self.tokenizer(text, return_tensors='pt',truncation=True, max_length=self.max_token_length,)

        tokenized_tokens = self.tokenizer.convert_ids_to_tokens(token_ids['input_ids'][0])  

        max_shift_attempts = 5
        node2token = {}
        char_pointer = 0
        subword_pointer = 0
        
        special_tokens = set(self.tokenizer.all_special_tokens)

        for node in all_word_nodes:
            word = node['text']
            word_start = text.lower().find(word.lower(), char_pointer)
            word_end = word_start + len(word)
            char_pointer = word_end
            print('node word:', word)

            compare_word = ''.join(c for c in ascii_only(word.lower()) if c.isalnum())
            matched = False

            attempt_count = 0
            start_pointer = subword_pointer

            while start_pointer < len(tokenized_tokens) and attempt_count < max_shift_attempts:
                
                if tokenized_tokens[start_pointer] in special_tokens:
                    start_pointer += 1
                    continue

                temp_token_idxs = []
                token_text_acc = ''

                for i in range(start_pointer, len(tokenized_tokens)):
                    token = tokenized_tokens[i]
                    normalized = token.lstrip("Ġ▁#").lower()

                    token_text_acc += ascii_only(normalized)
                    temp_token_idxs.append(i)

                    compare_token = ''.join(c for c in ascii_only(token_text_acc) if c.isalnum())

                    if compare_token == compare_word:
                        node2token[node['position']] = temp_token_idxs
                        subword_pointer = i + 1  
                        matched = True
                        break
                    elif len(compare_token) > len(compare_word):
                        break  

                if matched:
                    break
                else:
                    start_pointer += 1
                    attempt_count += 1

            if not matched:
                node2token[node['position']] = []

        if self.encoder_layer_depth == -1:

            with torch.no_grad():
                input_embeddings = self.model.get_input_embeddings()(token_ids['input_ids'])
                token_embeddings = input_embeddings.squeeze(0)  # [seq_len, hidden_dim]
        else:
            
            self.model.eval()
            special_token_ids = set(self.tokenizer.all_special_ids)
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_token_length)
                inputs = inputs.to(self.device)
                outputs = self.model(**inputs, output_hidden_states=True)
                
                token_embeddings = outputs.hidden_states[self.encoder_layer_depth].squeeze(0)
                

        embedding_dim = token_embeddings.shape[1]
        total_nodes = max(node2token.keys())+1 
        embedding_matrix_full = torch.full((total_nodes, embedding_dim), np.nan)

        max_len = token_embeddings.shape[0]

        for node_pos in range(total_nodes):
            token_indices = node2token.get(node_pos, [])
            token_indices = [i for i in token_indices if i < max_len]
            if token_indices:
                node_embedding = torch.mean(token_embeddings[token_indices], dim=0)
                embedding_matrix_full[node_pos] = node_embedding
            else:
                print(f"[WARN] No valid subword tokens for node {node['position']}")

        return {
            'text': text,
            'token_info': all_word_nodes,
            'graph': G_full,
            # 'adj_matrix_full': adj_matrix_full,
            'tokens': tokenized_tokens,
            'token_ids': token_ids,
            'node2token': node2token,
            'max_sent_len': len(all_word_nodes),
            # 'sent_lengths': [len(sent) for sent in sentences_token_info],
            'root_positions': root_positions,
            'embedding_matrix_full': embedding_matrix_full,
            'raw_word': raw_word_list,
        }


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.processed_data[idx]
        processed_item = {
            'text': item['text'],
            'token_info': item['token_info'],
            # 'sentences_token_info': item['sentences_token_info'],
            'graph': item['graph'],
            # 'sentence_graphs': item['sentence_graphs'],
            # 'adj_matrix_full': torch.FloatTensor(item['adj_matrix_full']),
            # 'adj_matrix_per_sent': torch.FloatTensor(item['adj_matrix_per_sent']),
            'tokens': item['tokens'],
            'token_ids': {k: v for k, v in item['token_ids'].items()},
            'node2token': item['node2token'],
            'max_sent_len': item['max_sent_len'],
            # 'sent_lengths': item['sent_lengths'],
            'root_positions': item['root_positions'],
            'embedding_matrix_full': item['embedding_matrix_full'],
            # 'embedding_matrix_per_sent': item['embedding_matrix_per_sent'],
            'raw_word': item['raw_word'],
        }

        if 'dummy_positions' in item:
            processed_item['dummy_positions'] = item['dummy_positions']
        
        if 'external_graph' in item:
            processed_item['external_graph'] = item['external_graph']

        if self.labels is not None:
            processed_item['label'] = torch.LongTensor([self.labels[idx]])
        
        return processed_item


def collate_fn_pickle(batch):

    batch_dict = {
        'text': [item['text'] for item in batch],
        'token_info': [item['token_info'] for item in batch],
        # 'sentences_token_info': [item['sentences_token_info'] for item in batch],
        'graph': [item['graph'] for item in batch],
        # 'sentence_graphs': [item['sentence_graphs'] for item in batch],
        'tokens': [item['tokens'] for item in batch],
        'node2token': [item['node2token'] for item in batch],
        'max_sent_len': [item['max_sent_len'] for item in batch],
        # 'sent_lengths': [item['sent_lengths'] for item in batch],
        'root_positions': [item['root_positions'] for item in batch],
        'label': [item['label'] for item in batch],
        'raw_word': [item['raw_word'] for item in batch],
    }

    if 'dummy_positions' in batch[0]:
        batch_dict['dummy_positions'] = [item['dummy_positions'] for item in batch]

    if 'external_graph' in batch[0]:
        batch_dict['external_graph'] = [item['external_graph'] for item in batch]

    max_nodes = max(item['embedding_matrix_full'].shape[0] for item in batch)

    padded_emb = []
    for item in batch:
        emb = item['embedding_matrix_full']
        pad_size = max_nodes - emb.shape[0]
        if pad_size > 0:
            padded = torch.nn.functional.pad(emb, (0, 0, 0, pad_size), value=np.nan)
        else:
            padded = emb
        padded_emb.append(padded)
    
    batch_dict['embedding_matrix_full'] = torch.stack(padded_emb)


    if 'token_ids' in batch[0]:
        max_length = max(item['token_ids']['input_ids'].shape[1] for item in batch)
        
        padded_token_ids = {}
        for key in batch[0]['token_ids'].keys():
            padded = []
            for item in batch:
                tensor = item['token_ids'][key].float()
                pad_size = max_length - tensor.shape[1]
                if pad_size > 0:

                    padded_tensor = torch.nn.functional.pad(
                        tensor, (0, pad_size), 
                        mode='constant', 
                        value=np.nan
                    )
                else:
                    padded_tensor = tensor
                padded.append(padded_tensor)
            padded_token_ids[key] = torch.cat(padded)
        
        batch_dict['token_ids'] = padded_token_ids

    return batch_dict

class DependencyGraphDatasetFP_PyG(Dataset):
    def __init__(self, base_dir, indices=None, graph_process_type = 'direct_root', tokenizer_type = 'xlnet', max_tokens=512, encoder_layer=-1):
        
        self.data_dir = os.path.join(base_dir, tokenizer_type, graph_process_type, str(encoder_layer))

        self.indices = indices
        self.max_tokens = max_tokens
        self.all_files = []
        for file_name in os.listdir(self.data_dir):
                if file_name.endswith('.pkl'):
                    self.all_files.append(os.path.join(self.data_dir, file_name))
        
        # 파일 이름 기준으로 정렬
        self.all_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        if self.indices is not None:
            print(f"Total {len(self.all_files)} by {len(self.indices)}")
        else:
            print(f"USE {len(self.all_files)}")

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.all_files)

    def _adj_to_edge_index(self, adj_matrix):
        edges = torch.where(adj_matrix > 0)
        return torch.stack(edges)

    def __getitem__(self, idx):
        if self.indices is not None:
            file_idx = self.indices[idx]
        else:
            file_idx = idx
        
        with open(self.all_files[file_idx], 'rb') as f:
            item = pickle.load(f)

        # pdb.set_trace()
        node_list = list(item['graph'].nodes)
        node_list.sort()
        adj_matrix_full = nx.adjacency_matrix(item['graph'], nodelist=node_list).todense()

        full_graph = Data(
            x=torch.FloatTensor(item['embedding_matrix_full']),
            edge_index=self._adj_to_edge_index(torch.FloatTensor(adj_matrix_full)),
            y=torch.LongTensor([item['label']]),
            idx=file_idx,
        )

        if 'external_graph' in item:
            G = item['external_graph']
            if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                return None

            ex_node_list = list(item['external_graph'].nodes)
            ex_node_list.sort()

            try:
                ex_adj_matrix_full = nx.adjacency_matrix(item['external_graph'], nodelist=ex_node_list).todense()
            except nx.NetworkXError:
                print(f"[❌] Skipping idx={idx}: external_graph is empty.")
                return None

            ex_full_graph = Data(
                x=torch.FloatTensor(item['embedding_matrix_full']),
                edge_index=self._adj_to_edge_index(torch.FloatTensor(ex_adj_matrix_full)),
                y=torch.LongTensor([item['label']]),
                idx=idx,
            )

        if 'dummy_positions' in item:
            full_graph.dummy_positions = item['dummy_positions']

        token_ids = {}


        for key, value in item['token_ids'].items():
            tensor = torch.FloatTensor(value)
            if tensor.size(0) < self.max_tokens:
                token_ids[key] = torch.nn.functional.pad(
                    tensor, 
                    (0, self.max_tokens - tensor.size(0)), 
                    value=float('nan')
                )
            else:
                token_ids[key] = tensor

        
        node2token = torch.full((2, self.max_tokens), float('nan'))

        for key, values in item['node2token'].items():
            if values and all(v < self.max_tokens for v in values):
                if key >= self.max_tokens:
                    break
                if isinstance(values[0], torch.Tensor):
                    node2token[0][key] = values[0][0]  # start
                else:
                    node2token[0][key] = values[0]
                node2token[1][key] = len(values)  # lengths



        meta_data = {
            'idx': torch.tensor(file_idx, dtype=torch.long),
            'text': item['text'],
            'tokens': item['tokens'],
            'token_ids': token_ids,
            'token_info': item['token_info'],
            # 'sentences_token_info': item['sentences_token_info'],
            # 'node2token': node2token,
            'max_sent_len': item['max_sent_len'],
            # 'sent_lengths': item['sent_lengths'],
            'node2token': node2token,
            'raw_word': item['raw_word'],
            'root_positions': item['root_positions'],
            'label' : item['label']
        }
        # pdb.set_trace()
        if 'dummy_positions' in item:
            meta_data['dummy_positions'] = item['dummy_positions']
            
        out_dict = {
            'meta_data': meta_data,
            'full_graph': full_graph,
            # 'sent_graph': sent_graph
            }
        if 'external_graph' in item:
            out_dict['external_graph']=ex_full_graph

        return out_dict



from torch_geometric.data import Batch

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    cleaned_batch = []
    for item in batch:
        full_graph = item['full_graph']
        num_nodes = full_graph.x.size(0) if full_graph.x is not None else 0

        if num_nodes == 0:
            continue

        cleaned_batch.append(item)

    if len(cleaned_batch) == 0:
        return None

    batch = cleaned_batch
    batch_dict = {
        'meta_data': { 
            'text': [item['meta_data']['text'] for item in batch],
            'token_info': [item['meta_data']['token_info'] for item in batch],
            'tokens': [item['meta_data']['tokens'] for item in batch],
            'node2token': [item['meta_data']['node2token'] for item in batch],
            'max_sent_len': [item['meta_data']['max_sent_len'] for item in batch],
            'root_positions': [item['meta_data']['root_positions'] for item in batch],
            'raw_word': [item['meta_data']['raw_word'] for item in batch],
        }
    }

    batch_dict['full_graph'] = Batch.from_data_list([item['full_graph'] for item in batch])

    if 'external_graph' in batch[0]:
        batch_dict['external_graph'] = Batch.from_data_list([item['external_graph'] for item in batch])

    if 'dummy_positions' in batch[0]['meta_data']:
        batch_dict['meta_data']['dummy_positions'] = [item['meta_data']['dummy_positions'] for item in batch]

    # print(f"[collate_fn] Skipped {len(batch) - len(cleaned_batch)} empty-graph samples.")
    return batch_dict    
    
import json
def build_texts_from_docids(dataset_split, doc_path, include_query=True, sep_token="[SEP]"):
    doc_map = {}
    with open(doc_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            doc_map[obj["docid"]] = obj["document"]

    texts = []
    for item in dataset_split:
        query = item["query"].strip()
        docid = item["docids"][0]
        document = doc_map[docid].strip()

        if include_query:
            full_input = f"{query} {sep_token} {document}"
        else:
            full_input = document

        texts.append(full_input)

    return texts


def load_doc_text(doc_path, annotation_ids):
    texts = []
    for annotation_id in annotation_ids:
        doc_file = os.path.join(doc_path, annotation_id)
        try:
            with open(doc_file, "r", encoding="utf-8") as f:
                texts.append(f.read())
        except FileNotFoundError:
            print(f"⚠️ NO FILE: {doc_file}")
            texts.append("")
    return texts

def build_texts_from_qa(dataset_split, include_query=True, sep_token="[SEP]"):

    texts = []
    for item in dataset_split:
        query = item["question"].strip()
        document = item["context"].strip()

        if include_query:
            full_input = f"{query} {sep_token} {document}"
        else:
            full_input = document

        texts.append(full_input)

    return texts    