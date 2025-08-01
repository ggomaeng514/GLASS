import signal
import sys
import torch
import copy
def handle_exit(sig, frame):
    print(f"[!] Caught signal {sig}. Cleaning up...")
    torch.cuda.empty_cache()  
    
    try:
        wandb.run.mark_failed()  
    except Exception as e:
        print(f"[wandb] Failed to mark as failed: {e}")

    sys.exit(1)  

signal.signal(signal.SIGTERM, handle_exit)  # kill $pid
signal.signal(signal.SIGINT, handle_exit)   # Ctrl+C


import warnings
import torchvision
torchvision.disable_beta_transforms_warning()
from transformers import AutoModel
from transformers import XLNetTokenizer, XLNetModel, XLNetForSequenceClassification
from transformers import GPT2Tokenizer, GPT2Model, GPT2ForSequenceClassification, GPT2Config
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DebertaV2Config
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, DatasetDict

from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DebertaV2Config
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader

from model import Our_Selector_V1,  Model_Align2, CustomGPT2Classifier, Our_Selector_V1_WithGNN, Our_Selector_V1_WithGNN_Double, BioMedLMForSequenceClassification, Predictor_only
from dataset import DependencyGraphDataset, DependencyGraphDatasetFP_PyG, collate_fn
# from utill import CosineWarmupSchedulerForLambda, get_all_metrics
from utill import CosineWarmupSchedulerForLambda
# from torch_geometric.loader.dataloader import Collater
from datasets import load_dataset, DatasetDict, Dataset

from transformers import BertTokenizer, BertModel
from learning import *

import argparse
import os
import pdb
import random
import psutil
from datetime import datetime
import wandb
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import json
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--predictor_link', type=str, default='xlnet-base-cased') # 'gpt2
parser.add_argument('--graph_process_type', type=str, default='direct_root',choices=['direct_root', 'dummy_root', 'neighbor_dummy_node'])
parser.add_argument('--tokenizer_type', type=str, default='deberta',choices=['xlnet', 'gpt2', 'deberta', 'roberta', 'deberta_large', 'deberta_small', 'biolinkBert'])
parser.add_argument('--target_model', type=str, default='deberta',choices=['xlnet', 'gpt2', 'deberta', 'roberta'])
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('-c', '--cpu_start', type=int, default=0)
parser.add_argument('--use_cpu_num', type=int, default=8)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--wandb_off', action='store_true')
parser.add_argument('--log_dir', type=str, default='/storage/personal/myhwang/NLP_FS/logs/')
parser.add_argument('--data_dir', type=str, default='/storage/personal/myhwang/NLP_FS/data/')
parser.add_argument('--not_FP', action='store_true')
parser.add_argument('--data_name', type=str, default='movies', choices=['ag_news', 'glue_sst2', 'glue_cola', 'imdb', 'cose', 'movies', 'bioasq', 'graph_sst2'])
parser.add_argument('--lambda_smo', type=float, default=1)
parser.add_argument('--FC_load', action='store_true')
# seletor learning method
parser.add_argument('--selected_model_type', type=str, default='RL',choices=['STE', 'concrete', 'hard_concrete', 'stg', 'RL'])
# seletor's threshold
parser.add_argument('--gate_threshold', type=float, default=0.5)
# encoder_layer
parser.add_argument('--encoder_layer', type=int, default=-1)
# reinforcement learning mask sampling
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--num_hops', type=int, default=0)
parser.add_argument('--use_weighted_adjacency', action='store_true')

# for infernece
parser.add_argument('--inference_threshold', type=str, default=None) 
parser.add_argument('--validation_load_path', type=str, default=None)
parser.add_argument('--load_checkpoint', type=str, default='roc',choices=['roc', 'roc_ratio'])

# for policy loss's softplus
parser.add_argument('--policy_KL', action='store_true')
parser.add_argument('--policy_soft', type=float, default=-1)
parser.add_argument('--lambda_kl', type=float, default=1) 
parser.add_argument('--baseline_type', type=str, default='basic',choices=['basic', 'div_mean', 'mean'])


# for reg
parser.add_argument('--lambda_reg', type=float, default=1)
parser.add_argument('--reg_type', type=str, default='L0', choices=['L0', 'group_sparsity_G', 'group_sparsity_E', 'group_sparsity_G_N', 'L0_group_sparsity_E_N', 'L0_group_sparsity_G', 'L0_group_sparsity_E', 'L0_group_sparsity_G_N', 'L0_group_sparsity_E_N'])
parser.add_argument('--reg_rate', type=float, default=0.5) 
parser.add_argument('--group_matric', type=str, default='cos_softmax_node_cut', choices=['cos_nagative_cut', 'cos_softmax_node_cut']) 

# for reg_scheduler
parser.add_argument('--reg_update_step', type=int, default=0)

parser.add_argument('--lambda_reg_scheduler', action='store_true')
parser.add_argument('--warm_up_max', type=float, default=0.)
parser.add_argument('--warm_up_step', type=int, default=0)

# for mix_up
parser.add_argument('--mix_up_rate', type=float, default=0.)
# masking method
parser.add_argument('--modified_method', type=str, default='attention_mask', choices=['attention_mask', 'word', 'embedding']) 
parser.add_argument('--replace_token', type=str, default='blank', choices=['mask', 'blank', 'unk', 'the', '_', ',', '[PAD]', 'None'])

# for GNN
parser.add_argument('--use_gnn', action='store_true')
parser.add_argument('--use_double_gnn', action='store_true')
parser.add_argument('--gnn_type', type=str, default='GCN', choices=['GCN', 'GAT', 'graphSAGE']) 

parser.add_argument('--gnn_hidden_channels', type=int, default=768)
parser.add_argument('--gnn_num_layers', type=int, default=0)

parser.add_argument('--gnn_out_channels', type=int, default=768) # only gnn double
parser.add_argument('--dropout_rate', type=float, default=0.5)

parser.add_argument('--gat_heads', type=int, default=0)

parser.add_argument('--adj_type', type=str, default='syntactic', choices=['syntactic', 'semantic', 'fully']) 

parser.add_argument('--sage_agg', type=str, default='mean', choices=['mean', 'pool', 'lstm']) 

parser.add_argument('--model_paths', nargs='+', type=str, help="--model_paths sem=/path/to/sem.pkl syn=/path/to/syn.pkl")
parser.add_argument('--target_ratio', type=float, default=0.1)

parser.add_argument('--sem_num_threshold', type=float, default=1.)

args = parser.parse_args()

# CUDA_VISIBLE_DEVICES=2 python main.py --wandb_off --batch_size 128 --lr 0.001
def debug_collate(batch):
    for key in batch[0]:
        try:
            stacked = torch.stack([item[key] for item in batch])
        except Exception as e:
            print(f"[❗] key '{key}' failed to collate: {e}")
    raise RuntimeError("Collate test complete")
# python main.py --tokenizer_type deberta --batch_size 128 --FC_load --selected_model_type RL --lr 0.001 --num_samples 8 --lambda_reg 0.05 --lambda_smo 0.005 --epochs 10 --encoder_layer 10 -c 50 --reg_type group_sparsity_G_N --use_weighted_adjacency


def main():
    args = parser.parse_args()
    og_args = copy.deepcopy(args)
    args.log_dir = os.path.join(args.log_dir, args.data_name)
    p = psutil.Process()
    p.cpu_affinity(range(args.cpu_start, args.cpu_start+args.use_cpu_num))

    current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    print(f"Current time: {current_time}")

    if args.validation_load_path is not None:
        args_pkl_path = os.path.join(os.path.dirname(args.validation_load_path), "args.pkl")
    if os.path.exists(args_pkl_path):
        gate_threshold = args.gate_threshold
        validation_load_path = args.validation_load_path
        target_ratio = args.target_ratio
        modified_method = args.modified_method
        replace_token = args.replace_token
        

        load_checkpoint = args.load_checkpoint
        
        with open(args_pkl_path, "rb") as f:
            try:
                loaded_args = pickle.load(f)
                print("Loaded Arguments:", loaded_args)
                
                
                args_dict = vars(args)
                args_dict.update(vars(loaded_args))
                
                
                print("Updated Arguments:", args)
            except pickle.UnpicklingError as e:
                print(f"Error loading pickle file: {e}")
                raise

        args.gate_threshold = gate_threshold
        args.validation_load_path = validation_load_path
        args.load_checkpoint = load_checkpoint
        args.target_ratio = target_ratio
        args.modified_method = modified_method
        args.replace_token = replace_token
        print("Setting Threshold:", args.gate_threshold)
    else:
        print(f"File not found: {args_pkl_path}")
        
        
        if args.validation_load_path is not None:
            last_part = os.path.basename(args.validation_load_path)
        elif args.inference_threshold is not None:
            last_part = os.path.basename(os.path.dirname(args.inference_threshold))

        
        date_time_part = last_part.split('_')[-2] + '_' + last_part.split('_')[-1]
        args.current_time = date_time_part
        
        print("Parser Arguments:", args)

    
    if args.validation_load_path is not None:
        args.log_dir = os.path.join(og_args.log_dir, args.data_name , f"fix_rate_inference_{os.path.basename(os.path.dirname(args.validation_load_path))}_{args.target_ratio}")
            
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    save_path = os.path.join(args.log_dir, "args.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(args, f)

    print(f"Arguments saved to {save_path}")

    # wandb setting
    torch.cuda.empty_cache()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device:{args.device}')

    if not args.wandb_off:
        WANDB_AUTH_KEY = '7eb12004104f53b29bf47c33ede534ad7e984527'
        wandb.login(key=WANDB_AUTH_KEY)
        tags = [args.tokenizer_type, args.graph_process_type, args.selected_model_type, str(args.seed)]

        tags.append(f"Rate_{args.target_ratio}")
        wandb.init(project="NLP_TS",
                name=f"fix_inference_'{os.path.basename(os.path.dirname(args.validation_load_path))}'_{args.target_ratio}_{args.seed}",
                notes=f"{args.current_time}",
                tags=tags
                )
        wandb.config.update(args)


    # load dataset  
    if args.data_name == 'ag_news':
        dataset = load_dataset(args.data_name)
        dataset['train'] = dataset['train'].add_column('idx', range(len(dataset['train'])))
        dataset['test'] = dataset['test'].add_column('idx', range(len(dataset['test'])))

        train_valid = dataset['train'].train_test_split(test_size=0.2, seed=42, shuffle=True, stratify_by_column='label')
        train_valid['valid'] = train_valid.pop('test')
        train_indices = train_valid['train']['idx']
        valid_indices = train_valid['valid']['idx']
        test_indices = dataset['test']['idx']
        print(f"Train 인덱스 개수: {len(train_indices)}")
        print(f"Valid 인덱스 개수: {len(valid_indices)}")

        
        train_labels = np.array(train_valid['train']['label'])
        valid_labels = np.array(train_valid['valid']['label'])
        test_labels = np.array(dataset['test']['label'])
    elif args.data_name == 'glue_sst2':
        dataset = load_dataset("glue", "sst2")
        dataset = {("valid" if k == "validation" else k): v for k, v in dataset.items()}
        train_indices = dataset['train']['idx']
        valid_indices = dataset['valid']['idx']
        test_indices = dataset['test']['idx']
        train_labels = np.array(dataset['train']['label'])
        valid_labels = np.array(dataset['valid']['label'])
        test_labels = np.array(dataset['test']['label'])
    elif args.data_name == 'glue_cola':
        dataset = load_dataset("glue", "cola")
        dataset = {("valid" if k == "validation" else k): v for k, v in dataset.items()}
        train_indices = dataset['train']['idx']
        valid_indices = dataset['valid']['idx']
        test_indices = dataset['test']['idx']
        train_labels = np.array(dataset['train']['label'])
        valid_labels = np.array(dataset['valid']['label'])
        test_labels = np.array(dataset['test']['label'])


    elif args.data_name == 'imdb':
        dataset_ = load_dataset("imdb")
        text_key = 'text'
        label_key = 'label'

        dataset_["train"]=dataset_["train"].add_column("idx", list(range(len(dataset_["train"]))))
        dataset_["test"]=dataset_["test"].add_column("idx", list(range(len(dataset_["test"]))))
        
        if os.path.isfile(os.path.join(args.data_dir, args.data_name,"./train_idx.npy")) and os.path.isfile(os.path.join(args.data_dir, args.data_name,"./valid_idx.npy")):
            train_idx = np.load(os.path.join(args.data_dir, args.data_name,"./train_idx.npy"))
            val_idx = np.load(os.path.join(args.data_dir, args.data_name,"./val_idx.npy"))

            train_dataset = dataset_["train"].select(train_idx)
            val_dataset = dataset_["train"].select(val_idx)
            test_dataset= dataset_["test"]
        else:
            split_dataset = dataset_["train"].train_test_split(test_size=0.1, seed=42)

            
            train_dataset = split_dataset["train"]
            val_dataset = split_dataset["test"]
            test_dataset= dataset_["test"]

            train_idx = np.array(train_dataset["idx"])
            val_idx = np.array(val_dataset["idx"])
            
            os.makedirs(os.path.join(args.data_dir, args.data_name), exist_ok=True)
            np.save(os.path.join(args.data_dir, args.data_name,"./train_idx.npy"), train_idx)
            np.save(os.path.join(args.data_dir, args.data_name,"./val_idx.npy"), val_idx)


        dataset = DatasetDict({
            "train": train_dataset,
            "valid": val_dataset,
            "test": test_dataset
        })
        train_indices = dataset['train']['idx']
        valid_indices = dataset['valid']['idx']
        test_indices = dataset['test']['idx']
        train_labels = np.array(dataset['train']['label'])
        valid_labels = np.array(dataset['valid']['label'])
        test_labels = np.array(dataset['test']['label'])

    elif args.data_name == 'cose':

        file_paths = {
            "train": os.path.join("/storage/personal/myhwang/NLP_FS/data/eraser/data/cose_simplified", "train.jsonl"),
            "valid": os.path.join("/storage/personal/myhwang/NLP_FS/data/eraser/data/cose_simplified", "val.jsonl"),
            "test": os.path.join("/storage/personal/myhwang/NLP_FS/data/eraser/data/cose_simplified", "test.jsonl")
        }

        # Load each split
        dataset_ = load_dataset("json", data_files=file_paths)

        # Add idx column for traceability
        label_map = {"false": 0, "true": 1}
        for split in dataset_.keys():
            dataset_[split] = dataset_[split].add_column("idx", list(range(len(dataset_[split]))))
            dataset_[split] = dataset_[split].map(
                lambda example: {"label": label_map[example["classification"]]}
            )
        dataset = DatasetDict({
            "train": dataset_["train"],
            "valid": dataset_["valid"],
            "test": dataset_["test"]
        })
        train_indices = dataset['train']['idx']
        valid_indices = dataset['valid']['idx']
        test_indices = dataset['test']['idx']
        train_labels = np.array(dataset['train']['label'])
        valid_labels = np.array(dataset['valid']['label'])
        test_labels = np.array(dataset['test']['label'])


    elif args.data_name == 'movies':
        dataset_train = load_dataset("json", data_files={"train": os.path.join("/storage/personal/myhwang/NLP_FS/data/eraser/movies", "train.jsonl")})["train"]
        dataset_valid = load_dataset("json", data_files={"valid": os.path.join("/storage/personal/myhwang/NLP_FS/data/eraser/movies", "val.jsonl")})["valid"]
        dataset_test = load_dataset("json", data_files={"test": os.path.join("/storage/personal/myhwang/NLP_FS/data/eraser/movies", "test.jsonl")})["test"]
        dataset_test = dataset_test.remove_columns(["docids"])


        dataset_ = DatasetDict({
            "train": dataset_train,
            "valid": dataset_valid,
            "test": dataset_test
        })
        # Add idx column for traceability
        label_map = {"NEG": 0, "POS": 1}
        for split in dataset_.keys():
            dataset_[split] = dataset_[split].add_column("idx", list(range(len(dataset_[split]))))
            dataset_[split] = dataset_[split].map(
                lambda example: {"label": label_map[example["classification"]]}
            )
        dataset = DatasetDict({
            "train": dataset_["train"],
            "valid": dataset_["valid"],
            "test": dataset_["test"]
        })

        train_indices = dataset['train']['idx']
        valid_indices = dataset['valid']['idx']
        test_indices = dataset['test']['idx']
        train_labels = np.array(dataset['train']['label'])
        valid_labels = np.array(dataset['valid']['label'])
        test_labels = np.array(dataset['test']['label'])


    elif args.data_name == 'bioasq':
        
        DATA_DIR = "/storage/personal/myhwang/NLP_FS/data"          
        SCRIPT    = "/storage/personal/myhwang/NLP_FS/data/biomedical/bigbio/biodatasets/bioasq_task_b/bioasq_task_b.py"

        
        dataset = load_dataset(
            SCRIPT,
            name="bioasq_blurb_bigbio_qa",
            data_dir=DATA_DIR,
            trust_remote_code=True
        )
        
        label_map = {"yes": 1, "no": 0}

        def add_label(example):
            
            example["label"] = label_map[example["answer"][0].lower()]
            return example

        
        dataset = dataset.map(add_label, remove_columns=["answer"])
        dataset = {("valid" if k == "validation" else k): v for k, v in dataset.items()}


        train_indices = dataset['train']['id']
        valid_indices = dataset['valid']['id']
        test_indices = dataset['test']['id']
        train_labels = np.array(dataset['train']['label'])
        valid_labels = np.array(dataset['valid']['label'])
        test_labels = np.array(dataset['test']['label'])        



    elif args.data_name == 'graph_sst2':
        # sentence tokens
        text_key = 'text'
        label_key = 'label'
        label_path = "/storage/personal/myhwang/NLP_FS/data/Graph-SST2/raw/Graph-SST2_graph_labels.txt"
        graph_labels = np.loadtxt(label_path, dtype=int)
        with open("/storage/personal/myhwang/NLP_FS/data/Graph-SST2/raw/Graph-SST2_sentence_tokens.json", "r") as f:
            graph_data = json.load(f)
            texts = [" ".join(tokens) for tokens in graph_data.values()] 
            train_num = 67349
            valid_num = 872
            test_num = 1821
            # indexing
            train_texts = texts[:train_num]
            valid_texts = texts[train_num:train_num + valid_num]
            test_texts  = texts[train_num + valid_num:train_num + valid_num + test_num]
            
            train_labels = graph_labels[:train_num].tolist()
            valid_labels = graph_labels[train_num:train_num + valid_num].tolist()
            test_labels  = graph_labels[train_num + valid_num:train_num + valid_num + test_num].tolist()
            
            train_idx = list(range(0, train_num))
            valid_idx = list(range(train_num, train_num + valid_num))
            test_idx  = list(range(train_num + valid_num, train_num + valid_num + test_num))
            

            dataset = DatasetDict({
                "train": Dataset.from_dict({"text": train_texts, "label": train_labels, "idx": train_idx}),
                "valid": Dataset.from_dict({"text": valid_texts, "label": valid_labels, "idx": valid_idx}),
                "test":  Dataset.from_dict({"text": test_texts,  "label": test_labels,  "idx": test_idx}),
            })
            # dataset["train"] = dataset["train"].filter(lambda example: example["idx"] != 36601)

            # 36601
        train_indices = dataset['train']['idx']
        valid_indices = dataset['valid']['idx']
        test_indices = dataset['test']['idx']        


    
    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    print("\nTrain set label 분포:")
    for label, count in zip(unique_train, counts_train):
        ratio = count / len(train_labels)
        print(f"Label {label}: {ratio:.2%}")

    
    unique_valid, counts_valid = np.unique(valid_labels, return_counts=True)
    print("\nValidation set label 분포:")
    for label, count in zip(unique_valid, counts_valid):
        ratio = count / len(valid_labels)
        print(f"Label {label}: {ratio:.2%}")

    
    unique_test, counts_test = np.unique(test_labels, return_counts=True)
    print("\nTest set label 분포:")
    for label, count in zip(unique_test, counts_test):
        ratio = count / len(test_labels)
        print(f"Label {label}: {ratio:.2%}")

    if args.not_FP:
        if args.data_name == 'ag_news':
            text_key = 'text'
            label_key = 'label'
        if args.data_name == 'glue_sst2':
            text_key = 'sentence'
            label_key = 'label'
        elif args.data_name == 'glue_cola':
            name = 'cola'
            text_key = 'sentence'
        train_dataset = DependencyGraphDataset(texts=train_valid['train'][text_key], labels=train_valid['train'][label_key], graph_process_type=args.graph_process_type, tokenizer_type=args.tokenizer_type)
        valid_dataset = DependencyGraphDataset(texts=train_valid['valid'][text_key], labels=train_valid['valid'][label_key], graph_process_type=args.graph_process_type, tokenizer_type=args.tokenizer_type)
        test_dataset = DependencyGraphDataset(texts=dataset['test'][text_key], labels=dataset['test'][label_key], graph_process_type=args.graph_process_type, tokenizer_type=args.tokenizer_type)
        
    else:
        
        train_indices = np.array(train_indices)
        valid_indices = np.array(valid_indices)
        test_indices = np.array(test_indices)

        # train_dataset = DependencyGraphDatasetFP_PyG(
        #     f'/storage/personal/myhwang/NLP_FS/data/{args.data_name}/train/', 
        #     indices=train_indices if 'glue' in args.data_name else None,
        #     graph_process_type=args.graph_process_type,
        #     tokenizer_type=args.tokenizer_type,
        #     encoder_layer=args.encoder_layer, 
        # )

        # valid_dataset = DependencyGraphDatasetFP_PyG(
        #     f'/storage/personal/myhwang/NLP_FS/data/{args.data_name}/valid/', 
        #     indices=valid_indices if 'glue' in args.data_name else None,
        #     graph_process_type=args.graph_process_type,
        #     tokenizer_type=args.tokenizer_type,
        #     encoder_layer=args.encoder_layer,  
        # )

        test_dataset = DependencyGraphDatasetFP_PyG(
            f'/storage/personal/myhwang/NLP_FS/data/{args.data_name}/test/', 
            indices=test_indices if 'glue' in args.data_name else None,
            graph_process_type=args.graph_process_type,
            tokenizer_type=args.tokenizer_type,
            encoder_layer=args.encoder_layer,  
        )

    # train_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True, collate_fn=collate_fn)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.use_cpu_num, drop_last=True, collate_fn=collate_fn)
    # valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=args.use_cpu_num, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=args.use_cpu_num, collate_fn=collate_fn)

    # tokenizer
    if args.tokenizer_type == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
        encoder = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=4)
        encoder.transformer.mask_emb.requires_grad_(False)
        mask_token_id = torch.tensor([tokenizer.mask_token_id], device=args.device)
        args.enc_mask_embedding = encoder.transformer.word_embedding(mask_token_id).detach()
    elif args.tokenizer_type == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        config = GPT2Config.from_pretrained("gpt2")
        config.num_labels = 2  
        encoder = CustomGPT2Classifier(config)
        
        mask_token_id = torch.tensor([tokenizer.unk_token_id]) # unk_token_id
        args.enc_mask_embedding = encoder.transformer.wte(mask_token_id).detach()
        encoder.config.pad_token_id = tokenizer.pad_token_id

    elif args.tokenizer_type == 'deberta':
        model_name = "microsoft/deberta-v3-base"
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        encoder = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=2).to(args.device)
        mask_token_id = torch.tensor([tokenizer.mask_token_id], device=args.device)
        args.enc_mask_embedding = encoder.deberta.embeddings.word_embeddings(mask_token_id).detach()
        
    elif args.tokenizer_type == 'roberta':
        model_name = "FacebookAI/roberta-base"
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        encoder = RobertaModel.from_pretrained(model_name)    

    elif args.tokenizer_type == 'BioMedLM':
        model_name = "stanford-crfm/BioMedLM"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModelForCausalLM.from_pretrained(model_name)   
        special_tokens_dict = {
            "additional_special_tokens": ["[CONTEXT]", "[QUESTION]", "[ANSWER]"]
        }
        tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        encoder.resize_token_embeddings(len(tokenizer))

    elif args.tokenizer_type == 'biolinkBert':
        model_name = 'michiyasunaga/BioLinkBERT-large'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name).to(args.device)


    elif args.tokenizer_type == 'bert':
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        encoder = BertModel.from_pretrained(model_name).to(args.device)
        mask_token_id = torch.tensor([tokenizer.mask_token_id], device=args.device)
        args.enc_mask_embedding = encoder.embeddings.word_embeddings(mask_token_id).detach()




    # target model
    if args.target_model == 'xlnet':
        predictor_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
        predictor = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=4)
        predictor.transformer.mask_emb.requires_grad_(False)

    elif args.target_model == 'gpt2':
        predictor_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        predictor_tokenizer.pad_token = predictor_tokenizer.eos_token
        config = GPT2Config.from_pretrained("gpt2")
        config.num_labels = 2  
        predictor = CustomGPT2Classifier(config)
        
        mask_token_id = torch.tensor([predictor_tokenizer.unk_token_id]) # unk_token_id
        args.mask_embedding = predictor.transformer.wte(mask_token_id).detach()
        predictor.config.pad_token_id = predictor_tokenizer.pad_token_id

    elif args.target_model == 'deberta':
        model_name = "microsoft/deberta-v3-base"
        predictor_tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        predictor = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=2).to(args.device)
        mask_token_id = torch.tensor([predictor_tokenizer.mask_token_id], device=args.device)
        args.mask_embedding = predictor.deberta.embeddings.word_embeddings(mask_token_id).detach()
        
    elif args.target_model == 'roberta':
        if args.data_name == 'imdb' :
            model_name = "textattack/roberta-base-imdb" 
        else:
            model_name = "textattack/roberta-base-ag-news"         
        predictor_tokenizer = RobertaTokenizer.from_pretrained(model_name)
        predictor = RobertaForSequenceClassification.from_pretrained(model_name).to(args.device)
        mask_token_id = torch.tensor([predictor_tokenizer.mask_token_id], device=args.device)
        args.mask_embedding = predictor.roberta.embeddings.word_embeddings(mask_token_id).detach()

    elif args.target_model == 'BioMedLM':
        model_name = "stanford-crfm/BioMedLM"
        predictor_tokenizer = AutoTokenizer.from_pretrained(model_name)
        special_tokens_dict = {
            "additional_special_tokens": ["[CONTEXT]", "[QUESTION]", "[ANSWER]"]
        }
        predictor_tokenizer.add_special_tokens(special_tokens_dict)
        config = GPT2Config.from_pretrained(model_name)
        config.num_labels = 2
        predictor_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        base_model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
        # Now wrap it
        predictor = BioMedLMForSequenceClassification(config=config)
        predictor.transformer.load_state_dict(base_model.transformer.state_dict(), strict=False)
        predictor= predictor.to(args.device)
        mask_token_id = torch.tensor([predictor_tokenizer.mask_token_id], device=args.device)
        args.mask_embedding = predictor.roberta.embeddings.word_embeddings(mask_token_id).detach()

    elif args.target_model == 'biolinkBert':
        model_name = 'michiyasunaga/BioLinkBERT-large'
        predictor_tokenizer = AutoTokenizer.from_pretrained(model_name)
        # load_model = torch.load("/storage/personal/myhwang/NLP_FS/logs/biolinkBert/LLM_only/biolinkBert_LLM_only_42_2025-05-09_20:55:35/biolinkBert_42_2025-05-09_20:55:35.pt")  
        
        load_model = torch.load("/storage/personal/myhwang/NLP_FS/logs/biolinkBert/LLM_only/biolinkBert_LLM_only_42_2025-05-12_22:15:02/biolinkBert_42_2025-05-12_22:15:02.pt")  

        predictor = load_model.predictor
        mask_token_id = torch.tensor([tokenizer.mask_token_id], device=args.device)
        args.mask_embedding = predictor.bert.embeddings.word_embeddings(mask_token_id).detach() 
        # # predictor = AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-large')
        # predictor = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(args.device)
        # pdb.set_trace()



        
    if args.tokenizer_type in ['xlnet', 'deberta', 'roberta', 'gpt2', 'bert']:
        node_embedding_dim = 768
    elif args.tokenizer_type in ['biolinkBert']:
        node_embedding_dim = 1024
    elif args.tokenizer_type in ['BioMedLM']:
        node_embedding_dim = 1024
    # elif args.tokenizer_type in 'biolinkBert':
    #     node_embedding_dim = 725

    if args.use_gnn:
        selected_model= Our_Selector_V1_WithGNN(
                                    args=args,
                                    device=args.device,
                                    node_hidden_dim=node_embedding_dim,
                                    
                                    gnn_in_channels=node_embedding_dim,
                                    gnn_hidden_channels=args.gnn_hidden_channels,
                                    gnn_out_channels=1,
                                    gnn_num_layers=args.gnn_num_layers,
                                    gat_heads=args.gat_heads,
                                    model_type=args.gnn_type,
                                    
                                    adj_type=args.adj_type,
                                    dropout=args.dropout_rate
                                    
                                    )
    elif args.use_double_gnn:
        selected_model= Our_Selector_V1_WithGNN_Double(
                                    args=args,
                                    device=args.device,
                                    node_hidden_dim=node_embedding_dim,
                                    
                                    syn_gnn_in_channels=node_embedding_dim,
                                    syn_gnn_hidden_channels=args.gnn_hidden_channels,
                                    syn_gnn_out_channels=args.gnn_out_channels,
                                    syn_gnn_num_layers=args.gnn_num_layers,
                                    syn_gat_heads=args.gat_heads,
                                    
                                    sem_gnn_in_channels=node_embedding_dim,
                                    sem_gnn_hidden_channels=args.gnn_hidden_channels,
                                    sem_gnn_out_channels=args.gnn_out_channels,
                                    sem_gnn_num_layers=args.gnn_num_layers,
                                    sem_gat_heads=args.gat_heads,
                                    model_type=args.gnn_type,
                                    dropout=args.dropout_rate
                                    )

    else:
        selected_model = Our_Selector_V1(
                                    args=args,
                                    device=args.device,
                                    node_hidden_dim=node_embedding_dim,
                                    )

    # model = Model_Align(tokenizer=tokenizer, predictor=predictor, selected_model=selected_model, args=args).to(args.device)

    # model_name = "bert-base-uncased"
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    # encoder = BertModel.from_pretrained(model_name).to(args.device)
    # pad_token_id = tokenizer.pad_token_id

    if args.modified_method == 'embedding' and args.replace_token != "None":
        # pdb.set_trace()
        token_id = predictor_tokenizer.convert_tokens_to_ids(args.replace_token)  # 또는 tokenizer.encode(word, add_special_tokens=False)[0]
        token_tensor = torch.tensor([token_id], device=args.device)
        
        if args.target_model == 'deberta':
            replace_embedding = predictor.deberta.embeddings.word_embeddings(token_tensor).detach()
    else:
        replace_embedding =None
    
    model = Model_Align2(tokenizer=tokenizer, predictor_tokenizer=predictor_tokenizer, predictor=predictor, selected_model=selected_model, args=args, mask_embedding=replace_embedding).to(args.device)


    if args.target_model == 'xlnet':
        sequence_summary_weights = torch.load(f'./predictor_weights/{args.data_name}/sequence_summary_weights.pt')
        logits_proj_weights = torch.load(f'./predictor_weights/{args.data_name}/logits_proj_weights.pt')

        model.predictor.sequence_summary.load_state_dict(sequence_summary_weights)
        model.predictor.logits_proj.load_state_dict(logits_proj_weights)

    elif args.target_model == 'deberta':
        pooler_weights = torch.load(f'./predictor_weights/{args.data_name}/pooler_weights.pt')
        classifier_weights = torch.load(f'./predictor_weights/{args.data_name}/classifier_weights_weights.pt')

        model.predictor.pooler.load_state_dict(pooler_weights)
        model.predictor.classifier.load_state_dict(classifier_weights)   

    elif args.target_model == 'gpt2':
        score_weights = torch.load(f'./predictor_weights/{args.data_name}/score_weights.pt')
        model.predictor.score.load_state_dict(score_weights)

    elif args.target_model == 'roberta':
        pass

    elif args.target_model == 'BioMedLM':
        predictor_weights = torch.load(f'./predictor_weights/{args.data_name}/BioMed_classifier_weights.pt')
        model.predictor.load_state_dict(predictor_weights)


    for param in model.parameters():
        param.requires_grad = False

    model.args.gate_threshold= args.gate_threshold 
    model.args.mix_up_rate = args.mix_up_rate
    model.args.num_samples = 1
    model.args.modified_method = args.modified_method
    model.args.replace_token = args.replace_token

    print("REF_PATH: ",og_args.validation_load_path)
    other_model_paths = parse_key_value_list(og_args.model_paths)

    for path_key, path_value in other_model_paths.items():
        if os.path.isfile(os.path.join(path_value,'test_sample_prob.pkl' )):
            print(f"✅ {path_key}: 파일 존재 → {path_value}")
        else:
            print(f"❌ {path_key}: 파일 없음! → {path_value}")

    # args.target_ratio = list(args.target_ratio.values())[0]
    # print(target_ratio)

    # model_thresholds = find_best_threshold_per_model(other_model_paths, target_ratio=args.target_ratio)


    # print(model_thresholds)
    masks = generate_fixed_rank_masks(
        args,
        other_model_paths,
        args.target_ratio  
    )
    
    CE_loss = nn.CrossEntropyLoss(reduction='none')
    
    for path_key, fix_mask in masks.items(): 
        total_test_loss_ce = 0
        total_base_loss_ce =0
        total_test_kl = 0
        all_test_logits = []
        all_test_labels = []
        all_baseline_test_logits = []

        all_filtered_tokens = {}
        all_filtered_words = {}
        total_above_threshold = 0
        total_test_tokens = 0
        model.eval()
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f"Test:", leave=False)
            for data in test_pbar:
                data['meta_data']['idx'] = data['full_graph']['idx']
                # outputs, regularizer, loss_smo, gate, hard_gate, total_token, (batch_filtered_token, batch_filtered_words) = model(data['full_graph'].to(args.device), data['meta_data'], test=True)
                # indices = data['meta_data']['idx'].tolist()  
                
                indices = data['meta_data']['idx'].tolist()  
                # pdb.set_trace()
                fix_mask_ = [fix_mask[i] for i in indices]
                # fix_mask_=fix_mask[indices]
                flat_mask = torch.cat(fix_mask_, dim=0).unsqueeze(1) 
                out = model.fix_test(data['full_graph'].to(args.device), data['meta_data'], flat_mask.to(args.device))
                # out = model(data['full_graph'].to(args.device), data['meta_data'], test=True)
                outputs = out['outputs']
                hard_gate = out['token_hard_gate']
                total_token = out['total_token']
                batch_filtered_token = out['filtered_tokens'] 
                batch_filtered_words = out['filtered_words'] 

                

                # pdb.set_trace()
                loss_ce = CE_loss(outputs.logits, data['full_graph']['y'])
                # loss_reg = torch.nanmean(regularizer)
                base_outputs = model.fix_test(data['full_graph'].to(args.device), data['meta_data'], flat_mask.to(args.device), baseline = True)

                # base_outputs = model(data['full_graph'].to(args.device), data['meta_data'], baseline = True)
                base_loss_ce = CE_loss(base_outputs.logits, data['full_graph']['y']).squeeze()
                
                total_base_loss_ce += base_loss_ce.sum().item()  
                print(total_base_loss_ce)
                

                temperature =1 
                teacher_probs = F.softmax(base_outputs.logits.repeat_interleave(args.num_samples, dim=0) / temperature, dim=-1)
                student_log_probs = F.log_softmax(outputs.logits / temperature, dim=-1)
                kl=(F.kl_div(student_log_probs, teacher_probs, reduction='none') * (temperature**2)).sum(dim=1)

                # print(f"val gate_inputs: {gate}")
                num_above_threshold = (hard_gate).sum()
                print(f"num_above_threshold: {num_above_threshold}/{total_token}")

                total_test_loss_ce += loss_ce.mean().item()
                total_test_kl += kl.mean().item()
                all_test_logits.append(outputs.logits.detach().cpu())
                all_test_labels.append(data['full_graph']['y'].repeat(args.num_samples).detach().cpu())
                all_baseline_test_logits.append(base_outputs.logits.detach().cpu())
                # num_above_threshold = (gate > args.gate_threshold).sum()

                total_above_threshold += num_above_threshold
                total_test_tokens += total_token
                all_filtered_tokens.update(batch_filtered_token)
                all_filtered_words.update(batch_filtered_words)
                # break
                
            

            total_test_loss_ce = total_test_loss_ce / len(test_loader)
            total_test_kl = total_test_kl / len(test_loader)


            
            test_logits = torch.cat(all_test_logits, dim=0)
            test_labels = torch.cat(all_test_labels, dim=0)
            test_preds = torch.argmax(test_logits, dim=1).cpu()
            test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()

            test_baseline_logits = torch.cat(all_baseline_test_logits, dim=0)
            test_baseline_preds = torch.argmax(test_baseline_logits, dim=1).cpu()
            test_baseline_probs = torch.softmax(test_baseline_logits, dim=1).cpu().numpy()

            print("Test Loss CE: {:.5f}".format(total_test_loss_ce))
            # 
            test_metrics = get_all_metrics(
                test_labels.cpu().numpy(), 
                test_preds.numpy(), 
                test_probs, 
                n_classes=2 if args.data_name != "ag_news" else 4, 
                prefix=f"test ({path_key})"
            )
            average_above_threshold = total_above_threshold / total_test_tokens
            print(f"Ratio of test gates above threshold_{path_key}: {average_above_threshold:.4f}")
            print(f"Total number of test gates above threshold_{path_key}: {total_above_threshold}")
            print(f"Total number of test gates:_{path_key} {total_test_tokens}")

            logging.debug(f"Test Loss CE_{path_key}: {total_test_loss_ce}")
            

            if not args.wandb_off:
                wandb.log({f"test_loss_ce ({path_key})": total_test_loss_ce})
                wandb.log(test_metrics)
                wandb.log({f"Ratio of test gates above threshold ({path_key})": average_above_threshold.item()})
                wandb.log({f"Total number of test gates above threshold ({path_key})": total_above_threshold.item()})
                wandb.log({f"Total number of test gates ({path_key})": total_test_tokens.item()})
                wandb.log({f"test_kl ({path_key})": total_test_kl})
            
            
            save_filtered_word_path = os.path.join(args.log_dir, f"test_filtered_words_({path_key}).pkl")
            with open(save_filtered_word_path, "wb") as f:
                pickle.dump(all_filtered_words, f)
            print(f"Filtered words saved to {save_filtered_word_path}")
            
            save_filtered_token_path = os.path.join(args.log_dir, f"test_filtered_tokens_({path_key}).pkl")
            with open(save_filtered_token_path, "wb") as f:
                pickle.dump(all_filtered_tokens, f)
            print(f"Filtered words saved to {save_filtered_token_path}")
            

            save_test_preds_path = os.path.join(args.log_dir, f"test_preds_({path_key}).npy")
            save_test_probs_path = os.path.join(args.log_dir, f"test_probs_({path_key}).npy")

            np.save(save_test_preds_path, test_preds.numpy())
            np.save(save_test_probs_path, test_probs)

            print(f"Test predictions saved to {save_test_preds_path}")
            print(f"Test probabilities saved to {save_test_probs_path}")

            save_test_preds_path = os.path.join(args.log_dir, f"test_baseline_preds_({path_key}).npy")
            save_test_probs_path = os.path.join(args.log_dir, f"test_baseline_probs_({path_key}).npy")

            np.save(save_test_preds_path, test_baseline_preds.numpy())
            np.save(save_test_probs_path, test_baseline_probs)


            state = {
                        'data' : args.data_name,
                        'lr' : args.lr,
                        'test_loss_ce' : total_test_loss_ce,
                        'Ratio of test gates above threshold' : average_above_threshold.item(),
                        'Total number of test gates above threshold' : total_above_threshold.item(),
                        'Total number of test gates' : total_test_tokens.item(),
                        'num_samples' : args.num_samples,
                        'batch_size' : args.batch_size,
                        'lambda_reg' : args.lambda_reg,
                        'lambda_smo' : args.lambda_smo,
                        'encoder_layer' : args.encoder_layer,
                        'num_hops' : args.num_hops,

                    }
            
            
            for key, value in test_metrics.items():
                state[key] = value

            with open(os.path.join(os.path.join(args.log_dir), f'top_performance_({path_key}).json'), 'w') as outfile:
                json.dump(state, outfile)
    


import numpy as np
import torch
def generate_fixed_rank_masks(args, model_paths, target_ratio=0.05, ignore_zeros=True):
    all_model_masks = {}

    for model_name, model_path in model_paths.items():
        prob_path = os.path.join(model_path, "test_sample_prob.pkl")
        if not os.path.exists(prob_path):
            print(f"❌ Missing: {prob_path}")
            continue

        model_probs = np.load(prob_path, allow_pickle=True)

        
        flat_probs = []
        flat_indices = []  

        for sample_idx, sample in enumerate(model_probs.values()):
            prob = np.asarray(sample["prob"])
            for token_idx, p in enumerate(prob):
                if not ignore_zeros or p > 0:
                    flat_probs.append(p)
                    flat_indices.append((sample_idx, token_idx))

        if len(flat_probs) == 0:
            raise ValueError("⚠️ No valid probabilities to select from.")

        flat_probs = np.array(flat_probs)
        total_tokens = len(flat_probs)
        num_top_tokens = max(1, int(total_tokens * target_ratio))

        
        topk_indices = np.argpartition(-flat_probs, num_top_tokens)[:num_top_tokens]
        selected_positions = {flat_indices[i] for i in topk_indices}

        
        model_masks = []
        for sample_idx, sample in enumerate(model_probs.values()):
            length = len(sample["prob"])
            mask = torch.zeros(length, dtype=torch.long)
            for token_idx in range(length):
                if (sample_idx, token_idx) in selected_positions:
                    mask[token_idx] = 1
            model_masks.append(mask)

        selected_count = sum(m.sum().item() for m in model_masks)

        if not args.wandb_off:
            wandb.log({
                f"{model_name}/selected_ratio": selected_count / total_tokens,
                f"{model_name}/total_tokens": total_tokens,
                f"{model_name}/selected_tokens": selected_count,
            })

        all_model_masks[model_name] = model_masks

    return all_model_masks



def parse_key_value_list(pairs):
    result = {}
    for pair in pairs:
        key, value = pair.split('=', 1)
        result[key] = value
    return result

def pad_or_truncate(mask, max_len=512):
    if len(mask) > max_len:
        return mask[:max_len]
    elif len(mask) < max_len:
        pad = torch.zeros(max_len - len(mask), dtype=mask.dtype)
        return torch.cat([mask, pad])
    else:
        return mask

if __name__ == "__main__":
    main()
