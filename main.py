import signal
import sys
import torch
import json
def handle_exit(sig, frame):
    print(f"[!] Caught signal {sig}. Cleaning up...")
    torch.cuda.empty_cache()  # 중요!
    # wandb에 실패로 기록
    try:
        wandb.run.mark_failed()  # ❗ 실패로 표시
    except Exception as e:
        print(f"[wandb] Failed to mark as failed: {e}")

    sys.exit(1)  # exit code != 0 → 실패로 기록됨

signal.signal(signal.SIGTERM, handle_exit)  # kill $pid
signal.signal(signal.SIGINT, handle_exit)   # Ctrl+C


import torchvision
torchvision.disable_beta_transforms_warning()
from transformers import AutoTokenizer, AutoModel
from transformers import XLNetTokenizer, XLNetModel, XLNetForSequenceClassification
from transformers import GPT2Tokenizer, GPT2Model, GPT2ForSequenceClassification, GPT2Config
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DebertaV2Config
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer, BertModel
# from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout
from datasets import load_dataset, DatasetDict, Dataset

# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader

from model import Predictor_only
from model import Our_Selector_V1,  Model_Align2, CustomGPT2Classifier, Our_Selector_V1_WithGNN, Our_Selector_V1_WithGNN_Double, BioMedLMForSequenceClassification, Predictor_only
from dataset import DependencyGraphDataset, DependencyGraphDatasetFP_PyG, collate_fn
from utill import CosineWarmupSchedulerForLambda
# from torch_geometric.loader.dataloader import Collater
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

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--predictor_link', type=str, default='xlnet-base-cased') # 'gpt2
parser.add_argument('--graph_process_type', type=str, default='direct_root',choices=['direct_root', 'dummy_root', 'neighbor_dummy_node'])
parser.add_argument('--tokenizer_type', type=str, default='deberta',choices=['xlnet', 'gpt2', 'deberta', 'roberta', 'biolinkBert', 'bert', 'deberta_small', 'deberta_large', 'BioMedLM'])
parser.add_argument('--target_model', type=str, default='deberta',choices=['xlnet', 'gpt2', 'deberta', 'roberta', 'biolinkBert'])
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
parser.add_argument('--data_name', type=str, default='glue_sst2', choices=['ag_news', 'glue_sst2', 'glue_cola', 'imdb', 'cose', 'movies', 'bioasq', 'graph_sst2'])
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
parser.add_argument('--inference_threshold', type=str, default=None) # train에서 threshold 적용시 결과 분석을 위함
parser.add_argument('--validation_load_path', type=str, default=None)
parser.add_argument('--load_checkpoint', type=str, default='roc',choices=['roc', 'roc_ratio'])

# for policy loss's softplus
parser.add_argument('--policy_KL', action='store_true')
parser.add_argument('--policy_soft', type=float, default=-1)
parser.add_argument('--lambda_kl', type=float, default=1) # 1이면 KL만 사용 [0, 1]
parser.add_argument('--baseline_type', type=str, default='basic',choices=['basic', 'div_mean', 'mean'])


# for reg
parser.add_argument('--lambda_reg', type=float, default=1)
parser.add_argument('--reg_type', type=str, default='L0', choices=['L0', 'group_sparsity_G', 'group_sparsity_E', 'group_sparsity_G_N', 'L0_group_sparsity_E_N', 'L0_group_sparsity_G', 'L0_group_sparsity_E', 'L0_group_sparsity_G_N', 'L0_group_sparsity_E_N'])
parser.add_argument('--reg_rate', type=float, default=0.5) # LO와 group_sparsity 함께 사용할때의 비율 reg_rate가 L0의 비율
parser.add_argument('--group_matric', type=str, default='cos_softmax_node_cut', choices=['cos_nagative_cut', 'cos_softmax_node_cut']) # LO와 group_sparsity 함께 사용할때의 비율 reg_rate가 L0의 비율

# for reg_scheduler
parser.add_argument('--reg_update_step', type=int, default=0)

parser.add_argument('--lambda_reg_scheduler', action='store_true')
parser.add_argument('--warm_up_max', type=float, default=0.)
parser.add_argument('--warm_up_step', type=int, default=0)

# for mix_up
parser.add_argument('--mix_up_rate', type=float, default=0.)
# masking method
parser.add_argument('--modified_method', type=str, default='attention_mask', choices=['attention_mask', 'word', 'embedding']) # masking input 처리 방법 attention_mask 조작 / raw_word 대체

# for GNN
parser.add_argument('--use_gnn', action='store_true')
parser.add_argument('--use_double_gnn', action='store_true')
parser.add_argument('--gnn_type', type=str, default='GCN', choices=['GCN', 'GAT', 'graphSAGE']) 

parser.add_argument('--gnn_hidden_channels', type=int, default=768)
parser.add_argument('--gnn_num_layers', type=int, default=0)

parser.add_argument('--gnn_out_channels', type=int, default=768) # only gnn double
parser.add_argument('--dropout_rate', type=float, default=0.5)

parser.add_argument('--gat_heads', type=int, default=0)

parser.add_argument('--adj_type', type=str, default='syntactic', choices=['syntactic', 'semantic', 'fully', 'cross']) 

parser.add_argument('--sage_agg', type=str, default='mean', choices=['mean', 'pool', 'lstm']) 

parser.add_argument('--sem_num_threshold', type=float, default=1.)

parser.add_argument('--replace_token', type=str, default='blank', choices=['mask', 'blank', 'unk', 'the', '_', ',', 'pad'])

# self-attention
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=2)


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
    args.log_dir = os.path.join(args.log_dir, args.data_name)
    p = psutil.Process()
    p.cpu_affinity(range(args.cpu_start, args.cpu_start+args.use_cpu_num))

    current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    print(f"Current time: {current_time}")
    if (args.validation_load_path is None) and (args.inference_threshold is None):
        args.log_dir= os.path.join(args.log_dir, f'{args.tokenizer_type}',f'{args.graph_process_type}', f"{args.tokenizer_type}_{args.graph_process_type}_{args.selected_model_type}_{args.seed}_{current_time}")  
        args.current_time = current_time
    else:
        # args 로드
        if args.validation_load_path is not None:
            args_pkl_path = os.path.join(args.validation_load_path, "args.pkl")
            
        elif args.inference_threshold is not None:
            args_pkl_path = os.path.join(os.path.dirname(args.inference_threshold), "args.pkl")
        if os.path.exists(args_pkl_path):
            gate_threshold = args.gate_threshold
            validation_load_path = args.validation_load_path
            load_checkpoint = args.load_checkpoint
            with open(args_pkl_path, "rb") as f:
                try:
                    loaded_args = pickle.load(f)
                    print("Loaded Arguments:", loaded_args)
                    
                    # 기존 args를 유지하면서 로드한 값으로 업데이트
                    args_dict = vars(args)
                    args_dict.update(vars(loaded_args))
                    
                    # 필요한 값 복원
                    print("Updated Arguments:", args)
                except pickle.UnpicklingError as e:
                    print(f"Error loading pickle file: {e}")
                    raise

            args.gate_threshold = gate_threshold
            args.validation_load_path = validation_load_path
            args.load_checkpoint = load_checkpoint
            print("Setting Threshold:", args.gate_threshold)
        else:
            print(f"File not found: {args_pkl_path}")
            # 마지막 디렉토리 이름 추출
            
            if args.validation_load_path is not None:
                last_part = os.path.basename(args.validation_load_path)
            elif args.inference_threshold is not None:
                last_part = os.path.basename(os.path.dirname(args.inference_threshold))

            # 날짜 및 시간 부분만 추출
            date_time_part = last_part.split('_')[-2] + '_' + last_part.split('_')[-1]
            args.current_time = date_time_part
            
            print("Parser Arguments:", args)
        if args.validation_load_path is not None:
            args.log_dir = os.path.join(args.validation_load_path, f"validation_threshold_{args.gate_threshold}")
            
    
    os.makedirs(args.log_dir, exist_ok=True)
    # args 저장
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
        if args.validation_load_path is None:
            if args.FC_load:
                tags.append('NotFC')
            if args.use_gnn:
                tags.append('GNN')
            if args.policy_soft > 0:
                tags.append('policy_softplus')
            if args.gnn_type == 'GAT':
                tags.append('GAT')
            elif args.gnn_type == 'graphSAGE':
                tags.append('graphSAGE')
                
            wandb.init(project="NLP_TS",
                    name=f"{args.tokenizer_type}_{args.graph_process_type}_{args.selected_model_type}_{args.seed}" if not args.FC_load else f"{args.tokenizer_type}_{args.graph_process_type}_{args.selected_model_type}_NotFC_{args.seed}",
                    notes=f"{args.current_time}",
                    tags=tags)
        else:
            tags.append(f"threshold_{args.gate_threshold}")
            if args.FC_load:
                tags.append('NotFC')
            if args.use_gnn:
                tags.append('GNN')
            if args.policy_soft > 0:
                tags.append('policy_softplus')
            if args.gnn_type == 'GAT':
                tags.append('GAT')
            elif args.gnn_type == 'graphSAGE':
                tags.append('graphSAGE')                
            wandb.init(project="NLP_TS",
                    name=f"{args.tokenizer_type}_{args.graph_process_type}_{args.selected_model_type}_{args.seed}_th_{args.gate_threshold}",
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

        # numpy array로 변환
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

            # 결과
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
        # dataset_train = load_dataset("json", data_files={"train": os.path.join("/storage/personal/myhwang/NLP_FS/data/BioASQ", "bioasq_yesno_train_512.jsonl")})["train"]
        
        # dataset_test = load_dataset("json", data_files={"test": os.path.join("/storage/personal/myhwang/NLP_FS/data/BioASQ", "bioasq_10b_yesno_test_512.jsonl")})["test"]
        # train_file = os.path.join("/storage/personal/myhwang/NLP_FS/data/BioASQ", "bioasq_yesno_train_clipped.jsonl")
        # test_file = os.path.join("/storage/personal/myhwang/NLP_FS/data/BioASQ", "bioasq_10b_yesno_test.jsonl")
        
        # dataset = DatasetDict()
        # dataset["train"] = load_dataset("json", data_files={"train": train_file})["train"]
        # dataset["test"] = load_dataset("json", data_files={"test": test_file})["test"]

        # # Train → Train + Valid split
        # split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        # dataset["train"] = split["train"]
        # dataset["valid"] = split["test"]

        # def rename_fields(example):
        #     return {
        #         "context": example["sentence2"],
        #         "question": example["sentence1"]
        #     }
        # for split_name_ in dataset:
        #     dataset[split_name_] = dataset[split_name_].map(rename_fields)

        # # 레이블 매핑: yes → 1, no → 0
        # label_map = {"yes": 1, "no": 0}
        # for split_name_ in dataset:
        #     dataset[split_name_] = dataset[split_name_].add_column("idx", list(range(len(dataset[split_name_]))))
        #     dataset[split_name_] = dataset[split_name_].map(lambda x: {"label": label_map[x["label"]]})
        DATA_DIR = "/storage/personal/myhwang/NLP_FS/data"          # 1번에서 만든 폴더
        SCRIPT    = "/storage/personal/myhwang/NLP_FS/data/biomedical/bigbio/biodatasets/bioasq_task_b/bioasq_task_b.py"

        # (B) BLURB Yes/No 벤치마크용 split (670 / 75 / 140)
        dataset = load_dataset(
            SCRIPT,
            name="bioasq_blurb_bigbio_qa",
            data_dir=DATA_DIR,
            trust_remote_code=True
        )
        
        label_map = {"yes": 1, "no": 0}

        def add_label(example):
            # answer는 ['yes'] 형태의 리스트 → 첫 원소만 꺼내서 매핑
            example["label"] = label_map[example["answer"][0].lower()]
            return example

        # 모든 split(train/validation/test)에 적용
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
            # 각 split에 대한 sample_idx 생성
            train_idx = list(range(0, train_num))
            valid_idx = list(range(train_num, train_num + valid_num))
            test_idx  = list(range(train_num + valid_num, train_num + valid_num + test_num))
            

            dataset = DatasetDict({
                "train": Dataset.from_dict({"text": train_texts, "label": train_labels, "idx": train_idx}),
                "valid": Dataset.from_dict({"text": valid_texts, "label": valid_labels, "idx": valid_idx}),
                "test":  Dataset.from_dict({"text": test_texts,  "label": test_labels,  "idx": test_idx}),
            })
            dataset["train"] = dataset["train"].filter(lambda example: example["idx"] != 36601)

            # 36601
        train_indices = dataset['train']['idx']
        valid_indices = dataset['valid']['idx']
        test_indices = dataset['test']['idx']        
    # Train set label 분포
    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    print("\nTrain set label 분포:")
    for label, count in zip(unique_train, counts_train):
        ratio = count / len(train_labels)
        print(f"Label {label}: {ratio:.2%}")

    # Validation set label 분포
    unique_valid, counts_valid = np.unique(valid_labels, return_counts=True)
    print("\nValidation set label 분포:")
    for label, count in zip(unique_valid, counts_valid):
        ratio = count / len(valid_labels)
        print(f"Label {label}: {ratio:.2%}")

    # Test set label 분포
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
        elif args.data_name == 'movies':
            text_key = 'text'
            label_key = 'label'
        elif args.data_name == 'bioasq':
            label_key = 'label'
            text_key = ['context', 'question']
        train_dataset = DependencyGraphDataset(texts=train_valid['train'][text_key], labels=train_valid['train'][label_key], graph_process_type=args.graph_process_type, tokenizer_type=args.tokenizer_type)
        valid_dataset = DependencyGraphDataset(texts=train_valid['valid'][text_key], labels=train_valid['valid'][label_key], graph_process_type=args.graph_process_type, tokenizer_type=args.tokenizer_type)
        test_dataset = DependencyGraphDataset(texts=dataset['test'][text_key], labels=dataset['test'][label_key], graph_process_type=args.graph_process_type, tokenizer_type=args.tokenizer_type)
        
    else:
        # 인덱스를 numpy 배열로 변환
        train_indices = np.array(train_indices)
        valid_indices = np.array(valid_indices)
        test_indices = np.array(test_indices)

        train_dataset = DependencyGraphDatasetFP_PyG(
            f'/storage/personal/myhwang/NLP_FS/data/{args.data_name}/train/', 
            indices=train_indices if 'glue' in args.data_name else None,
            graph_process_type=args.graph_process_type,
            tokenizer_type=args.tokenizer_type,
            encoder_layer=args.encoder_layer, 
        )

        valid_dataset = DependencyGraphDatasetFP_PyG(
            f'/storage/personal/myhwang/NLP_FS/data/{args.data_name}/valid/', 
            indices=valid_indices if 'glue' in args.data_name else None,
            graph_process_type=args.graph_process_type,
            tokenizer_type=args.tokenizer_type,
            encoder_layer=args.encoder_layer,  
        )

        test_dataset = DependencyGraphDatasetFP_PyG(
            f'/storage/personal/myhwang/NLP_FS/data/{args.data_name}/test/', 
            indices=test_indices if 'glue' in args.data_name else None,
            graph_process_type=args.graph_process_type,
            tokenizer_type=args.tokenizer_type,
            encoder_layer=args.encoder_layer,  
        )

    # train_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True, collate_fn=collate_fn)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.use_cpu_num, drop_last=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=args.use_cpu_num, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=args.use_cpu_num, collate_fn=collate_fn)

    # tokenizer
    if args.tokenizer_type == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
        encoder = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=4)
        encoder.transformer.mask_emb.requires_grad_(False)
        mask_token_id = torch.tensor([tokenizer.mask_token_id], device=args.device)
        args._enc_mask_embedding = encoder.transformer.word_embedding(mask_token_id).detach()
    elif args.tokenizer_type == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        config = GPT2Config.from_pretrained("gpt2")
        config.num_labels = 2  # 분류할 라벨 수 설정
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

    elif args.tokenizer_type == 'deberta_small':
        model_name = "microsoft/deberta-v3-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name).to(args.device)
        mask_token_id = torch.tensor([tokenizer.mask_token_id], device=args.device)
        args.enc_mask_embedding = encoder.embeddings.word_embeddings(mask_token_id).detach()

    elif args.tokenizer_type == 'deberta_large':
        model_name = "microsoft/deberta-v3-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name).to(args.device)
        mask_token_id = torch.tensor([tokenizer.mask_token_id], device=args.device)
        args.enc_mask_embedding = encoder.embeddings.word_embeddings(mask_token_id).detach()

    elif args.tokenizer_type == 'roberta':
        model_name = "FacebookAI/roberta-base"
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        encoder = RobertaModel.from_pretrained(model_name).to(args.device)
        mask_token_id = torch.tensor([tokenizer.mask_token_id], device=args.device)
        args.enc_mask_embedding = encoder.embeddings.word_embeddings(mask_token_id).detach()

    elif args.tokenizer_type == 'BioMedLM':
        model_name = "stanford-crfm/BioMedLM"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModelForCausalLM.from_pretrained(model_name)   
        special_tokens_dict = {
            "additional_special_tokens": ["[CONTEXT]", "[QUESTION]", "[ANSWER]"]
        }
        tokenizer.add_special_tokens(special_tokens_dict)
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        encoder.resize_token_embeddings(len(tokenizer))
        mask_token_id = torch.tensor([tokenizer.mask_token_id], device=args.device)
        args.enc_mask_embedding = encoder.embeddings.word_embeddings(mask_token_id).detach()

    elif args.tokenizer_type == 'biolinkBert':
        model_name = 'michiyasunaga/BioLinkBERT-large'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name).to(args.device)
        mask_token_id = torch.tensor([tokenizer.mask_token_id], device=args.device)
        args.enc_mask_embedding = encoder.embeddings.word_embeddings(mask_token_id).detach()

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
        config.num_labels = 2  # 분류할 라벨 수 설정
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
        predictor_tokenizer = AutoTokenizer.from_pretrained("./predictor_weights/bioasq/tokenizer")
        predictor = AutoModelForSequenceClassification.from_pretrained("./predictor_weights/bioasq/BioMedLM")
        # model_name = "stanford-crfm/BioMedLM"
        # predictor_tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModelForSequenceClassification.from_pretrained("./predictor_weights/bioasq/BioMedLM")
        # special_tokens_dict = {
        #     "additional_special_tokens": ["[CONTEXT]", "[QUESTION]", "[ANSWER]"]
        # }
        # predictor_tokenizer.add_special_tokens(special_tokens_dict)
        # config = GPT2Config.from_pretrained(model_name)
        # config.num_labels = 2
        # predictor_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # base_model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
        # # Now wrap it
        # predictor = BioMedLMForSequenceClassification(config=config)
        # predictor.transformer.load_state_dict(base_model.transformer.state_dict(), strict=False)
        # predictor.resize_token_embeddings(len(tokenizer))
        # predictor= predictor.to(args.device)
        # mask_token_id = torch.tensor([predictor_tokenizer.mask_token_id], device=args.device)
        # args.mask_embedding = predictor.roberta.embeddings.word_embeddings(mask_token_id).detach()

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
        
    if args.tokenizer_type in ['xlnet', 'deberta', 'roberta', 'gpt2', 'bert' ,'deberta_small']:
        node_embedding_dim = 768
    elif args.tokenizer_type in ['biolinkBert']:
        node_embedding_dim = 1024
    elif args.tokenizer_type in ['BioMedLM', 'deberta_large']:
        node_embedding_dim = 1024
        # pdb.set_trace()
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

    model = Model_Align2(tokenizer=tokenizer, predictor_tokenizer=predictor_tokenizer, predictor=predictor, selected_model=selected_model, args=args, mask_embedding=None).to(args.device)

    
    if not args.FC_load:
        optimizer = torch.optim.Adam([
        param for name, param in model.named_parameters() 
        if ('predictor' not in name) or 
           ('output' in name) or 
           ('sequence_summary' in name)
        ], lr=args.lr)
    
    else:
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

        elif args.target_model == 'biolinkBert':
            pass
            # pooler_weights = torch.load(f'./predictor_weights/{args.data_name}/pooler_weights.pt')
            # classifier_weights = torch.load(f'./predictor_weights/{args.data_name}/classifier_weights.pt')
            # model.predictor.bert.pooler.load_state_dict(pooler_weights)
            # model.predictor.classifier.load_state_dict(classifier_weights)
            # load_model= torch.load("/storage/personal/myhwang/NLP_FS/logs/biolinkBert/LLM_only/biolinkBert_LLM_only_42_2025-05-09_20:55:35/biolinkBert_42_2025-05-09_20:55:35.pt")
            # torch.save(load_model.predictor.bert.pooler.state_dict(), f'./predictor_weights/{self.args.data_name}/pooler_weights.pt')
            
            # model.predictor = torch.load("/storage/personal/myhwang/NLP_FS/logs/biolinkBert/LLM_only/biolinkBert_LLM_only_42_2025-05-09_20:55:35/biolinkBert_42_2025-05-09_20:55:35.pt")


        if args.inference_threshold is not None:
            
            # "/storage/personal/myhwang/NLP_FS/logs/biolinkBert/LLM_only/biolinkBert_LLM_only_42_2025-05-09_20:55:35"/deberta_42.pt
            

            model=torch.load(args.inference_threshold).to(args.device)
            model.args.mix_up_rate = args.mix_up_rate
            for param in model.parameters():
                param.requires_grad = False
            # for param in model.selected_model.parameters():
            #     param.requires_grad = True
            for param in model.predictor.pooler.parameters():
                param.requires_grad = True
            for param in model.predictor.classifier.parameters():
                param.requires_grad = True

            for name, param in model.named_parameters():
                if param.device != args.device:
                    print(f"Parameter {name} is not on the correct device: {param.device}")
            for name, param in model.selected_model.named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")

            optimizer = torch.optim.Adam([
                param for name, param in model.named_parameters() 
                if ('pooler', 'classifier' in name)
                ], lr=args.lr)    
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001, last_epoch=-1)
        elif args.validation_load_path is not None:
            
            if args.load_checkpoint == 'roc':   
                # model=torch.load(os.path.join(args.validation_load_path,  f"{args.tokenizer_type}_{args.seed}.pt")).to(args.device)
                saved_model = torch.load(os.path.join(args.validation_load_path, f"{args.tokenizer_type}_{args.seed}.pt"), map_location=args.device)
                model.load_state_dict(saved_model.state_dict())                
            elif args.load_checkpoint == 'roc_ratio':
                
                # model=torch.load(os.path.join(args.validation_load_path,  f"{args.tokenizer_type}_{args.seed}_roc_ratio.pt")).to(args.device)
                saved_model = torch.load(os.path.join(args.validation_load_path, f"{args.tokenizer_type}_{args.seed}_roc_ratio.pt"), map_location=args.device)
                model.load_state_dict(saved_model.state_dict())
            else:
                warnings.warn("모델 로드 실패: checkpoint 값이 올바르지 않습니다.", UserWarning)
            
            model.args.gate_threshold= args.gate_threshold 
            model.args.mix_up_rate = args.mix_up_rate
            model.args.num_samples = 1
            
            optimizer = None
            scheduler = None
        else:
            
            for param in model.parameters():
                param.requires_grad = False
            for param in model.selected_model.parameters():
                param.requires_grad = True
                

            for name, param in model.named_parameters():
                if param.device != args.device:
                    print(f"Parameter {name} is not on the correct device: {param.device}")
            for name, param in model.selected_model.named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")
            # for name, param in model.named_parameters():
            #     print(f"{name}: requires_grad={param.requires_grad}")

            optimizer = torch.optim.Adam([
                param for name, param in model.named_parameters() 
                if ('selected_model' in name)
                ], lr=args.lr)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001, last_epoch=-1)
    # 모든 파라미터가 GPU에 있는지 확인
    # for name, param in model.named_parameters():
    #     if param.device != args.device:
    #         print(f"Parameter {name} is not on the correct device: {param.device}")
    # optimizer_selector = torch.optim.Adam(selected_model.parameters(), args.lr)
    # logits_proj 모듈 확인
    # for name, param in model.selected_model.graph_a_gather.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")
    
    # 학습할 파라미터 선택
    lambda_reg_scheduler = CosineWarmupSchedulerForLambda(warmup_step=args.warm_up_step, min_lambda=args.lambda_reg, max_lambda=args.warm_up_max)

    print("============================= Train =============================")

    model = Model(
                model=model,                
                optimizer=optimizer,
                scheduler=scheduler,
                lambda_reg_scheduler=lambda_reg_scheduler,
                args=args) 
    if args.selected_model_type == 'RL':
        if args.inference_threshold is not None:
            model.train_RL_train_threshold_testing(train_loader, valid_loader, wandb)
        elif args.validation_load_path is not None:
            args.num_samples = 1
            model.args = args
            print("============================= Valid & Inference =============================")
            print("============================= Test & Inference =============================")
            if 'glue' in args.data_name:
                model.valid_RL(valid_loader, wandb)
                model.test_RL_glue(test_loader, wandb)
            else:
                model.test_RL(test_loader, wandb)

            return
        else:
            model.train_RL(train_loader, valid_loader, wandb)
            print("============================= Test & Inference =============================")
            # model.valid_RL(valid_loader, wandb)
            model.test_RL(test_loader, wandb)
    else:
        model.train(train_loader, valid_loader, wandb)

        print("============================= Test & Inference =============================")
        model.test(test_loader, wandb)


if __name__ == "__main__":
    main()
