import torchvision
torchvision.disable_beta_transforms_warning()
from transformers import AutoModel
from transformers import XLNetTokenizer, XLNetModel, XLNetForSequenceClassification
from transformers import GPT2Tokenizer, GPT2Model, GPT2ForSequenceClassification, GPT2Config
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification, DebertaV2Config, DebertaV2Tokenizer, DebertaV2Model
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from datasets import load_dataset, DatasetDict, Dataset
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn as nn

from model import Our_Selector_V1, Model_Align2, CustomGPT2Classifier, BioMedLMForSequenceClassification
from dataset import DependencyGraphDataset, DependencyGraphDatasetFP_PyG, collate_fn
from learning_predictor import *

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

parser = argparse.ArgumentParser()
parser.add_argument('--predictor_link', type=str, default='xlnet-base-cased') # 'gpt2
parser.add_argument('--graph_process_type', type=str, default='direct_root',choices=['direct_root', 'dummy_root', 'neighbor_dummy_node'])
parser.add_argument('--tokenizer_type', type=str, default='deberta',choices=['xlnet', 'gpt2', 'deberta', 'roberta', 'BioMedLM', 'biolinkBert'])
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('-c','--cpu_start', type=int, default=0)
parser.add_argument('--use_cpu_num', type=int, default=8)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--wandb_off', action='store_true')
parser.add_argument('--log_dir', type=str, default='/storage/personal/myhwang/NLP_FS/logs/')
parser.add_argument('--data_dir', type=str, default='/storage/personal/myhwang/NLP_FS/data/')
parser.add_argument('--not_FP', action='store_true')
parser.add_argument('--gate_threshold', type=float, default=0.5)
parser.add_argument('--data_name', type=str, default='glue_sst2', choices=['ag_news', 'glue_sst2', 'glue_cola', 'imdb', 'cose', 'movies', 'cose_simplified', 'bioasq', 'graph_sst2'])
args = parser.parse_args()

# CUDA_VISIBLE_DEVICES=4 python main_predictor.py --wandb_off --batch_size 128 --lr 0.001

def main():
    args = parser.parse_args()

    p = psutil.Process()
    p.cpu_affinity(range(args.cpu_start, args.cpu_start+args.use_cpu_num))

    current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    args.current_time = current_time
    args.log_dir= os.path.join(args.log_dir, f'{args.tokenizer_type}',f'LLM_only', f"{args.tokenizer_type}_LLM_only_{args.seed}_{current_time}")
    os.makedirs(args.log_dir, exist_ok=True)
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
        wandb.init(project="NLP_TS",
                name=f"{args.tokenizer_type}_LLM_only_{args.seed}",
                notes=f"{current_time}",
                tags=['LLM_only']
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
        
        if os.path.isfile(os.path.join(args.data_dir, args.data_name, "./train_idx.npy")) and os.path.isfile(os.path.join(args.data_dir, args.data_name,"./valid_idx.npy")):
            train_idx = np.load(os.path.join(args.data_dir, args.data_name, "./train_idx.npy"))
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
            "train": os.path.join("/storage/personal/myhwang/NLP_FS/data/eraser/data/", "cose_simplified", "train.jsonl"),
            "valid": os.path.join("/storage/personal/myhwang/NLP_FS/data/eraser/data/", "cose_simplified", "val.jsonl"),
            "test": os.path.join("/storage/personal/myhwang/NLP_FS/data/eraser/data/", "cose_simplified", "test.jsonl")
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
        elif args.data_name == 'glue_sst2':
            text_key = 'sentence'
            label_key = 'label'
        elif args.data_name == 'glue_cola':
            name = 'cola'
            text_key = 'sentence'
            label_key = 'label'
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
            encoder_layer=10,  
        )

        valid_dataset = DependencyGraphDatasetFP_PyG(
            f'/storage/personal/myhwang/NLP_FS/data/{args.data_name}/valid/', 
            indices=valid_indices if 'glue' in args.data_name else None,
            graph_process_type=args.graph_process_type,
            tokenizer_type=args.tokenizer_type,
            encoder_layer=10,  
        )

        test_dataset = DependencyGraphDatasetFP_PyG(
            f'/storage/personal/myhwang/NLP_FS/data/{args.data_name}/test/', 
            indices=test_indices if 'glue' in args.data_name else None,
            graph_process_type=args.graph_process_type,
            tokenizer_type=args.tokenizer_type,
            encoder_layer=10,  
        )



    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate_fn)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.use_cpu_num, drop_last=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    if args.tokenizer_type == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
        predictor = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=4)
        predictor.transformer.mask_emb.requires_grad_(True)
    elif args.tokenizer_type == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        # predictor = GPT2Model.from_pretrained("gpt2")
        # pdb.set_trace()

        # mask_token_id = torch.tensor([tokenizer.unk_token_id]) # unk_token_id
        # args.mask_embedding = predictor.transformer.wte(mask_token_id).detach()
        # predictor.config.pad_token_id = tokenizer.pad_token_id
        
        config = GPT2Config.from_pretrained("gpt2")
        config.num_labels = 2  # 분류할 라벨 수 설정
        predictor = CustomGPT2Classifier(config)

        mask_token_id = torch.tensor([tokenizer.unk_token_id]) # unk_token_id
        args.mask_embedding = predictor.transformer.wte(mask_token_id).detach()
        predictor.config.pad_token_id = tokenizer.unk_token_id
    
    elif args.tokenizer_type == 'deberta':
        
        model_name = "microsoft/deberta-v3-base"
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        predictor = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=2).to(args.device)

        mask_token_id = torch.tensor([tokenizer.mask_token_id], device=args.device)
        args.mask_embedding = predictor.deberta.embeddings.word_embeddings(mask_token_id).detach()


    elif args.tokenizer_type == 'roberta':
        if args.data_name == 'imdb' :
            model_name = "textattack/roberta-base-imdb" 
        else:
            model_name = "textattack/roberta-base-ag-news"        
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        predictor = RobertaForSequenceClassification.from_pretrained(model_name).to(args.device)  
        # pdb.set_trace() 
        mask_token_id = torch.tensor([tokenizer.mask_token_id], device=args.device)
        args.mask_embedding = predictor.roberta.embeddings.word_embeddings(mask_token_id).detach()     
    
    elif args.tokenizer_type == 'BioMedLM':
        model_name = "stanford-crfm/BioMedLM"
        if os.path.exists("./predictor_weights/bioasq/tokenizer_BioMedLM"):
            print("✅ Custom tokenizer found. Loading it.")
            tokenizer = AutoTokenizer.from_pretrained("./predictor_weights/bioasq/tokenizer_BioMedLM")
        else:
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            special_tokens_dict = {
                "additional_special_tokens": ["[CONTEXT]", "[QUESTION]", "[ANSWER]"]
            }
            tokenizer.add_special_tokens(special_tokens_dict)
            config = GPT2Config.from_pretrained(model_name)
            config.num_labels = 2
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.save_pretrained("./predictor_weights/bioasq/tokenizer_BioMedLM")
        
        if os.path.exists("./predictor_weights/bioasq/BioMedLM_init"):
            print("✅ Custom model found. Loading it.")
            predictor = AutoModelForSequenceClassification.from_pretrained("./predictor_weights/bioasq/BioMedLM_init")
        else:
            
            config = GPT2Config.from_pretrained(model_name)
            base_model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
            # Now wrap it
            predictor = BioMedLMForSequenceClassification(config=config)
            predictor.transformer.load_state_dict(base_model.transformer.state_dict(), strict=False)
            predictor.resize_token_embeddings(len(tokenizer))
        predictor= predictor.to(args.device)
        mask_token_id = torch.tensor([tokenizer.unk_token_id], device=args.device)
        args.mask_embedding = predictor.transformer.wte(mask_token_id).detach()

    elif args.tokenizer_type == 'biolinkBert':
        model_name = 'michiyasunaga/BioLinkBERT-large'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # predictor = AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-large')
        predictor = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(args.device)
        mask_token_id = torch.tensor([tokenizer.mask_token_id], device=args.device)
        args.mask_embedding = predictor.bert.embeddings.word_embeddings(mask_token_id).detach()   


    # pdb.set_trace()
    model = Predictor_only(tokenizer=tokenizer, predictor=predictor, args=args).to(args.device)
    # 모든 파라미터가 GPU에 있는지 확인
    # model=torch.load('/storage/personal/myhwang/NLP_FS/logs/gpt2/LLM_only/gpt2_LLM_only_42_2025-03-26_21:05:25/gpt2_42.pt')
    # torch.save(model.state_dict(), './predictor_weights/gpt_weight.pt')
    # 필요한 가중치 추출
    if args.tokenizer_type == 'xlnet':
        sequence_summary_weights = model.predictor.sequence_summary.state_dict()
        logits_proj_weights = model.predictor.logits_proj.state_dict()

        # torch.save(sequence_summary_weights, f'./predictor_weights/{args.data_name}/sequence_summary_weights.pt')
        # torch.save(logits_proj_weights, f'./predictor_weights/{args.data_name}/logits_proj_weights.pt')

        # for param in model.parameters():
        #     param.requires_grad = False
        # for param in model.predictor.sequence_summary.parameters():
        #     param.requires_grad = True
        # for param in model.predictor.logits_proj.parameters():
        #     param.requires_grad = True
        # pdb.set_trace()
        # for name, param in model.named_parameters():
        #     if param.device != args.device:
        #         print(f"Parameter {name} is not on the correct device: {param.device}")
        # # optimizer_selector = torch.optim.Adam(selected_model.parameters(), args.lr)

        # # sequence_summary 모듈 확인
        # for name, param in model.predictor.sequence_summary.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")

        # # logits_proj 모듈 확인
        # for name, param in model.predictor.logits_proj.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
        
        # # 학습할 파라미터 선택
        # # optimizer 설정 수정
        # trainable_params = [
        #     {'params': model.predictor.sequence_summary.parameters()},
        #     {'params': model.predictor.logits_proj.parameters()}
        # ]
        for param in model.parameters():
            param.requires_grad = True

        trainable_params = [
            {'params': model.predictor.parameters()},
        ]
        
    elif args.tokenizer_type == 'deberta':

        pooler_weights = model.predictor.pooler.state_dict()
        classifier_weights = model.predictor.classifier.state_dict()

        # # pdb.set_trace()
        # torch.save(pooler_weights, './predictor_weights/pooler_weights.pt')
        # torch.save(classifier_weights, './predictor_weights/classifier_weights_weights.pt')

        # pdb.set_trace()
        for name, param in model.named_parameters():
            if param.device != args.device:
                print(f"Parameter {name} is not on the correct device: {param.device}")
        # optimizer_selector = torch.optim.Adam(selected_model.parameters(), args.lr)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.predictor.pooler.parameters():
            param.requires_grad = True
        for param in model.predictor.classifier.parameters():
            param.requires_grad = True
        # pooler 모듈 확인
        for name, param in model.predictor.pooler.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")

        # classifier 모듈 확인
        for name, param in model.predictor.classifier.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")
        for name, param in model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")
        trainable_params = [
            {'params': model.predictor.pooler.parameters()},
            {'params': model.predictor.classifier.parameters()}
        ]

    elif args.tokenizer_type == 'gpt2':
        score_weights = model.predictor.score.state_dict()

        # # torch.save(score_weights, './predictor_weights/score_weights.pt')
        # if args.tokenizer_type == 'gpt2':
        #     score_weights = torch.load('./predictor_weights/score_weights.pt')
        #     model.predictor.score.load_state_dict(score_weights)
        # pdb.set_trace()
        for param in model.parameters():
            param.requires_grad = False
        for param in model.predictor.score.parameters():
            param.requires_grad = True
        # pdb.set_trace()
        for name, param in model.named_parameters():
            if param.device != args.device:
                print(f"Parameter {name} is not on the correct device: {param.device}")
        # optimizer_selector = torch.optim.Adam(selected_model.parameters(), args.lr)

        # sequence_summary 모듈 확인
        for name, param in model.predictor.score.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")

        
        # 학습할 파라미터 선택
        # optimizer 설정 수정
        trainable_params = [
            {'params': model.predictor.score.parameters()},
        ]
    elif args.tokenizer_type == 'roberta':

        for param in model.parameters():
            param.requires_grad = False
        for param in model.predictor.classifier.parameters():
            param.requires_grad = True

        for name, param in model.named_parameters():
            if param.device != args.device:
                print(f"Parameter {name} is not on the correct device: {param.device}")
        trainable_params = [
            {'params': model.predictor.classifier.parameters()},
        ]
    elif args.tokenizer_type == 'BioMedLM':
        # pdb.set_trace()
        for param in model.parameters():
            param.requires_grad = False
        for param in model.predictor.classifier.parameters():
            param.requires_grad = True

        for name, param in model.named_parameters():
            if param.device != args.device:
                print(f"Parameter {name} is not on the correct device: {param.device}")
        trainable_params = [
            {'params': model.predictor.classifier.parameters()},
            # {'params': model.predictor.parameters()},
        ]

    elif args.tokenizer_type == 'biolinkBert':
        # pdb.set_trace()


        ############################ 학습파라미터 조정##############################
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = [
            {'params': model.parameters()},
        ]
        

        # for param in model.parameters():
        #     param.requires_grad = False
        # for param in model.predictor.bert.pooler.parameters():
        #     param.requires_grad = True
        # for param in model.predictor.classifier.parameters():
        #     param.requires_grad = True
        # # pooler 모듈 확인
        # trainable_params = [
        #     {'params': model.predictor.bert.pooler.parameters()},
        #     {'params': model.predictor.classifier.parameters()}
        # ]
    # pdb.set_trace()
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001, last_epoch=-1)


    print("============================= Train =============================")

    model = Model(
                model=model,                
                optimizer=optimizer,
                scheduler=scheduler,
                args=args) 
    # model.train(train_loader, valid_loader, wandb)

    print("============================= Test & Inference =============================")
    
    # pooler_weights = torch.load(f'./predictor_weights/{args.data_name}/pooler_weights.pt')
    # classifier_weights = torch.load(f'./predictor_weights/{args.data_name}/classifier_weights_weights.pt')
    # # pdb.set_trace()
    # model.model.predictor.pooler.load_state_dict(pooler_weights)
    # model.model.predictor.classifier.load_state_dict(classifier_weights)   


    model.test(test_loader, wandb)


class Predictor_only(nn.Module):
    def __init__(self, tokenizer, predictor, args):
        super(Predictor_only, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.predictor = predictor

    def forward(self, data, meta_data):

        if args.data_name == 'bioasq':
            # pdb.set_trace()
            if self.args.tokenizer_type == 'BioMedLM':
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
            elif self.args.tokenizer_type == 'biolinkBert':
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


if __name__ == "__main__":
    main()
