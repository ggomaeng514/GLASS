
####################################################################################################################

from dataset import DependencyGraphDataset, collate_fn_pickle, build_texts_from_docids, load_doc_text, build_texts_from_qa
from datasets import load_dataset, DatasetDict, Dataset
import os
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import psutil
import json
import pdb

def process_and_save_dataset(data_name, split_name, output_dir, tokenizer_type, graph_type, batch_size=1, encoder_layer_depth = -1):
    
    if data_name == 'ag_news':
        text_key = 'text'
        label_key = 'label'
        
        dataset_ = load_dataset('ag_news')
        dataset_['train'] = dataset_['train'].add_column('idx', list(range(len(dataset_['train']))))
        dataset_['test']  = dataset_['test'].add_column('idx', list(range(len(dataset_['test'] ))))

        
        train_idx_file = os.path.join(output_dir, data_name, 'train_idx.npy')
        val_idx_file   = os.path.join(output_dir, data_name, 'val_idx.npy')


        if os.path.isfile(train_idx_file) and os.path.isfile(val_idx_file):
            train_idx = np.load(train_idx_file)
            val_idx   = np.load(val_idx_file)

            train_dataset = dataset_['train'].select(train_idx)
            val_dataset   = dataset_['train'].select(val_idx)
            test_dataset  = dataset_['test']

        
        else:
            split = dataset_['train'].train_test_split(test_size=0.1, seed=42)
            train_dataset = split['train']
            val_dataset   = split['test']
            test_dataset  = dataset_['test']

            train_idx = np.array(train_dataset['idx'])
            val_idx   = np.array(val_dataset['idx'])

            os.makedirs(os.path.join(output_dir, data_name), exist_ok=True)
            np.save(train_idx_file, train_idx)
            np.save(val_idx_file,   val_idx)

        
        dataset = DatasetDict({
            'train': train_dataset,
            'valid': val_dataset,
            'test':  test_dataset
        })
    elif 'glue' in data_name:
        if data_name == 'glue_sst2':
            name = 'sst2'
            text_key = 'sentence'
            label_key = 'label'
        elif data_name == 'glue_cola':
            name = 'cola'
            text_key = 'sentence'
            label_key = 'label'
        elif data_name == 'glue_mrpc':
            name = 'mrpc'
            text_key = ('sentence1', 'sentence2')
            label_key = 'label'
        elif data_name == 'glue_qqp':
            name = 'qqp'
            text_key = ('question1', 'question2')
            label_key = 'label'
        elif data_name == 'glue_qnli':
            name = 'qnli'
            text_key = ('question', 'sentence')
            label_key = 'label'
        elif data_name == 'glue_mnli':
            name = 'mnli'
            text_key = ('premise', 'hypothesis')
            label_key = 'label'
        elif data_name == 'glue_rte':
            name = 'rte'
            text_key = ('sentence1', 'sentence2')
            label_key = 'label'
        elif data_name == 'glue_wnli':
            name = 'wnli'
            text_key = ('sentence1', 'sentence2')
            label_key = 'label'
        
        dataset = load_dataset("glue", name)
        dataset = {("valid" if k == "validation" else k): v for k, v in dataset.items()}

    elif data_name == 'imdb':
        dataset_ = load_dataset("imdb")
        text_key = 'text'
        label_key = 'label'

        dataset_["train"]=dataset_["train"].add_column("idx", list(range(len(dataset_["train"]))))
        dataset_["test"]=dataset_["test"].add_column("idx", list(range(len(dataset_["test"]))))
        
        if os.path.isfile(os.path.join(output_dir, data_name,"./train_idx.npy")) and os.path.isfile(os.path.join(output_dir, data_name,"./valid_idx.npy")):
            train_idx = np.load(os.path.join(output_dir, data_name,"./train_idx.npy"))
            val_idx = np.load(os.path.join(output_dir, data_name,"./val_idx.npy"))

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
            
            os.makedirs(os.path.join(output_dir, data_name), exist_ok=True)
            np.save(os.path.join(output_dir, data_name,"./train_idx.npy"), train_idx)
            np.save(os.path.join(output_dir, data_name,"./val_idx.npy"), val_idx)


        dataset = DatasetDict({
            "train": train_dataset,
            "valid": val_dataset,
            "test": test_dataset
        })

    elif data_name == 'cose':

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

    elif data_name == 'movies':
        dataset_train = load_dataset("json", data_files={"train": os.path.join("//storage/personal/myhwang/NLP_FS/data/eraser/movies", "train.jsonl")})["train"]
        dataset_valid = load_dataset("json", data_files={"valid": os.path.join("//storage/personal/myhwang/NLP_FS/data/eraser/movies", "val.jsonl")})["valid"]
        dataset_test = load_dataset("json", data_files={"test": os.path.join("//storage/personal/myhwang/NLP_FS/data/eraser/movies", "test.jsonl")})["test"]
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
    elif data_name == 'bioasq':
        

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


    elif data_name == 'graph_sst2':
        
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


    if data_name == 'cose':
        
        label_key = 'label'
        doc_path = "/storage/personal/myhwang/NLP_FS/data/eraser/data/cose_simplified/docs.jsonl"
        texts = build_texts_from_docids(dataset[split_name], doc_path)

    elif data_name == 'movies':
        label_key = 'label'
        doc_path = "/storage/personal/myhwang/NLP_FS/data/eraser/movies/docs/"
        texts = load_doc_text(doc_path, dataset[split_name]["annotation_id"])
        

    elif data_name == 'bioasq':
        if tokenizer_type in ['BioMedLM']:
            label_key = 'label'
            text_key = ['context', 'question']
            # texts = {
            #     'context': dataset[split_name]['context'],
            #     'question': dataset[split_name]['question'],
            # }
            # texts = [{k: v for k, v in zip(text_key, vals)} for vals in zip(*[dataset[split_name][k] for k in text_key])]
            texts = [{k: row[k] for k in text_key} for row in dataset[split_name]]
        elif tokenizer_type == 'biolinkBert':
            label_key = 'label'
            text_key = ['context', 'question']
            texts = build_texts_from_qa(dataset[split_name], include_query=True, sep_token="[SEP]") 
            
        elif tokenizer_type in ['galactica']:
            label_key = 'label'
            text_key = ['context', 'question']
            texts = [{k: row[k] for k in text_key} for row in dataset[split_name]]
            


        # texts = dataset[split_name][text_key]
    elif isinstance(text_key, tuple):
        
        texts = list(zip(*[dataset[split_name][key] for key in text_key]))
    
    else:
        
        texts = dataset[split_name][text_key]

    
    
    graph_dataset = DependencyGraphDataset(
        texts,
        dataset[split_name][label_key], 
        graph_process_type=graph_type, 
        tokenizer_type=tokenizer_type,
        encoder_layer_depth = encoder_layer_depth,

        data_name = data_name,
        idx = dataset[split_name]['idx'] if 'idx' in dataset[split_name].features else None,
    )
    
    
    data_loader = DataLoader(
        graph_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn_pickle
    )
    
               
    save_dir = os.path.join(output_dir, data_name, split_name, tokenizer_type, graph_type, str(encoder_layer_depth))

    os.makedirs(save_dir, exist_ok=True)
    
    
    total_batches = len(data_loader)
    
    
    for batch_idx, batch in tqdm(enumerate(data_loader), 
                                total=total_batches, 
                                desc=f"Processing {split_name}-{tokenizer_type}-{graph_type}"):
        for i in range(len(batch['text'])):
            sample_dict = {
                'text': batch['text'][i],
                'label': batch['label'][i],
                'token_info': batch['token_info'][i],
                # 'sentences_token_info': batch['sentences_token_info'][i],
                'graph': batch['graph'][i],
                # 'sentence_graphs': batch['sentence_graphs'][i],
                # 'adj_matrix_full': batch['adj_matrix_full'][i].numpy(),
                # 'adj_matrix_per_sent': batch['adj_matrix_per_sent'][i].numpy(),
                'tokens': batch['tokens'][i],
                'token_ids': {
                    key: batch['token_ids'][key][i].numpy() 
                    for key in batch['token_ids'].keys()
                },
                'node2token': batch['node2token'][i],
                'max_sent_len': batch['max_sent_len'][i],
                # 'sent_lengths': batch['sent_lengths'][i],
                'root_positions': batch['root_positions'][i],
                'embedding_matrix_full': batch['embedding_matrix_full'][i].numpy(),
                # 'embedding_matrix_per_sent': batch['embedding_matrix_per_sent'][i].numpy()
                'raw_word': batch['raw_word'][i],
            }
            
            if 'dummy_positions' in batch:
                sample_dict['dummy_positions'] = batch['dummy_positions'][i]
            
            
            if 'sem_graph' in batch:
                sample_dict['sem_graph'] = batch['sem_graph'][i]

            
            global_idx = batch_idx * batch_size + i
            
            

            save_path = os.path.join(save_dir, f'sample_{global_idx}.pkl')

            with open(save_path, 'wb') as f:
                pickle.dump(sample_dict, f)
    
    print(f"{split_name} 데이터셋 저장 완료!")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='glue_sst2', choices=['ag_news', 'glue_sst2', 'glue_cola', 'imdb', 'cose', 'movies', 'bioasq', 'graph_sst2'])
parser.add_argument('--graph_process_type', type=str, default='direct_root',choices=['direct_root', 'dummy_root', 'neighbor_dummy_node'])
parser.add_argument('--tokenizer_type', type=str, default='deberta',choices=['xlnet', 'gpt2', 'deberta', 'roberta', 'BioMedLM', 'biolinkBert', 'galactica', 'bert', 'deberta_large', 'deberta_small'])
parser.add_argument('--output_dir', type=str, default='/storage/personal/myhwang/NLP_FS/data/')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--encoder_layer_depth', type=int, default=-1)  # 0~12
parser.add_argument('-c', '--cpu_start', type=int, default=0)

args = parser.parse_args()

def main():


    for split in ['train', 'valid', 'test']:
        process_and_save_dataset(args.data_name, split, args.output_dir, args.tokenizer_type, args.graph_process_type, args.batch_size, encoder_layer_depth = args.encoder_layer_depth)
        
        


if __name__ == "__main__":
    
    p = psutil.Process()
    try:
        p.cpu_affinity(list(range(args.cpu_start, args.cpu_start+5)))
        print(f"Worker CPU affinity: {p.cpu_affinity()}")
    except Exception as e:
        print(f"Worker CPU affinity: {e}")
    main()