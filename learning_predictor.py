import logging
import os
import sys
import pdb
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utill import save_checkpoint, get_all_metrics, save_test
from transformers import AutoModelForSequenceClassification

class Model(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        
        self.writer = SummaryWriter(log_dir=self.args.log_dir)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

    def train(self, train_loader, valid_loader, wandb):
        iteration = 0

        top_score = 0
        old_params = {name: param.clone() for name, param in self.model.named_parameters()}
        for epoch in tqdm(range(self.args.epochs), desc="Training Epochs"): 
            all_train_logits = []
            all_train_labels = []
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
            for i, data in enumerate(train_pbar):
                self.model.train()
                torch.cuda.empty_cache()
                iteration += 1

                outputs = self.model(data['full_graph'].to(self.args.device), data['meta_data'])
                loss_ce = F.cross_entropy(outputs.logits, data['full_graph']['y'].to(self.args.device))

                loss = loss_ce

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)

                print(f"\nIteration {iteration}:")
                print("Loss:", loss.item())

                print("CE:", loss_ce.item())
                if not self.args.wandb_off:
                    wandb.log({"train_loss": loss.item()}, step=iteration)
                    wandb.log({"train_loss_ce": loss_ce.item()}, step=iteration)
                all_train_logits.append(outputs.logits.detach().cpu())
                all_train_labels.append(data['full_graph']['y'].detach().cpu())
                self.optimizer.step()
                self.model.zero_grad(set_to_none=True)

                for name, param in self.model.named_parameters():
                    if ('predictor' not in name) or ('logits_proj' in name) or ('sequence_summary' in name)or ('pooler' in name) or ('classifier' in name):
                        if torch.equal(old_params[name], param):
                            print(f"Parameter {name} did not change")
                        else:
                            print(f"Parameter {name} changed")
                is_last_batch = ((iteration + 1) % len(train_loader) == 0)
                if is_last_batch:


                    print(f"Iteration {iteration}:") if is_last_batch else print(f"Iteration {iteration} (Epoch {epoch+1}):")
                    print("Train Loss:", loss.item())
                    print("Train Loss CE:", loss_ce.item())

                    train_logits = torch.cat(all_train_logits, dim=0)
                    train_labels = torch.cat(all_train_labels, dim=0)
                    train_preds = torch.argmax(train_logits, dim=1).cpu()
                    train_probs = torch.softmax(train_logits, dim=1).cpu().numpy()
                    
                    logging.debug(f"Iteration {iteration}:") if is_last_batch else logging.debug(f"Iteration {iteration} (Epoch {epoch+1}):")
                    logging.debug(f"Train Loss: {loss.item()} / Train Loss CE: {loss_ce.item()}")

                    train_metrics = get_all_metrics(
                        train_labels.cpu().numpy(), 
                        train_preds.numpy(), 
                        train_probs, 
                        n_classes=2,
                        prefix="train"
                    )
                    for key, value in train_metrics.items():
                        self.writer.add_scalar(key, value, iteration)
                    if not self.args.wandb_off:
                        wandb.log({"train_loss": loss.item()}, step=iteration)
                        wandb.log({"train_loss_ce": loss_ce.item()}, step=iteration)
                        wandb.log(train_metrics, step=iteration)

                    total_valid_loss = 0
                    total_valid_loss_ce = 0
                    all_valid_logits = []
                    all_valid_labels = []

                    self.model.eval()
                    torch.cuda.empty_cache()
                    
                    with torch.no_grad():
                        valid_pbar = tqdm(valid_loader, desc=f"Valid Itteration {iteration}", leave=False)
                        for data in valid_pbar:
                            outputs = self.model(data['full_graph'].to(self.args.device), data['meta_data'])

                            loss_ce = F.cross_entropy(outputs.logits, data['full_graph']['y'].to(self.args.device))

                            loss = loss_ce
                            total_valid_loss += loss.item()
                            total_valid_loss_ce += loss_ce.item()
                            all_valid_logits.append(outputs.logits.detach().cpu())
                            all_valid_labels.append(data['full_graph']['y'].detach().cpu())

                        total_valid_loss = total_valid_loss / len(valid_loader)
                        total_valid_loss_ce = total_valid_loss_ce / len(valid_loader)
                        valid_logits = torch.cat(all_valid_logits, dim=0)
                        valid_labels = torch.cat(all_valid_labels, dim=0)
                        

                        valid_preds = torch.argmax(valid_logits, dim=1).cpu()
                        valid_probs = torch.softmax(valid_logits, dim=1).cpu().numpy()
                        print("Valid Loss:", total_valid_loss)
                        print("Valid Loss CE:", total_valid_loss_ce)
                        valid_metrics = get_all_metrics(
                            valid_labels.cpu().numpy(), 
                            valid_preds.numpy(), 
                            valid_probs, 
                            n_classes=2,
                            prefix="valid"
                        )

                        logging.debug(f"Valid Loss: {total_valid_loss} / Valid Loss CE: {total_valid_loss_ce}")
                        self.writer.add_scalar('valid_loss', total_valid_loss, iteration)
                        self.writer.add_scalar('valid_loss_ce', total_valid_loss_ce, iteration)
                        for key, value in valid_metrics.items():
                            self.writer.add_scalar(key, value, iteration)
                        if not self.args.wandb_off:
                            wandb.log({"valid_loss": total_valid_loss}, step=iteration)
                            wandb.log({"valid_loss_ce": total_valid_loss_ce}, step=iteration)
                            wandb.log(valid_metrics, step=iteration)
                                            
                    total_valid_score = valid_metrics['valid_roc_auc_class_1']
                    if (total_valid_score > top_score): 
                        top_score = total_valid_score
                        state = {
                                    'epoch ' : epoch,
                                    'iteration ' : iteration,
                                    'data' : self.args.data_name,
                                    'lr' : self.args.lr,
                                    'valid_loss' : total_valid_loss,
                                    'valid_loss_ce' : total_valid_loss_ce,
                                }
                        logging.debug(f"Best_Iteration: {iteration}")
                        torch.save(self.model, os.path.join(self.writer.log_dir, f'{self.args.tokenizer_type}_{self.args.seed}_{self.args.current_time}.pt'))

                        for key, value in valid_metrics.items():
                            state[key] = value
                        save_checkpoint(os.path.join(self.writer.log_dir), state)
                                        
                        os.makedirs(f'./predictor_weights/{self.args.data_name}', exist_ok=True)
                        if self.args.tokenizer_type == 'xlnet':
                            torch.save(self.model.sequence_summary.state_dict(), f'./predictor_weights/{self.args.data_name}/sequence_summary_weights.pt')
                            torch.save(self.model.logits_proj.state_dict(), f'./predictor_weights/{self.args.data_name}/logits_proj_weights.pt')

                        elif self.args.tokenizer_type == 'deberta':
                            torch.save(self.model.predictor.pooler.state_dict(), f'./predictor_weights/{self.args.data_name}/pooler_weights.pt')
                            torch.save(self.model.predictor.classifier.state_dict(), f'./predictor_weights/{self.args.data_name}/classifier_weights_weights.pt')
                            
                        elif self.args.tokenizer_type == 'gpt2':
                            torch.save(self.model.predictor.score.state_dict(), f'./predictor_weights/{self.args.data_name}/score_weights.pt')
                        elif self.args.tokenizer_type == 'BioMedLM':
                            torch.save(self.model.predictor.classifier.state_dict(), f'./predictor_weights/{self.args.data_name}/classifier_weights.pt')
                            self.model.predictor.save_pretrained(f"./predictor_weights/bioasq/BioMedLM_{self.args.current_time}")
                        elif self.args.tokenizer_type == 'biolinkBert':
                            torch.save(self.model.predictor.bert.pooler.state_dict(), f'./predictor_weights/{self.args.data_name}/pooler_weights_{self.args.current_time}.pt')
                            self.model.predictor.save_pretrained(f"./predictor_weights/bioasq/biolinkBert_{self.args.current_time}")
                            torch.save(self.model.predictor.classifier.state_dict(), f'./predictor_weights/{self.args.data_name}/classifier_weights_{self.args.current_time}.pt')
                            self.model.predictor.save_pretrained(f"./predictor_weights/bioasq/biolinkBert_{self.args.current_time}")

    def test(self, test_loader, wandb):                 
        

        print('{} testing...'.format(self.args.tokenizer_type))

        self.model = torch.load(os.path.join(self.writer.log_dir, f'{self.args.tokenizer_type}_{self.args.seed}_{self.args.current_time}.pt'))

        total_test_loss = 0
        total_test_loss_ce = 0
        all_test_logits = []
        all_test_labels = []
        self.model.eval()
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f"Test Itteration", leave=False)
            for data in test_pbar:
                outputs = self.model(data['full_graph'].to(self.args.device), data['meta_data'])
                loss_ce = F.cross_entropy(outputs.logits, data['full_graph']['y'].to(self.args.device))
                loss = loss_ce 
                total_test_loss += loss.item()
                total_test_loss_ce += loss_ce.item()
                all_test_logits.append(outputs.logits.detach().cpu())
                all_test_labels.append(data['full_graph']['y'].detach().cpu())

            test_loss = total_test_loss / len(test_loader)
            test_loss_ce = total_test_loss_ce / len(test_loader)
            test_logits = torch.cat(all_test_logits, dim=0)
            test_labels = torch.cat(all_test_labels, dim=0)
            
            test_preds = torch.argmax(test_logits, dim=1).cpu()
            test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()

            print("Test Loss:", test_loss)
            print("Test Loss CE:", test_loss_ce)
            test_metrics = get_all_metrics(
                test_labels.cpu().numpy(), 
                test_preds.numpy(), 
                test_probs, 
                n_classes=2,
                prefix="test"
            )   

            logging.debug(f"\nTest Loss: {test_loss} / Test Loss CE: {test_loss_ce}")
            self.writer.add_scalar('test_loss', test_loss)
            self.writer.add_scalar('test_loss_ce', test_loss_ce)
            for key, value in test_metrics.items():
                self.writer.add_scalar(key, value)

            if not self.args.wandb_off:
                wandb.log({"test_loss": test_loss})
                wandb.log({"test_loss_ce": test_loss_ce})
                wandb.log(test_metrics)


        state = {
                'data' : self.args.data_name,
                'lr' : self.args.lr,
                'test_loss' : test_loss,
                'test_loss_ce' : test_loss_ce,
                }
        for key, value in test_metrics.items():
            state[key] = value
        save_test(os.path.join(self.writer.log_dir), state)                
        pass
