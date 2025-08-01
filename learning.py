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
from utill import save_checkpoint, get_all_metrics, save_test, batch_to_adj_matrices, get_rank
import pickle
import torch.nn as nn

class Model(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.lambda_reg_scheduler = kwargs['lambda_reg_scheduler']
        self.CE_loss = nn.CrossEntropyLoss(reduction='none')
        
        self.writer = SummaryWriter(log_dir=self.args.log_dir)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

    def train(self, train_loader, valid_loader, wandb):
        iteration = 0
        top_score = float('inf')
        old_params = {name: param.clone() for name, param in self.model.named_parameters()}

        for epoch in tqdm(range(self.args.epochs), desc="Training Epochs"): 
            all_train_logits = []
            all_train_labels = []
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
            for i, data in enumerate(train_pbar):
                # 
                self.model.train()
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                # 
                iteration += 1

                
                outputs, regularizer, gate = self.model(data['full_graph'].to(self.args.device), data['meta_data'])
                    

                loss_ce = F.cross_entropy(outputs.logits, data['full_graph']['y'].to(self.args.device))
                
                
                loss_reg = torch.nanmean(regularizer)
                
                num_above_threshold = (gate > self.args.gate_threshold).sum()
                print(f"num_above_threshold: {num_above_threshold}/{gate.shape[0]}")
                loss = loss_ce + self.args.lambda_reg * loss_reg

                loss.backward(retain_graph=True)
                print(f"\nIteration {iteration}:")
                print("Loss: {:.5f}".format(loss.item()))
                print("Regularizer: {:.5f}".format(loss_reg.item()))
                print("CE: {:.5f}".format(loss_ce.item()))
                
                if not self.args.wandb_off:
                    wandb.log({"train_loss": loss.item()}, step=iteration)
                    wandb.log({"train_loss_ce": loss_ce.item()}, step=iteration)
                    wandb.log({"train_loss_reg": loss_reg.item()}, step=iteration)
                all_train_logits.append(outputs.logits.detach().cpu())
                all_train_labels.append(data['full_graph']['y'].detach().cpu())

                self.optimizer.step()
                self.model.zero_grad(set_to_none=True)

                for name, param in self.model.named_parameters():
                    if ('predictor' not in name) or ('logits_proj' in name) or ('sequence_summary' in name) or ('gate' in name):
                        if torch.equal(old_params[name], param):
                            print(f"Parameter {name} did not change")
                        # else:
                        #     print(f"Parameter {name} changed")

                print(f"Iteration {iteration} (Epoch {epoch+1}):")
                print("Train Loss: {:.5f}".format(loss.item()))
                print("Train Loss CE: {:.5f}".format(loss_ce.item()))
                print("Train Loss Reg: {:.5f}".format(loss_reg.item()))

                train_logits = torch.cat(all_train_logits, dim=0)
                train_labels = torch.cat(all_train_labels, dim=0)

                train_preds = torch.argmax(train_logits, dim=1).cpu()
                train_probs = torch.softmax(train_logits, dim=1).cpu().numpy()
                
                logging.debug(f"Iteration {iteration} (Epoch {epoch+1}):")
                logging.debug(f"Train Loss: {loss.item()} / Train Loss CE: {loss_ce.item()} / Train Loss Reg: {loss_reg.item()}")
                self.writer.add_scalar('Iteration', iteration, iteration)
                self.writer.add_scalar('train_loss', loss.item(), iteration)
                self.writer.add_scalar('train_loss_ce', loss_ce.item(), iteration)
                self.writer.add_scalar('train_loss_reg', loss_reg.item(), iteration)
                train_metrics = get_all_metrics(
                    train_labels.cpu().numpy(), 
                    train_preds.numpy(), 
                    train_probs, 
                    n_classes=2 if self.args.data_name != "ag_news" else 4, 
                    prefix="train"
                )

                for key, value in train_metrics.items():
                    self.writer.add_scalar(key, value, iteration)
                if not self.args.wandb_off:
                    wandb.log(train_metrics, step=iteration)


                is_last_batch = (iteration == len(train_loader) - 1)

                if iteration % 100 == 0 or is_last_batch:
                    self.optimizer.zero_grad()

                    total_valid_loss = 0
                    total_valid_loss_ce = 0
                    total_valid_loss_reg = 0
                    all_valid_logits = []
                    all_valid_labels = []

                    total_above_threshold = 0
                    total_valid_tokens = 0
                    self.model.eval()
                    with torch.no_grad():
                        valid_pbar = tqdm(valid_loader, desc=f"Valid Itteration {iteration} (Epoch {epoch+1}):", leave=False)
                        for data in valid_pbar:
                            
                            outputs, regularizer, gate = self.model(data['full_graph'].to(self.args.device), data['meta_data'], test=True)
                            
                            loss_ce = F.cross_entropy(outputs.logits, data['full_graph']['y'].to(self.args.device))
                            loss_reg = torch.nanmean(regularizer)
                            loss = loss_ce
                            total_valid_loss += loss.item()
                            total_valid_loss_ce += loss_ce.item()
                            total_valid_loss_reg += loss_reg.item()
                            all_valid_logits.append(outputs.logits.detach().cpu())
                            all_valid_labels.append(data['full_graph']['y'].detach().cpu())
                            num_above_threshold = (gate > self.args.gate_threshold).sum()

                            total_above_threshold += num_above_threshold
                            total_valid_tokens += gate.numel()

                            
                        total_valid_loss = total_valid_loss / len(valid_loader)
                        total_valid_loss_ce = total_valid_loss_ce / len(valid_loader)
                        total_valid_loss_reg = total_valid_loss_reg / len(valid_loader)
                        valid_logits = torch.cat(all_valid_logits, dim=0)
                        valid_labels = torch.cat(all_valid_labels, dim=0)

                        valid_preds = torch.argmax(valid_logits, dim=1).cpu()
                        valid_probs = torch.softmax(valid_logits, dim=1).cpu().numpy()
                        print("Valid Loss: {:.5f}".format(total_valid_loss))
                        print("Valid Loss CE: {:.5f}".format(total_valid_loss_ce))
                        print("Valid Loss Reg: {:.5f}".format(total_valid_loss_reg))
                        valid_metrics = get_all_metrics(
                            valid_labels.cpu().numpy(), 
                            valid_preds.numpy(), 
                            valid_probs, 
                            n_classes=2 if self.args.data_name != "ag_news" else 4, 
                            prefix="valid"
                        )
                        average_above_threshold = total_above_threshold / total_valid_tokens
                        print(f"Ratio of valid gates above threshold: {average_above_threshold:.4f}")
                        print(f"Total number of valid gates above threshold: {total_above_threshold}")
                        print(f"Total number of valid gates: {total_valid_tokens}")

                        logging.debug(f"Valid Loss: {total_valid_loss} / Valid Loss CE: {total_valid_loss_ce} / Valid Loss Reg: {total_valid_loss_reg}")
                        self.writer.add_scalar('valid_loss', total_valid_loss, iteration)
                        self.writer.add_scalar('valid_loss_ce', total_valid_loss_ce, iteration)
                        self.writer.add_scalar('valid_loss_reg', total_valid_loss_reg, iteration)
                        for key, value in valid_metrics.items():
                            self.writer.add_scalar(key, value, iteration)
                        self.writer.add_scalar(f"Ratio of valid gates above threshold", average_above_threshold, iteration)
                        self.writer.add_scalar(f"Total number of valid gates above threshold", total_above_threshold, iteration)
                        self.writer.add_scalar(f"Total number of valid gates", total_valid_tokens, iteration)
                        if not self.args.wandb_off:
                            wandb.log({"valid_loss": total_valid_loss}, step=iteration)
                            wandb.log({"valid_loss_ce": total_valid_loss_ce}, step=iteration)
                            wandb.log({"valid_loss_reg": total_valid_loss_reg}, step=iteration)
                            wandb.log(valid_metrics, step=iteration)
                            wandb.log({"Ratio of valid gates above threshold": average_above_threshold.item()}, step=iteration)
                            wandb.log({"Total number of valid gates above threshold": total_above_threshold.item()}, step=iteration)
                            wandb.log({"Total number of valid gates": total_valid_tokens}, step=iteration)
                        
                    if (total_valid_loss < top_score): 
                        state = {
                                    'epoch ' : epoch,
                                    'iteration ' : iteration,
                                    'data' : self.args.data_name,
                                    'lr' : self.args.lr,
                                    'valid_loss' : total_valid_loss,
                                    'valid_loss_ce' : total_valid_loss_ce,
                                    'valid_loss_reg' : total_valid_loss_reg,
                                    'Ratio of valid gates above threshold' : average_above_threshold.item(),
                                    'Total number of valid gates above threshold' : total_above_threshold.item(),
                                    'Total number of valid gates' : total_valid_tokens,
                                }
                        top_score = total_valid_loss
                        logging.debug(f"Best_Iteration: {iteration}")
                        torch.save(self.model, os.path.join(self.writer.log_dir, f'{self.args.tokenizer_type}_{self.args.seed}.pt'))

                        
                        for key, value in valid_metrics.items():
                            state[key] = value
                        save_checkpoint(os.path.join(self.writer.log_dir), state)

    def test(self, test_loader, wandb):                 
        print('{} testing...'.format(self.args.tokenizer_type))
        model = torch.load(os.path.join(self.writer.log_dir, f'{self.args.tokenizer_type}_{self.args.seed}.pt'))
        total_test_loss = 0
        total_test_loss_ce = 0
        total_test_loss_reg = 0
        all_test_logits = []
        all_test_labels = []
        gate_info = {}
        total_above_threshold = 0
        total_test_tokens = 0
        model.eval()
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f"Test Itteration", leave=False)
            for data in test_pbar:

                outputs = model(data['full_graph'].to(self.args.device), data['meta_data'], baseline = True)
                

                loss_ce = F.cross_entropy(outputs.logits, data['full_graph']['y'].to(self.args.device))
                loss_reg = torch.nanmean(regularizer)
                loss = loss_ce + self.args.lambda_reg * loss_reg
                total_test_loss += loss.item()
                total_test_loss_ce += loss_ce.item()
                total_test_loss_reg += loss_reg.item()
                all_test_logits.append(outputs.logits.detach().cpu())
                all_test_labels.append(data['full_graph']['y'].detach().cpu())
                
                num_above_threshold = (test_gate_inputs.squeeze(-1) > self.args.gate_threshold).sum()
                total_above_threshold += num_above_threshold
                total_test_tokens += test_gate_inputs.numel()
                sample_indices = data['meta_data']['idx'].cpu().numpy()
                ptr = data['full_graph']['ptr'].cpu().numpy()
                
                print(f"test gate_inputs: {test_gate_inputs}")
                print(f"num_above_threshold: {num_above_threshold}/{test_gate_inputs.shape[0]}")


                for batch_idx in range(len(ptr) - 1):
                    start_idx = ptr[batch_idx]
                    end_idx = ptr[batch_idx + 1]
                    
                    text = data['meta_data']['text'][batch_idx]
                    test_gates = test_gate_inputs[start_idx:end_idx].cpu().numpy()
                    label = data['full_graph']['y'][batch_idx].cpu().numpy()
                    
                    sample_id = int(sample_indices[batch_idx])
                    gate_info[sample_id] = {
                        'text': text,
                        'test_gates': test_gates,
                        'label': label
                    }


            test_loss = total_test_loss / len(test_loader)
            test_loss_ce = total_test_loss_ce / len(test_loader)
            test_loss_reg = total_test_loss_reg / len(test_loader)
            test_logits = torch.cat(all_test_logits, dim=0)
            test_labels = torch.cat(all_test_labels, dim=0)
            test_preds = torch.argmax(test_logits, dim=1).cpu()
            test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()

            print("Test Loss: {:.5f}".format(test_loss))
            print("Test Loss CE: {:.5f}".format(test_loss_ce))
            print("Test Loss Reg: {:.5f}".format(test_loss_reg))
            test_metrics = get_all_metrics(
                test_labels.cpu().numpy(), 
                test_preds.numpy(), 
                test_probs, 
                n_classes=2 if self.args.data_name != "ag_news" else 4, 
                prefix="test"
            )   
            average_above_threshold = total_above_threshold / total_test_tokens
            print(f"Ratio of test gates above threshold: {average_above_threshold:.4f}")
            print(f"Total number of test gates above threshold: {total_above_threshold}")
            print(f"Total number of test gates: {total_test_tokens}")

            logging.debug(f"\nTest Loss: {test_loss} / Test Loss CE: {test_loss_ce} / Test Loss Reg: {test_loss_reg}")
            self.writer.add_scalar('test_loss', test_loss)
            self.writer.add_scalar('test_loss_ce', test_loss_ce)
            self.writer.add_scalar('test_loss_reg', test_loss_reg)
            for key, value in test_metrics.items():
                self.writer.add_scalar(key, value)
            self.writer.add_scalar(f"Ratio of test gates above threshold", average_above_threshold.item())
            self.writer.add_scalar(f"Total number of test gates above threshold", total_above_threshold.item())
            self.writer.add_scalar(f"Total number of test gates", total_test_tokens)

            if not self.args.wandb_off:
                wandb.log({"test_loss": test_loss})
                wandb.log({"test_loss_ce": test_loss_ce})
                wandb.log({"test_loss_reg": test_loss_reg})    
                wandb.log(test_metrics)
                wandb.log({"Ratio of test gates above threshold": average_above_threshold.item()})
                wandb.log({"Total number of test gates above threshold": total_above_threshold.item()})
                wandb.log({"Total number of test gates": total_test_tokens})


        state = {
                    'data' : self.args.data_name,
                    'lr' : self.args.lr,
                    'test_loss' : test_loss,
                    'test_loss_ce' : test_loss_ce,
                    'test_loss_reg' : test_loss_reg,
                }

        for key, value in test_metrics.items():
            state[key] = value
        state['Ratio of test gates above threshold'] = average_above_threshold.item()
        state['Total number of test gates above threshold'] = total_above_threshold.item()
        state['Total number of test gates'] = total_test_tokens
        save_test(os.path.join(self.writer.log_dir), state)                

        gate_save_path = os.path.join(self.writer.log_dir, 'test_gate_info.pkl')
        with open(gate_save_path, 'wb') as f:
            pickle.dump(gate_info, f)
        print(os.path.join(self.writer.log_dir))



    def train_RL(self, train_loader, valid_loader, wandb):
            iteration = 0

            top_score = 0
            top_score2 = 0
            train_score_matrices = {}
            for epoch in tqdm(range(self.args.epochs), desc="Training Epochs"): 
                all_train_logits = []
                all_train_labels = []
                train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
                for i, data in enumerate(train_pbar):
                    # 
                    self.model.train()
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    iteration += 1
                    # pdb.set_trace()
                    data['meta_data']['idx'] = data['full_graph']['idx']
                    if self.args.data_name =="graph_sst2" and self.args.adj_type =='cross':
                        inputs = (data['full_graph'].to(self.args.device), data['external_graph'].to(self.args.device))
                        out = self.model(inputs, data['meta_data'])
                    elif self.args.data_name =="graph_sst2" and self.args.adj_type =='semantic':
                        out = self.model(data['external_graph'].to(self.args.device), data['meta_data'])
                    else:
                        out = self.model(data['full_graph'].to(self.args.device), data['meta_data'])
                    outputs = out['outputs']
                    regularizer = out['regularizer']
                    loss_smo = out['loss_smo']
                    gate = out['token_soft_gate']
                    hard_gate = out['token_hard_gate']
                    total_token = out['total_token']
                    if self.args.mix_up_rate != 0:
                        mixup_reg = out['mixup_reg']

                    loss_ce = self.CE_loss(outputs.logits, data['full_graph']['y'].repeat(self.args.num_samples).to(self.args.device))
                    loss_reg = torch.nanmean(regularizer)


                    base_outputs = self.model(data['full_graph'].to(self.args.device), data['meta_data'], baseline = True)
                    base_loss_ce = self.CE_loss(base_outputs.logits, data['full_graph']['y'].to(self.args.device)).squeeze()

                    eps = 1e-8
                    log_probs = torch.log(hard_gate * gate + (1 - hard_gate) * (1 - gate)+ eps).squeeze()
                        

                    base_loss_ce = base_loss_ce.repeat(self.args.num_samples)
                    

                    if self.args.policy_KL:
                        temperature =1 
                        teacher_probs = F.softmax(base_outputs.logits.repeat_interleave(self.args.num_samples, dim=0) / temperature, dim=-1)
                        student_log_probs = F.log_softmax(outputs.logits / temperature, dim=-1)
                        kl=(F.kl_div(student_log_probs, teacher_probs, reduction='none') * (temperature**2)).sum(dim=1)
                        if self.args.lambda_kl < 1:
                            policy_KL = (-(kl).unsqueeze(-1) * log_probs).mean()
                            policy_label = (-(loss_ce - base_loss_ce).unsqueeze(-1) * log_probs ).mean()
                            policy_loss = self.args.lambda_kl *policy_KL + (1-self.args.lambda_kl) * policy_label
                        else:
                            policy_loss = (-(kl).unsqueeze(-1) * log_probs).mean()
                    elif self.args.policy_soft == -1:
                        if  self.args.baseline_type == 'basic':
                            advantage=(loss_ce - base_loss_ce).unsqueeze(-1) 
                        elif self.args.baseline_type == 'div_mean':
                            advantage = (loss_ce - base_loss_ce).unsqueeze(-1) 
                            advantage = advantage - advantage.mean()
                        elif self.args.baseline_type == 'mean':
                            advantage = loss_ce.unsqueeze(-1)
                            advantage = advantage - advantage.mean()
                            
                        policy_loss = (-advantage * log_probs ).mean()
                    else:
                        policy_loss = (-(F.softplus((loss_ce - base_loss_ce).unsqueeze(-1), beta=self.args.policy_soft) * log_probs )).mean()

                    avg_ce_loss = loss_ce.mean()
                    avg_base_ce_loss = base_loss_ce.mean()

                    num_above_threshold = (gate > self.args.gate_threshold).sum()
                    print(f"num_above_threshold: {num_above_threshold}/{total_token}")

                    total_loss = -policy_loss + self.args.lambda_reg*loss_reg + self.args.lambda_smo*loss_smo

                    if self.args.mix_up_rate != 0:
                        mixup_reg.backward(retain_graph=True)
                    if self.args.reg_update_step != 0:
                        if (iteration % self.args.reg_update_step) == 0:
                            (self.args.lambda_reg*loss_reg + self.args.lambda_smo*loss_smo).backward(retain_graph=True)
                        (-policy_loss).backward()
                        
                    else:
                        total_loss.backward()

                    
                    if not self.args.wandb_off:
                        wandb.log({"train_policy_loss": policy_loss.item()}, step=iteration)
                        wandb.log({"train_loss_ce": avg_ce_loss.item()}, step=iteration)
                        wandb.log({"train_loss_reg": loss_reg.item()}, step=iteration)
                        wandb.log({"train_loss_smo": loss_smo.item()}, step=iteration)
                        if self.args.policy_KL:  
                            wandb.log({"train_kl": kl.mean().item()}, step=iteration)    
                        
                                    
                        
                        if self.args.mix_up_rate != 0:
                            wandb.log({"train_loss_mixup": mixup_reg.item()}, step=iteration)
                    all_train_logits.append(outputs.logits.detach().cpu())
                    all_train_labels.append(data['full_graph']['y'].repeat(self.args.num_samples).detach().cpu())

                    self.optimizer.step()
                    self.model.zero_grad(set_to_none=True)


                    print(f"Iteration {iteration} (Epoch {epoch+1}):")
                    print("Train Loss policy: {:.5f}".format(policy_loss.item()))
                    print("Train Loss CE: {:.5f}".format(avg_ce_loss.item()))
                    print("Train Loss Reg: {:.5f}".format(loss_reg.item()))
                    print("Train Loss Smo: {:.5f}".format(loss_smo.item()))
                    if self.args.mix_up_rate != 0:
                        print("Regularizer mixup: {:.5f}".format(mixup_reg.item()))
                    # 
                    train_logits = torch.cat(all_train_logits, dim=0)
                    train_labels = torch.cat(all_train_labels, dim=0)
                    train_preds = torch.argmax(train_logits, dim=1).cpu()
                    train_probs = torch.softmax(train_logits, dim=1).cpu().numpy()
                    
                    logging.debug(f"Iteration {iteration} (Epoch {epoch+1}):")
                    logging.debug(f"Train policy_loss: {policy_loss.item()} / Train Loss CE: {avg_ce_loss.item()} / Train Loss Base CE: {avg_base_ce_loss.item()} / Train Loss Reg: {loss_reg.item()}")
                    self.writer.add_scalar('Iteration', iteration, iteration)
                    self.writer.add_scalar('train_policy_loss', policy_loss.item(), iteration)
                    self.writer.add_scalar('train_loss_ce', avg_ce_loss.item(), iteration)
                    self.writer.add_scalar('train_loss_base_ce', avg_base_ce_loss.item(), iteration)
                    
                    self.writer.add_scalar('train_loss_reg', loss_reg.item(), iteration)
                    self.writer.add_scalar('train_loss_smo', loss_smo.item(), iteration)

                    if self.args.mix_up_rate != 0:
                        self.writer.add_scalar('train_loss_mixup', mixup_reg.item(), iteration)
                    
                    train_metrics = get_all_metrics(
                        train_labels.cpu().numpy(), 
                        train_preds.numpy(), 
                        train_probs, 
                        n_classes=2 if self.args.data_name != "ag_news" else 4, 
                        prefix="train"
                    )

                    for key, value in train_metrics.items():
                        self.writer.add_scalar(key, value, iteration)
                    if not self.args.wandb_off: 
                        wandb.log(train_metrics, step=iteration)



                    is_last_batch = ((iteration + 1) % len(train_loader) == 0)

                    if self.args.data_name  in  ['glue_sst2', 'glue_cola', 'imdb', 'cose', 'movies', 'bioasq']:
                        valid_iter = 200
                    elif self.args.data_name in  ['ag_news']:
                        valid_iter = 500
                    else:
                        valid_iter = len(train_loader)
                    if iteration % valid_iter == 0 or is_last_batch:


                        total_valid_loss = 0
                        total_valid_score = 0
                        total_valid_score2 = 0
                        total_valid_policy_loss = 0
                        total_valid_loss_ce = 0
                        total_valid_loss_reg = 0
                        total_valid_loss_smo = 0
                        total_valid_loss_mixup = 0
                        total_valid_kl = 0
                        all_valid_logits = []
                        all_valid_labels = []

                        all_filtered_tokens = {}
                        all_filtered_words = {}
                        all_grouped_prob = {}
                        all_score_matrices = {}
                        total_above_threshold = 0
                        total_valid_tokens = 0
                        self.model.eval()
                        with torch.no_grad():
                            valid_pbar = tqdm(valid_loader, desc=f"Valid Itteration {iteration} (Epoch {epoch+1}):", leave=False)
                            for data in valid_pbar:
                                
                                data['meta_data']['idx'] = data['full_graph']['idx']
                                
                                if self.args.data_name =="graph_sst2" and self.args.adj_type =='cross':
                                    inputs = (data['full_graph'].to(self.args.device), data['external_graph'].to(self.args.device))
                                    out = self.model(inputs, data['meta_data'], test=True)
                                elif self.args.data_name =="graph_sst2" and self.args.adj_type =='semantic':
                                    out = self.model(data['external_graph'].to(self.args.device), data['meta_data'], test=True)
                                else:
                                    out = self.model(data['full_graph'].to(self.args.device), data['meta_data'], test=True)
                                outputs = out['outputs']
                                regularizer = out['regularizer']
                                loss_smo = out['loss_smo']
                                gate = out['token_soft_gate']
                                hard_gate = out['token_hard_gate']
                                total_token = out['total_token']
                                batch_filtered_token = out['filtered_tokens']
                                batch_filtered_words = out['filtered_words']
                                if self.args.mix_up_rate != 0:
                                    mixup_reg = out['mixup_reg']
                                
                                loss_ce = self.CE_loss(outputs.logits, data['full_graph']['y'].repeat(self.args.num_samples).to(self.args.device))
                                loss_reg = torch.nanmean(regularizer)

                                base_outputs = self.model(data['full_graph'].to(self.args.device), data['meta_data'], baseline = True)
                                base_loss_ce = self.CE_loss(base_outputs.logits, data['full_graph']['y'].to(self.args.device)).squeeze()
                                
                                eps = 1e-8
                                log_probs = torch.log(hard_gate * gate + (1 - hard_gate) * (1 - gate)+ eps).squeeze()

                                base_loss_ce = base_loss_ce.repeat(self.args.num_samples)

                                temperature =1 
                                teacher_probs = F.softmax(base_outputs.logits.repeat_interleave(self.args.num_samples, dim=0) / temperature, dim=-1)
                                student_log_probs = F.log_softmax(outputs.logits / temperature, dim=-1)
                                kl=(F.kl_div(student_log_probs, teacher_probs, reduction='none') * (temperature**2)).sum(dim=1)
                                if self.args.policy_KL:
                                    if self.args.lambda_kl < 1:
                                        policy_KL = (-(kl).unsqueeze(-1) * log_probs).mean()
                                        policy_label = (-(loss_ce - base_loss_ce).unsqueeze(-1) * log_probs ).mean()
                                        policy_loss = self.args.lambda_kl *policy_KL + (1-self.args.lambda_kl) * policy_label
                                    else:
                                        policy_loss = (-(kl).unsqueeze(-1) * log_probs).mean() 
                                elif self.args.policy_soft == -1:
                                    if  self.args.baseline_type == 'basic':
                                        advantage=(loss_ce - base_loss_ce).unsqueeze(-1) 
                                    elif self.args.baseline_type == 'div_mean':
                                        advantage = (loss_ce - base_loss_ce).unsqueeze(-1) 
                                        advantage = advantage - advantage.mean()
                                    elif self.args.baseline_type == 'mean':
                                        advantage = loss_ce.unsqueeze(-1)
                                        advantage = advantage - advantage.mean()
                                        
                                    policy_loss = (-advantage * log_probs ).mean()
                                else:
                                    policy_loss = (-(F.softplus((loss_ce - base_loss_ce).unsqueeze(-1), beta=self.args.policy_soft) * log_probs )).mean()
                                    

                                num_above_threshold = (gate > self.args.gate_threshold).sum()
                                print(f"num_above_threshold: {num_above_threshold}/{total_token}")

                                total_valid_loss+= (-policy_loss + self.args.lambda_reg*loss_reg + self.args.lambda_smo*loss_smo).item()

                                total_valid_policy_loss += policy_loss.item()
                                total_valid_loss_ce += loss_ce.mean().item()
                                total_valid_loss_reg += loss_reg.item()
                                total_valid_loss_smo += loss_smo.item()
                                total_valid_kl += kl.mean().item()
                                if self.args.mix_up_rate != 0:
                                    total_valid_loss_mixup += mixup_reg.item()
                                all_valid_logits.append(outputs.logits.detach().cpu())
                                all_valid_labels.append(data['full_graph']['y'].repeat(self.args.num_samples).detach().cpu())


                                total_above_threshold += num_above_threshold
                                total_valid_tokens += total_token
                                all_filtered_tokens.update(batch_filtered_token)
                                all_filtered_words.update(batch_filtered_words)

                            save_grouped_prob_path = os.path.join(self.writer.log_dir, f"grouped_prob.pkl")
                            with open(save_grouped_prob_path, "wb") as f:
                                pickle.dump(all_grouped_prob, f)
                            print(f"Filtered words saved to {save_grouped_prob_path}")

                            total_valid_loss = total_valid_loss / len(valid_loader)
                            total_valid_policy_loss = total_valid_policy_loss / len(valid_loader)
                            total_valid_loss_ce = total_valid_loss_ce / len(valid_loader)
                            total_valid_loss_reg = total_valid_loss_reg / len(valid_loader)
                            total_valid_loss_smo = total_valid_loss_smo / len(valid_loader)
                            total_valid_loss_mixup = total_valid_loss_mixup / len(valid_loader)
                            total_valid_kl = total_valid_kl / len(valid_loader)

                            valid_logits = torch.cat(all_valid_logits, dim=0)
                            valid_labels = torch.cat(all_valid_labels, dim=0)
                            valid_preds = torch.argmax(valid_logits, dim=1).cpu()
                            valid_probs = torch.softmax(valid_logits, dim=1).cpu().numpy()
                            print("Valid Loss: {:.5f}".format(total_valid_loss))
                            print("Valid Loss policy: {:.5f}".format(total_valid_policy_loss))
                            print("Valid Loss CE: {:.5f}".format(total_valid_loss_ce))
                            print("Valid Loss Reg: {:.5f}".format(total_valid_loss_reg))
                            print("Valid Loss Smo: {:.5f}".format(total_valid_loss_smo))
                            if self.args.mix_up_rate != 0:
                                print("Valid Loss mixup: {:.5f}".format(total_valid_loss_mixup))
                            valid_metrics = get_all_metrics(
                                valid_labels.cpu().numpy(), 
                                valid_preds.numpy(), 
                                valid_probs, 
                                n_classes=2 if self.args.data_name != "ag_news" else 4, 
                                prefix="valid"
                            )
                            average_above_threshold = total_above_threshold / total_valid_tokens
                            print(f"Ratio of valid gates above threshold: {average_above_threshold:.4f}")
                            print(f"Total number of valid gates above threshold: {total_above_threshold}")
                            print(f"Total number of valid gates: {total_valid_tokens}")

                            logging.debug(f"Valid loss: {total_valid_loss} / Valid policy_loss: {total_valid_policy_loss} / Valid Loss CE: {total_valid_loss_ce} / Valid Loss Reg: {total_valid_loss_reg}")
                            self.writer.add_scalar('valid_policy_loss', total_valid_policy_loss, iteration)
                            self.writer.add_scalar('valid_loss_ce', total_valid_loss_ce, iteration)
                            self.writer.add_scalar('valid_loss_reg', total_valid_loss_reg, iteration)
                            self.writer.add_scalar('valid_loss_smo', total_valid_loss_smo, iteration)
                            if self.args.mix_up_rate != 0:
                                self.writer.add_scalar('valid_loss_mixup', total_valid_loss_mixup, iteration)

                            for key, value in valid_metrics.items():
                                self.writer.add_scalar(key, value, iteration)
                            self.writer.add_scalar(f"Ratio of valid gates above threshold", average_above_threshold, iteration)
                            self.writer.add_scalar(f"Total number of valid gates above threshold", total_above_threshold, iteration)
                            self.writer.add_scalar(f"Total number of valid gates", total_valid_tokens.item(), iteration)
                            if not self.args.wandb_off:
                                wandb.log({"valid_loss": total_valid_loss}, step=iteration)
                                wandb.log({"valid_policy_loss": total_valid_policy_loss}, step=iteration)
                                wandb.log({"valid_loss_ce": total_valid_loss_ce}, step=iteration)
                                wandb.log({"valid_loss_reg": total_valid_loss_reg}, step=iteration)
                                wandb.log({"valid_loss_smo": total_valid_loss_smo}, step=iteration)
                                if self.args.mix_up_rate != 0:
                                    wandb.log({"valid_loss_mixup": total_valid_loss_mixup}, step=iteration)
                                wandb.log(valid_metrics, step=iteration)
                                wandb.log({"Ratio of valid gates above threshold": average_above_threshold.item()}, step=iteration)
                                wandb.log({"Total number of valid gates above threshold": total_above_threshold.item()}, step=iteration)
                                wandb.log({"Total number of valid gates": total_valid_tokens.item()}, step=iteration)
                                wandb.log({"valid_kl": total_valid_kl}, step=iteration)  
                            

                        if self.args.lambda_reg_scheduler:
                            self.args.lambda_reg = self.lambda_reg_scheduler.step() # reg_scheduler
                            print('Change lambda_reg :',self.args.lambda_reg)
                            wandb.log({"lambda_reg": self.args.lambda_reg}, step=iteration)

                        total_valid_score= valid_metrics['valid_roc_auc_class_1'] if self.args.data_name != "ag_news" else valid_metrics['valid_mean_roc_auc']
                        if (total_valid_score > top_score): 
                            top_score = total_valid_score
                            state = {
                                        'epoch ' : epoch,
                                        'iteration ' : iteration,
                                        'data' : self.args.data_name,
                                        'lr' : self.args.lr,
                                        'valid_loss' : total_valid_loss,
                                        'valid_loss_policy' : total_valid_policy_loss,
                                        'valid_loss_ce' : total_valid_loss_ce,
                                        'valid_loss_reg' : total_valid_loss_reg,
                                        'valid_loss_smo' : total_valid_loss_smo,
                                        'Ratio of valid gates above threshold' : average_above_threshold.item(),
                                        'Total number of valid gates above threshold' : total_above_threshold.item(),
                                        'Total number of valid gates' : total_valid_tokens.item(),
                                        'num_samples' : self.args.num_samples,
                                        'batch_size' : self.args.batch_size,
                                        'lambda_reg' : self.args.lambda_reg,
                                        'lambda_smo' : self.args.lambda_smo,
                                        'encoder_layer' : self.args.encoder_layer,
                                        'num_hops' : self.args.num_hops,

                                    }
                            logging.debug(f"Best_Iteration: {iteration}")
                            torch.save(self.model, os.path.join(self.writer.log_dir, f'{self.args.tokenizer_type}_{self.args.seed}.pt'))
                            
                            for key, value in valid_metrics.items():
                                state[key] = value
                            save_checkpoint(os.path.join(self.writer.log_dir), state)
                            
                        total_valid_score2= (valid_metrics['valid_roc_auc_class_1']-average_above_threshold) if self.args.data_name != "ag_news" else (valid_metrics['valid_mean_roc_auc']-average_above_threshold)
                        if (total_valid_score2 > top_score2): 
                            top_score2 = total_valid_score2
                            state = {
                                        'epoch ' : epoch,
                                        'iteration ' : iteration,
                                        'data' : self.args.data_name,
                                        'lr' : self.args.lr,
                                        'valid_loss' : total_valid_loss,
                                        'valid_loss_policy' : total_valid_policy_loss,
                                        'valid_loss_ce' : total_valid_loss_ce,
                                        'valid_loss_reg' : total_valid_loss_reg,
                                        'valid_loss_smo' : total_valid_loss_smo,
                                        'Ratio of valid gates above threshold' : average_above_threshold.item(),
                                        'Total number of valid gates above threshold' : total_above_threshold.item(),
                                        'Total number of valid gates' : total_valid_tokens.item(),
                                        'num_samples' : self.args.num_samples,
                                        'batch_size' : self.args.batch_size,
                                        'lambda_reg' : self.args.lambda_reg,
                                        'lambda_smo' : self.args.lambda_smo,
                                        'encoder_layer' : self.args.encoder_layer,
                                        'num_hops' : self.args.num_hops,

                                    }
                            logging.debug(f"Best_roc_ratio_Iteration: {iteration}")
                            torch.save(self.model, os.path.join(self.writer.log_dir, f'{self.args.tokenizer_type}_{self.args.seed}_roc_ratio.pt'))
                            
                            for key, value in valid_metrics.items():
                                state[key] = value
                            save_checkpoint(os.path.join(self.writer.log_dir), state)

                save_train_score_matrices_path = os.path.join(self.writer.log_dir, f"train_score_matrices.pkl")
                with open(save_train_score_matrices_path, "wb") as f:
                    pickle.dump(train_score_matrices, f)
                print(f"Filtered words saved to {save_train_score_matrices_path}")

                save_valid_score_matrices_path = os.path.join(self.writer.log_dir, f"valid_score_matrices.pkl")
                with open(save_valid_score_matrices_path, "wb") as f:
                    pickle.dump(all_score_matrices, f)
                print(f"Filtered words saved to {save_valid_score_matrices_path}")

    def valid_RL(self, valid_loader, wandb):
            total_valid_loss = 0
            total_valid_policy_loss = 0
            total_valid_loss_ce = 0
            total_valid_loss_reg = 0
            total_valid_loss_smo = 0
            total_valid_kl = 0
            all_valid_logits = []
            all_valid_labels = []

            all_filtered_tokens = {}
            all_filtered_words = {}
            all_sample_prob = {}
            all_grouped_prob = {}
            all_score_matrices = {}
            all_adj_matrices = {}
            total_above_threshold = 0
            total_valid_tokens = 0
            self.model.eval()
            with torch.no_grad():
                valid_pbar = tqdm(valid_loader, desc=f"Valid:", leave=False)
                for data in valid_pbar:
                    
                    data['meta_data']['idx'] = data['full_graph']['idx']

                    if self.args.data_name =="graph_sst2" and self.args.adj_type =='cross':
                        inputs = (data['full_graph'].to(self.args.device), data['external_graph'].to(self.args.device))
                        out = self.model(inputs, data['meta_data'], test=True)
                    elif self.args.data_name =="graph_sst2" and self.args.adj_type =='semantic':
                        out = self.model(data['external_graph'].to(self.args.device), data['meta_data'], test=True)
                    else:
                        out = self.model(data['full_graph'].to(self.args.device), data['meta_data'], test=True)
                    out = self.model(data['full_graph'].to(self.args.device), data['meta_data'], test=True)
                    outputs = out['outputs']
                    regularizer = out['regularizer']
                    loss_smo = out['loss_smo']
                    gate = out['token_soft_gate']
                    hard_gate = out['token_hard_gate']
                    total_token = out['total_token']
                    batch_filtered_token = out['filtered_tokens'] 
                    batch_filtered_words = out['filtered_words']
                    batch_sample_prob_dict = out['sample_prob_dict'] 
                    batch_grouped_prob_dict = out['grouped_prob_dict'] 
                    batch_score_matrices = out['score_matrices']
                    batch_adj_matrices = out['adj_matrix']

                    loss_ce = self.CE_loss(outputs.logits, data['full_graph']['y'].repeat(self.args.num_samples).to(self.args.device))
                    loss_reg = torch.nanmean(regularizer)

                    base_outputs = self.model(data['full_graph'].to(self.args.device), data['meta_data'], baseline = True)
                    base_loss_ce = self.CE_loss(base_outputs.logits, data['full_graph']['y'].to(self.args.device)).squeeze()
                    

                    eps = 1e-8
                    log_probs = torch.log(hard_gate * gate + (1 - hard_gate) * (1 - gate)+ eps).squeeze()

                    base_loss_ce = base_loss_ce.repeat(self.args.num_samples)
                    
                    temperature =1 
                    teacher_probs = F.softmax(base_outputs.logits.repeat_interleave(self.args.num_samples, dim=0) / temperature, dim=-1)
                    student_log_probs = F.log_softmax(outputs.logits / temperature, dim=-1)
                    kl=(F.kl_div(student_log_probs, teacher_probs, reduction='none') * (temperature**2)).sum(dim=1)
                    if self.args.policy_KL:
                        if self.args.lambda_kl < 1:
                            policy_KL = (-(kl).unsqueeze(-1) * log_probs).mean() 
                            policy_label = (-(loss_ce - base_loss_ce).unsqueeze(-1) * log_probs ).mean()
                            policy_loss = self.args.lambda_kl *policy_KL + (1-self.args.lambda_kl) * policy_label
                        else:
                            policy_loss = (-(kl).unsqueeze(-1) * log_probs).mean() 
                    elif self.args.policy_soft == -1:
                        if  self.args.baseline_type == 'basic':
                            advantage=(loss_ce - base_loss_ce).unsqueeze(-1) 
                        elif self.args.baseline_type == 'div_mean':
                            advantage = (loss_ce - base_loss_ce).unsqueeze(-1) 
                            advantage = advantage - advantage.mean()
                        elif self.args.baseline_type == 'mean':
                            advantage = loss_ce.unsqueeze(-1)
                            advantage = advantage - advantage.mean()
                            
                        policy_loss = (-advantage * log_probs ).mean()
                    else:
                        policy_loss = (-(F.softplus((loss_ce - base_loss_ce).unsqueeze(-1), beta=self.args.policy_soft) * log_probs )).mean()

                    num_above_threshold = (gate > self.args.gate_threshold).sum()
                    print(f"num_above_threshold: {num_above_threshold}/{total_token}")

                    total_valid_loss += (-policy_loss +  self.args.lambda_reg*loss_reg + self.args.lambda_smo*loss_smo).item()

                    total_valid_policy_loss += policy_loss.item()

                    total_valid_loss_ce += loss_ce.mean().item()
                    total_valid_loss_reg += loss_reg.item()
                    total_valid_loss_smo += loss_smo.item()
                    total_valid_kl += kl.mean().item()
                    all_valid_logits.append(outputs.logits.detach().cpu())
                    all_valid_labels.append(data['full_graph']['y'].repeat(self.args.num_samples).detach().cpu())

                    total_above_threshold += num_above_threshold
                    total_valid_tokens += total_token
                    all_filtered_tokens.update(batch_filtered_token)
                    all_filtered_words.update(batch_filtered_words)
                    all_sample_prob.update(batch_sample_prob_dict)
                    all_grouped_prob.update(batch_grouped_prob_dict)
                    all_score_matrices.update(batch_score_matrices)
                    all_adj_matrices.update(batch_adj_matrices)


                total_valid_loss = total_valid_loss / len(valid_loader)
                total_valid_policy_loss = total_valid_policy_loss / len(valid_loader)
                total_valid_loss_ce = total_valid_loss_ce / len(valid_loader)
                total_valid_loss_reg = total_valid_loss_reg / len(valid_loader)
                total_valid_loss_smo = total_valid_loss_smo / len(valid_loader)
                total_valid_kl = total_valid_kl / len(valid_loader)

                
                valid_logits = torch.cat(all_valid_logits, dim=0)
                valid_labels = torch.cat(all_valid_labels, dim=0)
                valid_preds = torch.argmax(valid_logits, dim=1).cpu()
                valid_probs = torch.softmax(valid_logits, dim=1).cpu().numpy()
                print("Valid Loss: {:.5f}".format(total_valid_loss))
                print("Valid Loss policy: {:.5f}".format(total_valid_policy_loss))
                print("Valid Loss CE: {:.5f}".format(total_valid_loss_ce))
                print("Valid Loss Reg: {:.5f}".format(total_valid_loss_reg))
                print("Valid Loss Smo: {:.5f}".format(total_valid_loss_smo))

                valid_metrics = get_all_metrics(
                    valid_labels.cpu().numpy(), 
                    valid_preds.numpy(), 
                    valid_probs, 
                    n_classes=2 if self.args.data_name != "ag_news" else 4, 
                    prefix="valid"
                )
                average_above_threshold = total_above_threshold / total_valid_tokens
                print(f"Ratio of valid gates above threshold: {average_above_threshold:.4f}")
                print(f"Total number of valid gates above threshold: {total_above_threshold}")
                print(f"Total number of valid gates: {total_valid_tokens}")

                logging.debug(f"Valid loss: {total_valid_loss} / Valid policy_loss: {total_valid_policy_loss} / Valid Loss CE: {total_valid_loss_ce} / Valid Loss Reg: {total_valid_loss_reg}")
                self.writer.add_scalar('valid_policy_loss', total_valid_policy_loss)
                self.writer.add_scalar('valid_loss_ce', total_valid_loss_ce)
                self.writer.add_scalar('valid_loss_reg', total_valid_loss_reg)
                self.writer.add_scalar('valid_loss_smo', total_valid_loss_smo)

                for key, value in valid_metrics.items():
                    self.writer.add_scalar(key, value)
                self.writer.add_scalar(f"Ratio of valid gates above threshold", average_above_threshold)
                self.writer.add_scalar(f"Total number of valid gates above threshold", total_above_threshold)
                self.writer.add_scalar(f"Total number of valid gates", total_valid_tokens.item())
                if not self.args.wandb_off:
                    wandb.log({"valid_loss": total_valid_loss})
                    wandb.log({"valid_policy_loss": total_valid_policy_loss})
                    wandb.log({"valid_loss_ce": total_valid_loss_ce})
                    wandb.log({"valid_loss_reg": total_valid_loss_reg})
                    wandb.log({"valid_loss_smo": total_valid_loss_smo})
                    wandb.log(valid_metrics)
                    wandb.log({"Ratio of valid gates above threshold": average_above_threshold.item()})
                    wandb.log({"Total number of valid gates above threshold": total_above_threshold.item()})
                    wandb.log({"Total number of valid gates": total_valid_tokens.item()})
                    wandb.log({"valid_kl": total_valid_kl})     

                save_filtered_word_path = os.path.join(self.writer.log_dir, f"valid_filtered_words.pkl")
                with open(save_filtered_word_path, "wb") as f:
                    pickle.dump(all_filtered_words, f)
                print(f"Filtered words saved to {save_filtered_word_path}")

                save_filtered_token_path = os.path.join(self.writer.log_dir, f"valid_filtered_tokens.pkl")
                with open(save_filtered_token_path, "wb") as f:
                    pickle.dump(all_filtered_tokens, f)
                print(f"Filtered words saved to {save_filtered_token_path}")

                thresholds = np.arange(0.1, 1.0, 0.1)
                all_sample_prob = get_rank(all_sample_prob, thresholds)

                save_sample_prob_path = os.path.join(self.writer.log_dir, f"valid_sample_prob.pkl")
                with open(save_sample_prob_path, "wb") as f:
                    pickle.dump(all_sample_prob, f)
                print(f"Filtered words saved to {save_sample_prob_path}")

                save_grouped_prob_path = os.path.join(self.writer.log_dir, f"valid_grouped_prob.pkl")
                with open(save_grouped_prob_path, "wb") as f:
                    pickle.dump(all_grouped_prob, f)
                print(f"Filtered words saved to {save_grouped_prob_path}")

                save_valid_preds_path = os.path.join(self.writer.log_dir, "valid_preds.npy")
                save_valid_probs_path = os.path.join(self.writer.log_dir, "valid_probs.npy")

                np.save(save_valid_preds_path, valid_preds.numpy())
                np.save(save_valid_probs_path, valid_probs)

                print(f"Valid predictions saved to {save_valid_preds_path}")
                print(f"Valid probabilities saved to {save_valid_probs_path}")

                state = {
                            'data' : self.args.data_name,
                            'lr' : self.args.lr,
                            'valid_loss' : total_valid_loss,
                            'valid_loss_policy' : total_valid_policy_loss,
                            'valid_loss_ce' : total_valid_loss_ce,
                            'valid_loss_reg' : total_valid_loss_reg,
                            'valid_loss_smo' : total_valid_loss_smo,
                            'Ratio of valid gates above threshold' : average_above_threshold.item(),
                            'Total number of valid gates above threshold' : total_above_threshold.item(),
                            'Total number of valid gates' : total_valid_tokens.item(),
                            'num_samples' : self.args.num_samples,
                            'batch_size' : self.args.batch_size,
                            'lambda_reg' : self.args.lambda_reg,
                            'lambda_smo' : self.args.lambda_smo,
                            'encoder_layer' : self.args.encoder_layer,
                            'num_hops' : self.args.num_hops,

                        }
                
                for key, value in valid_metrics.items():
                    state[key] = value
                save_checkpoint(os.path.join(self.writer.log_dir), state)

                save_valid_score_matrices_path = os.path.join(self.writer.log_dir, f"valid_score_matrices.pkl")
                with open(save_valid_score_matrices_path, "wb") as f:
                    pickle.dump(all_score_matrices, f)
                print(f"Filtered words saved to {save_valid_score_matrices_path}")

                save_valid_adj_matrices_path = os.path.join(self.writer.log_dir, f"valid_adj_matrices.pkl")
                with open(save_valid_adj_matrices_path, "wb") as f:
                    pickle.dump(all_adj_matrices, f)
                print(f"Filtered words saved to {save_valid_adj_matrices_path}")

    def test_RL(self, test_loader, wandb):
            # 
            total_test_loss = 0
            total_test_policy_loss = 0
            total_test_loss_ce = 0
            total_test_loss_reg = 0
            total_test_loss_smo = 0
            total_test_kl = 0
            all_test_logits = []
            all_test_labels = []
            all_baseline_test_logits = []



            all_filtered_tokens = {}
            all_filtered_words = {}
            all_sample_prob = {}
            all_grouped_prob = {}
            all_score_matrices = {}
            all_adj_matrices = {}
            total_above_threshold = 0
            total_test_tokens = 0
            self.model.eval()
            with torch.no_grad():
                test_pbar = tqdm(test_loader, desc=f"Test:", leave=False)
                for data in test_pbar:
                    
                    data['meta_data']['idx'] = data['full_graph']['idx']
                    
                    if self.args.data_name =="graph_sst2" and self.args.adj_type =='cross':
                        inputs = (data['full_graph'].to(self.args.device), data['external_graph'].to(self.args.device))
                        out = self.model(inputs, data['meta_data'], test=True)
                    elif self.args.data_name =="graph_sst2" and self.args.adj_type =='semantic':
                        out = self.model(data['external_graph'].to(self.args.device), data['meta_data'], test=True)
                    else:
                        out = self.model(data['full_graph'].to(self.args.device), data['meta_data'], test=True)

                    outputs = out['outputs']
                    regularizer = out['regularizer']
                    loss_smo = out['loss_smo']
                    gate = out['token_soft_gate']
                    hard_gate = out['token_hard_gate']
                    total_token = out['total_token']
                    batch_filtered_token = out['filtered_tokens'] 
                    batch_filtered_words = out['filtered_words'] 
                    batch_sample_prob_dict = out['sample_prob_dict'] 
                    batch_grouped_prob_dict = out['grouped_prob_dict'] 
                    batch_score_matrices = out['score_matrices'] 
                    batch_adj_matrices = out['adj_matrix']
                

                    loss_ce = self.CE_loss(outputs.logits, data['full_graph']['y'].repeat(self.args.num_samples).to(self.args.device))
                    loss_reg = torch.nanmean(regularizer)

                    base_outputs = self.model(data['full_graph'].to(self.args.device), data['meta_data'], baseline = True)
                    base_loss_ce = self.CE_loss(base_outputs.logits, data['full_graph']['y'].to(self.args.device)).squeeze()
                    

                    eps = 1e-8
                    log_probs = torch.log(hard_gate * gate + (1 - hard_gate) * (1 - gate)+ eps).squeeze()

                    base_loss_ce = base_loss_ce.repeat(self.args.num_samples)

                    temperature =1 
                    teacher_probs = F.softmax(base_outputs.logits.repeat_interleave(self.args.num_samples, dim=0) / temperature, dim=-1)
                    student_log_probs = F.log_softmax(outputs.logits / temperature, dim=-1)
                    kl=(F.kl_div(student_log_probs, teacher_probs, reduction='none') * (temperature**2)).sum(dim=1)
                    if self.args.policy_KL:
                        if self.args.lambda_kl < 1:
                            policy_KL = (-(kl).unsqueeze(-1) * log_probs).mean()
                            policy_label = (-(loss_ce - base_loss_ce).unsqueeze(-1) * log_probs ).mean()
                            policy_loss = self.args.lambda_kl *policy_KL + (1-self.args.lambda_kl) * policy_label
                        else:
                            policy_loss = (-(kl).unsqueeze(-1) * log_probs).mean()
                    elif self.args.policy_soft == -1:
                        if  self.args.baseline_type == 'basic':
                            advantage=(loss_ce - base_loss_ce).unsqueeze(-1) 
                        elif self.args.baseline_type == 'div_mean':
                            advantage = (loss_ce - base_loss_ce).unsqueeze(-1) 
                            advantage = advantage - advantage.mean()
                        elif self.args.baseline_type == 'mean':
                            advantage = loss_ce.unsqueeze(-1)
                            advantage = advantage - advantage.mean()
                            
                        policy_loss = (-advantage * log_probs ).mean()
                    else:
                        policy_loss = (-(F.softplus((loss_ce - base_loss_ce).unsqueeze(-1), beta=self.args.policy_soft) * log_probs )).mean()

                    num_above_threshold = (gate > self.args.gate_threshold).sum()
                    print(f"num_above_threshold: {num_above_threshold}/{total_token}")

                    total_test_loss += (-policy_loss +  self.args.lambda_reg*loss_reg + self.args.lambda_smo*loss_smo).item()

                    total_test_policy_loss += policy_loss.item()
                    total_test_loss_ce += loss_ce.mean().item()
                    total_test_loss_reg += loss_reg.item()
                    total_test_loss_smo += loss_smo.item()
                    total_test_kl += kl.mean().item()
                    all_test_logits.append(outputs.logits.detach().cpu())
                    all_test_labels.append(data['full_graph']['y'].repeat(self.args.num_samples).detach().cpu())
                    data['meta_data']['idx'] = data['full_graph']['idx']
                    
                    if self.args.data_name =="graph_sst2" and self.args.adj_type =='cross':
                        inputs = (data['full_graph'].to(self.args.device), data['external_graph'].to(self.args.device))
                        out = self.model(inputs, data['meta_data'], train_threshold=True)
                    elif self.args.data_name =="graph_sst2" and self.args.adj_type =='semantic':
                        out = self.model(data['external_graph'].to(self.args.device), data['meta_data'], train_threshold=True)
                    all_baseline_test_logits.append(base_outputs.logits.detach().cpu())

                    total_above_threshold += num_above_threshold
                    total_test_tokens += total_token
                    all_filtered_tokens.update(batch_filtered_token)
                    all_filtered_words.update(batch_filtered_words)
                    all_sample_prob.update(batch_sample_prob_dict)
                    all_grouped_prob.update(batch_grouped_prob_dict)
                    all_score_matrices.update(batch_score_matrices)
                    all_adj_matrices.update(batch_adj_matrices)

                total_test_loss = total_test_loss / len(test_loader)
                total_test_policy_loss = total_test_policy_loss / len(test_loader)
                total_test_loss_ce = total_test_loss_ce / len(test_loader)
                total_test_loss_reg = total_test_loss_reg / len(test_loader)
                total_test_loss_smo = total_test_loss_smo / len(test_loader)
                total_test_kl = total_test_kl / len(test_loader)

                test_logits = torch.cat(all_test_logits, dim=0)
                test_labels = torch.cat(all_test_labels, dim=0)
                test_preds = torch.argmax(test_logits, dim=1).cpu()
                test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()

                test_baseline_logits = torch.cat(all_baseline_test_logits, dim=0)
                test_baseline_preds = torch.argmax(test_baseline_logits, dim=1).cpu()
                test_baseline_probs = torch.softmax(test_baseline_logits, dim=1).cpu().numpy()

                print("Test Loss: {:.5f}".format(total_test_loss))
                print("Test Loss policy: {:.5f}".format(total_test_policy_loss))
                print("Test Loss CE: {:.5f}".format(total_test_loss_ce))
                print("Test Loss Reg: {:.5f}".format(total_test_loss_reg))
                print("Test Loss Smo: {:.5f}".format(total_test_loss_smo))
                # 
                test_metrics = get_all_metrics(
                    test_labels.cpu().numpy(), 
                    test_preds.numpy(), 
                    test_probs, 
                    n_classes=2 if self.args.data_name != "ag_news" else 4, 
                    prefix="test"
                )
                average_above_threshold = total_above_threshold / total_test_tokens
                print(f"Ratio of test gates above threshold: {average_above_threshold:.4f}")
                print(f"Total number of test gates above threshold: {total_above_threshold}")
                print(f"Total number of test gates: {total_test_tokens}")

                logging.debug(f"Test loss: {total_test_loss} / Test policy_loss: {total_test_policy_loss} / Test Loss CE: {total_test_loss_ce} / Test Loss Reg: {total_test_loss_reg}")
                self.writer.add_scalar('test_policy_loss', total_test_policy_loss)
                self.writer.add_scalar('test_loss_ce', total_test_loss_ce)
                self.writer.add_scalar('test_loss_reg', total_test_loss_reg)
                self.writer.add_scalar('test_loss_smo', total_test_loss_smo)

                for key, value in test_metrics.items():
                    self.writer.add_scalar(key, value)
                self.writer.add_scalar(f"Ratio of test gates above threshold", average_above_threshold)
                self.writer.add_scalar(f"Total number of test gates above threshold", total_above_threshold)
                self.writer.add_scalar(f"Total number of test gates", total_test_tokens.item())
                if not self.args.wandb_off:
                    wandb.log({"test_loss": total_test_loss})
                    wandb.log({"test_policy_loss": total_test_policy_loss})
                    wandb.log({"test_loss_ce": total_test_loss_ce})
                    wandb.log({"test_loss_reg": total_test_loss_reg})
                    wandb.log({"test_loss_smo": total_test_loss_smo})
                    wandb.log(test_metrics)
                    wandb.log({"Ratio of test gates above threshold": average_above_threshold.item()})
                    wandb.log({"Total number of test gates above threshold": total_above_threshold.item()})
                    wandb.log({"Total number of test gates": total_test_tokens.item()})
                    wandb.log({"test_kl": total_test_kl})     

                save_filtered_word_path = os.path.join(self.writer.log_dir, f"test_filtered_words.pkl")
                with open(save_filtered_word_path, "wb") as f:
                    pickle.dump(all_filtered_words, f)
                print(f"Filtered words saved to {save_filtered_word_path}")

                save_filtered_token_path = os.path.join(self.writer.log_dir, f"test_filtered_tokens.pkl")
                with open(save_filtered_token_path, "wb") as f:
                    pickle.dump(all_filtered_tokens, f)
                print(f"Filtered words saved to {save_filtered_token_path}")


                thresholds = np.arange(0.1, 1.0, 0.1)
                all_sample_prob = get_rank(all_sample_prob, thresholds)

                save_sample_prob_path = os.path.join(self.writer.log_dir, f"test_sample_prob.pkl")
                with open(save_sample_prob_path, "wb") as f:
                    pickle.dump(all_sample_prob, f)
                print(f"Filtered words saved to {save_sample_prob_path}")

                save_grouped_prob_path = os.path.join(self.writer.log_dir, f"test_grouped_prob.pkl")
                with open(save_grouped_prob_path, "wb") as f:
                    pickle.dump(all_grouped_prob, f)
                print(f"Filtered words saved to {save_grouped_prob_path}")

                save_test_preds_path = os.path.join(self.writer.log_dir, "test_preds.npy")
                save_test_probs_path = os.path.join(self.writer.log_dir, "test_probs.npy")

                np.save(save_test_preds_path, test_preds.numpy())
                np.save(save_test_probs_path, test_probs)

                print(f"Test predictions saved to {save_test_preds_path}")
                print(f"Test probabilities saved to {save_test_probs_path}")

                save_test_preds_path = os.path.join(self.writer.log_dir, "test_baseline_preds.npy")
                save_test_probs_path = os.path.join(self.writer.log_dir, "test_baseline_probs.npy")

                np.save(save_test_preds_path, test_baseline_preds.numpy())
                np.save(save_test_probs_path, test_baseline_probs)


                state = {
                            'data' : self.args.data_name,
                            'lr' : self.args.lr,
                            'test_loss' : total_test_loss,
                            'test_loss_policy' : total_test_policy_loss,
                            'test_loss_ce' : total_test_loss_ce,
                            'test_loss_reg' : total_test_loss_reg,
                            'test_loss_smo' : total_test_loss_smo,
                            'Ratio of test gates above threshold' : average_above_threshold.item(),
                            'Total number of test gates above threshold' : total_above_threshold.item(),
                            'Total number of test gates' : total_test_tokens.item(),
                            'num_samples' : self.args.num_samples,
                            'batch_size' : self.args.batch_size,
                            'lambda_reg' : self.args.lambda_reg,
                            'lambda_smo' : self.args.lambda_smo,
                            'encoder_layer' : self.args.encoder_layer,
                            'num_hops' : self.args.num_hops,

                        }

                for key, value in test_metrics.items():
                    state[key] = value
                save_checkpoint(os.path.join(self.writer.log_dir), state)

                save_test_score_matrices_path = os.path.join(self.writer.log_dir, f"test_score_matrices.pkl")
                with open(save_test_score_matrices_path, "wb") as f:
                    pickle.dump(all_score_matrices, f)
                print(f"Filtered words saved to {save_test_score_matrices_path}")

                save_test_adj_matrices_path = os.path.join(self.writer.log_dir, f"test_adj_matrices.pkl")
                with open(save_test_adj_matrices_path, "wb") as f:
                    pickle.dump(all_adj_matrices, f)
                print(f"Filtered words saved to {save_test_adj_matrices_path}")

    def test_RL_glue(self, test_loader, wandb):

            total_test_loss_reg = 0
            total_test_loss_smo = 0
            total_test_kl = 0
            all_test_logits = []
            all_test_labels = []
            all_baseline_test_logits = []

            all_filtered_tokens = {}
            all_filtered_words = {}
            all_sample_prob = {}
            all_grouped_prob = {}
            all_score_matrices = {}
            total_above_threshold = 0
            total_test_tokens = 0

            test_ids = []
            self.model.eval()
            with torch.no_grad():
                test_pbar = tqdm(test_loader, desc=f"Test:", leave=False)
                for data in test_pbar:
                    
                    data['meta_data']['idx'] = data['full_graph']['idx']

                    if self.args.data_name =="graph_sst2" and self.args.adj_type =='cross':
                        inputs = (data['full_graph'].to(self.args.device), data['external_graph'].to(self.args.device))
                        out = self.model(inputs, data['meta_data'], test=True)
                    elif self.args.data_name =="graph_sst2" and self.args.adj_type =='semantic':
                        out = self.model(data['external_graph'].to(self.args.device), data['meta_data'], test=True)
                    else:
                        out = self.model(data['full_graph'].to(self.args.device), data['meta_data'], test=True)
                    outputs = out['outputs']
                    regularizer = out['regularizer']
                    loss_smo = out['loss_smo']
                    gate = out['token_soft_gate']
                    hard_gate = out['token_hard_gate']
                    total_token = out['total_token']
                    batch_filtered_token = out['filtered_tokens']
                    batch_filtered_words = out['filtered_words']
                    batch_sample_prob_dict = out['sample_prob_dict'] 
                    batch_grouped_prob_dict = out['grouped_prob_dict'] 
                    batch_score_matrices = out['score_matrices'] 

                    loss_reg = torch.nanmean(regularizer)

                    base_outputs = self.model(data['full_graph'].to(self.args.device), data['meta_data'], baseline = True)
                    base_loss_ce = self.CE_loss(base_outputs.logits, data['full_graph']['y'].to(self.args.device)).squeeze()
                    
                    eps = 1e-8
                    log_probs = torch.log(hard_gate * gate + (1 - hard_gate) * (1 - gate)+ eps).squeeze()

                    base_loss_ce = base_loss_ce.repeat(self.args.num_samples)
                    
                    temperature =1 
                    teacher_probs = F.softmax(base_outputs.logits.repeat_interleave(self.args.num_samples, dim=0) / temperature, dim=-1)
                    student_log_probs = F.log_softmax(outputs.logits / temperature, dim=-1)
                    kl=(F.kl_div(student_log_probs, teacher_probs, reduction='none') * (temperature**2)).sum(dim=1)
                    

                    num_above_threshold = (gate > self.args.gate_threshold).sum()
                    print(f"num_above_threshold: {num_above_threshold}/{total_token}")

                    total_test_loss_reg += loss_reg.item()
                    total_test_loss_smo += loss_smo.item()
                    total_test_kl += kl.mean().item()
                    all_test_logits.append(outputs.logits.detach().cpu())
                    
                    all_baseline_test_logits.append(base_outputs.logits.detach().cpu())

                    total_above_threshold += num_above_threshold
                    total_test_tokens += total_token
                    all_filtered_tokens.update(batch_filtered_token)
                    all_filtered_words.update(batch_filtered_words)
                    all_sample_prob.update(batch_sample_prob_dict)
                    all_grouped_prob.update(batch_grouped_prob_dict)
                    all_score_matrices.update(batch_score_matrices)

                    test_ids.extend(data['full_graph']['idx'])

                total_test_loss_reg = total_test_loss_reg / len(test_loader)
                total_test_loss_smo = total_test_loss_smo / len(test_loader)
                total_test_kl = total_test_kl / len(test_loader)

                
                test_logits = torch.cat(all_test_logits, dim=0)
                baseline_test_logits = torch.cat(all_baseline_test_logits, dim=0)
                test_preds = torch.argmax(test_logits, dim=1).cpu()
                test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
                baseline_test_preds = torch.argmax(baseline_test_logits, dim=1).cpu()
                baseline_test_probs = torch.softmax(baseline_test_logits, dim=1).cpu().numpy()
                print("Test Loss Reg: {:.5f}".format(total_test_loss_reg))
                print("Test Loss Smo: {:.5f}".format(total_test_loss_smo))

                average_above_threshold = total_above_threshold / total_test_tokens
                print(f"Ratio of test gates above threshold: {average_above_threshold:.4f}")
                print(f"Total number of test gates above threshold: {total_above_threshold}")
                print(f"Total number of test gates: {total_test_tokens}")

                logging.debug(f"Test Loss Reg: {total_test_loss_reg} / Test Loss Smo: {total_test_loss_smo}")
                self.writer.add_scalar('test_loss_reg', total_test_loss_reg)
                self.writer.add_scalar('test_loss_smo', total_test_loss_smo)
                self.writer.add_scalar('test_kl', total_test_kl)
                self.writer.add_scalar(f"Ratio of test gates above threshold", average_above_threshold)
                self.writer.add_scalar(f"Total number of test gates above threshold", total_above_threshold)
                self.writer.add_scalar(f"Total number of test gates", total_test_tokens.item())
                if not self.args.wandb_off:
                    wandb.log({"test_loss_reg": total_test_loss_reg})
                    wandb.log({"test_loss_smo": total_test_loss_smo})
                    wandb.log({"Ratio of test gates above threshold": average_above_threshold.item()})
                    wandb.log({"Total number of test gates above threshold": total_above_threshold.item()})
                    wandb.log({"Total number of test gates": total_test_tokens.item()})
                    wandb.log({"test_kl": total_test_kl})     
                
                
                save_filtered_word_path = os.path.join(self.writer.log_dir, f"test_filtered_words.pkl")
                with open(save_filtered_word_path, "wb") as f:
                    pickle.dump(all_filtered_words, f)
                print(f"Filtered words saved to {save_filtered_word_path}")
                save_filtered_token_path = os.path.join(self.writer.log_dir, f"test_filtered_tokens.pkl")
                with open(save_filtered_token_path, "wb") as f:
                    pickle.dump(all_filtered_tokens, f)
                print(f"Filtered words saved to {save_filtered_token_path}")
                thresholds = np.arange(0.1, 1.0, 0.1)
                test_sample_prob = get_rank(test_sample_prob, thresholds)
                save_sample_prob_path = os.path.join(self.writer.log_dir, f"test_sample_prob.pkl")
                with open(save_sample_prob_path, "wb") as f:
                    pickle.dump(all_sample_prob, f)
                print(f"Filtered words saved to {save_sample_prob_path}")
                save_grouped_prob_path = os.path.join(self.writer.log_dir, f"test_grouped_prob.pkl")
                with open(save_grouped_prob_path, "wb") as f:
                    pickle.dump(all_grouped_prob, f)
                print(f"Filtered words saved to {save_grouped_prob_path}")


                save_test_preds_path = os.path.join(self.writer.log_dir, "test_preds.npy")
                save_test_probs_path = os.path.join(self.writer.log_dir, "test_probs.npy")

                np.save(save_test_preds_path, test_preds.numpy())
                np.save(save_test_probs_path, test_probs)

                print(f"Valid predictions saved to {save_test_preds_path}")
                print(f"Valid probabilities saved to {save_test_probs_path}")

                state = {
                            'data' : self.args.data_name,
                            'lr' : self.args.lr,
                            # 'test_loss' : total_test_loss,
                            # 'test_loss_policy' : total_test_policy_loss,
                            # 'test_loss_ce' : total_test_loss_ce,
                            'test_loss_reg' : total_test_loss_reg,
                            'test_loss_smo' : total_test_loss_smo,
                            'Ratio of test gates above threshold' : average_above_threshold.item(),
                            'Total number of test gates above threshold' : total_above_threshold.item(),
                            'Total number of test gates' : total_test_tokens.item(),
                            'num_samples' : self.args.num_samples,
                            'batch_size' : self.args.batch_size,
                            'lambda_reg' : self.args.lambda_reg,
                            'lambda_smo' : self.args.lambda_smo,
                            'encoder_layer' : self.args.encoder_layer,
                            'num_hops' : self.args.num_hops,

                        }

                save_test_score_matrices_path = os.path.join(self.writer.log_dir, f"test_score_matrices.pkl")
                with open(save_test_score_matrices_path, "wb") as f:
                    pickle.dump(all_score_matrices, f)
                print(f"Filtered words saved to {save_test_score_matrices_path}")                

                glue_submission_path = os.path.join(self.writer.log_dir, f"{self.args.data_name}_mask_submission.tsv")
                task = "MRPC"
                header = "index\tprediction\n"
                file_name = f"{task}.tsv"

                with open(file_name, "w", encoding="utf-8") as f:
                    f.write(header)
                    for idx, pred in zip(test_ids, test_preds):

                        f.write(f"{idx}\t{int(pred)}\n")

                print(f"GLUE-style submission masked model pred saved to {glue_submission_path}")
                
                glue_baseline_submission_path = os.path.join(self.writer.log_dir, f"{self.args.data_name}_baseline_submission.tsv")
                with open(glue_baseline_submission_path, "w") as f:
                    for idx, pred in zip(test_ids, baseline_test_preds):
                        f.write(f"{idx}\t{pred.item()}\n")

                print(f"GLUE-style submission baseline model pred saved to {glue_baseline_submission_path}")