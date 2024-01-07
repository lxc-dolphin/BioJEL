# coding=utf-8

""" Finetuning BioBERT models on MedMentions.
    Adapted from HuggingFace `examples/run_glue.py`"""

import argparse
import glob
import logging
import os
import random
import math
import pickle

import matplotlib.pyplot as plt


import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
import json

import pdb

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    # BertConfig,
    # BertForSequenceClassification,
    # BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    get_linear_schedule_with_warmup,
)

from utils_e2e_span import get_BC_examples_new_test, get_examples, convert_examples_to_features, get_BC_examples_new,get_BC_examples_new_dev 
from utils_e2e_span import get_GE_examples

from utils_e2e_span import get_mentions_tokens, convert_tags_to_ids, get_candi_tokens
from utils_e2e_span import InputFeatures1


from modeling_bert import BertModel
from tokenization_bert import BertTokenizer
from configuration_bert import BertConfig
from modeling_e2e_span import DualEncoderBert, PreDualEncoder


from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in [BertConfig]), ()
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train_1(args, model, tokenizer):
    
    # accuracy, num_case, accuracy_rate = train_accuracy(args, tokenizer , model, prefix="")
    
    # print(accuracy,num_case,accuracy_rate)
    
    
    tb_writer = SummaryWriter()
    dataset, (all_bc_token, all_bc_mask), all_seq_tags,all_result,all_protname \
    = load_and_creat_BC_datasets(args, tokenizer, model = model)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    

    dataset_dev, (all_bc_token, all_bc_mask), all_seq_tags,all_result,all_protname_dev \
    = load_and_creat_BC_datasets_dev(args, tokenizer, model = model)

    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices)
    
    train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, 
                                        sampler=train_sampler)
    valid_dataloader = DataLoader(dataset_dev, batch_size=args.train_batch_size,
                                        shuffle=False)
        
    
    
    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    
    
    # train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False)
    
    
    
    Prot_names  = all_protname
    
    
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    
    args.save_steps =len(train_dataloader)-5
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    
    # Check if saved optimizer or scheduler states exist
    if args.resume_path is not None and os.path.isfile(os.path.join(args.resume_path, "optimizer.pt")) \
            and os.path.isfile(os.path.join(args.resume_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.resume_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.resume_path, "scheduler.pt")))
        logger.info("INFO: Optimizer and scheduler state loaded successfully.")
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.resume_path is not None:
        # set global_step to global_step of last saved checkpoint from model path
        # global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        global_step = int(args.resume_path.split("/")[-2].split("-")[-1])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)



    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility


    loss_eval_tt = []
    loss_train_tt = []
    
    for epoch_num in train_iterator:

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        
        for step, batch in enumerate(epoch_iterator):
            
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            
            model.train()
            
            batch = tuple(t.to(args.device) for t in batch)

            inputs_bc = {
                    "mention_token_ids" : batch[0],
                    "mention_token_masks" : batch[1],
                    "target": batch[3]
                    }
            test_LOSS, prelogit = model.forward_1(**inputs_bc)

            
            if args.gradient_accumulation_steps > 1: # default : 1 
                test_LOSS = test_LOSS / args.gradient_accumulation_steps
            
            test_LOSS.backward()    
            
            tr_loss += test_LOSS.item()
            
            if (step + 1) % args.gradient_accumulation_steps == 0: # gradient_accumulation_steps = 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                
                # local_rank: default -1
                # logging step: 10
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
                # saving step: 100
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
                
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
            
        loss_eval = evaluate_intrain(args, model, valid_dataloader, prefix="",)
            
            
        loss_eval_tt.append(loss_eval)
        
        loss_train_tt.append(tr_loss/global_step)
            
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    
    if args.local_rank in [-1, 0]:
        tb_writer.close()
    
    
    
    
    
    
    return global_step, tr_loss / global_step, loss_train_tt, loss_eval_tt


def evaluate_1(args, model, tokenizer, prefix=""):
    
    eval_output_dir = args.output_dir
    
    dataset, (all_bc_token, all_bc_mask), all_seq_tags,all_result,all_protname \
    = load_and_creat_BC_datasets_Test(args, tokenizer, model = model)

    # validation_split = 0.2
    
    # dataset_size = len(dataset)
    # indices = list(range(dataset_size))
    # split = int(np.floor(validation_split * dataset_size))
    
    # np.random.seed(42)
    # np.random.shuffle(indices)
    # train_indx, val_indx = indices[split:], indices[:split]
    
    # train_sampler = SubsetRandomSampler(train_indx)
    # valid_sampler = SubsetRandomSampler(val_indx)

    
    entity_path = './data/BC4GE_data_PosiNegaCandi_test25n.json'
        
    a_file = open(entity_path, "r")
    Gene_data_PosiNega = json.loads(a_file.read())
    a_file.close()

    # for case_id, val in Gene_data_PosiNega.items():
    #     num_po_ne = len(val[2])+len(val[3]) # total number of positive and negative cases
            
    
    
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
         os.makedirs(eval_output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    
    # eval_dataset = eval_dataset[10000:]
    eval_dataloader = DataLoader(dataset, batch_size=args.eval_batch_size,shuffle=False)    
    result = {}
    overall_loss = 0
    num_case = 0
    ii = 0
    
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        
        ii += 1
        if ii < 5581:
            with torch.no_grad():
                num_case += 1
                inputs_bc = {
                            "mention_token_ids" : batch[0],
                            "mention_token_masks" : batch[1],
                            "target": batch[3]
                            }
                
                hinge_LOSS, eval_logit = model.forward_1(**inputs_bc)
                
                preds = eval_logit.detach().cpu().numpy()
                preds = preds.tolist()[0][0]
                lable = batch[3].detach().cpu().numpy()
                lable = lable.tolist()[0]
                hinge_LOSS = hinge_LOSS.detach().cpu().numpy()
                
                if all_protname[num_case] not in result:
                    result[all_protname[num_case]] = {}
                    result[all_protname[num_case]][num_case] = {'true:': lable, 'predict:': preds} 
                    
                else:
                    result[all_protname[num_case]][num_case] = {'true:': lable, 'predict:': preds}
            
                overall_loss += hinge_LOSS
                
                if num_case % 500 == 0:
                    print(overall_loss/num_case)
        else:
            break
        
    
    
    
    return result, overall_loss/num_case


def test_ge11(args, model, tokenizer, prefix=""):
    
    eval_output_dir = args.output_dir
    
    dataset, (all_bc_token, all_bc_mask), all_seq_tags,all_protname,sen_enti_candi_number \
    = load_and_creat_GE(args, tokenizer, model = model)

    # validation_split = 0.2
    
    # dataset_size = len(dataset)
    # indices = list(range(dataset_size))
    # split = int(np.floor(validation_split * dataset_size))
    
    # np.random.seed(42)
    # np.random.shuffle(indices)
    # train_indx, val_indx = indices[split:], indices[:split]
    
    # train_sampler = SubsetRandomSampler(train_indx)
    # valid_sampler = SubsetRandomSampler(val_indx)

    
    # entity_path = './data/BC4GE_data_PosiNegaCandi_test25n.json'
        
    # a_file = open(entity_path, "r")
    # Gene_data_PosiNega = json.loads(a_file.read())
    # a_file.close()

    # for case_id, val in Gene_data_PosiNega.items():
    #     num_po_ne = len(val[2])+len(val[3]) # total number of positive and negative cases
            
    
    
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
         os.makedirs(eval_output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    
    # eval_dataset = eval_dataset[10000:]
    eval_dataloader = DataLoader(dataset, batch_size=args.eval_batch_size,shuffle=False)    
    all_results = []
    num_case = 0
    ii = 0
    
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    
    # for enti_candi_number in sen_enti_candi_number: # for each sentence
    #     for candi_number in enti_candi_number: # for each entity in each sentence
    #         candi_count = 0
    #         print(candi_count)
    
    
    case_count = 0
    protein_name = all_protname[0]
    pre_result_by_protname = []
    
    
    
    pre_protname_id = []
    pre_protname_value = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)


        with torch.no_grad():
            num_case += 1
            inputs_bc = {
                        "mention_token_ids" : batch[0],
                        "mention_token_masks" : batch[1],
                        "target": batch[3]
                        }
            
            hinge_LOSS, eval_logit = model.forward_1(**inputs_bc)
            
            preds = eval_logit.detach().cpu().numpy()
            preds = preds.tolist()[0][0]
            hinge_LOSS = hinge_LOSS.detach().cpu().numpy()
            
            all_results.append(preds)
            

        if protein_name == all_protname[case_count]:
            pre_result_by_protname.append(preds)
        else:
            max_value = max(pre_result_by_protname)
            idx = pre_result_by_protname.index(max_value)
            pre_protname_id.append(idx)
            pre_protname_value.append(max_value)

            protein_name = all_protname[case_count]
            pre_result_by_protname = []
            pre_result_by_protname.append(preds)
            
        case_count += 1
        
        # if case_count == 50:
        #     break
    f = open("GE11_train_candidate_25_preds.pkl","wb")
    # jj = json.dumps(sentences_all)
    pickle.dump(all_results,f)
    # f.write(jj)
    f.close()
    
    f = open("GE11_train_candidate_25_preds_candi_number.pkl","wb")
    # jj = json.dumps(sentences_all)
    pickle.dump(sen_enti_candi_number,f)
    # f.write(jj)
    f.close()
    
    # for enti_candi_number in sen_enti_candi_number: # for each sentence
    #     for candi_number in enti_candi_number: # for each entity in each sentence
    #         candi_count = 0
    #         print(candi_count)
    
    
    
    
    return 0



def evaluate_intrain(args, model, eval_dataloader, prefix=""):
    eval_output_dir = args.output_dir
    
    # eval_dataset, (all_bc_token, all_bc_mask), all_seq_tags,all_result,all_protname \
    # = load_and_creat_BC_datasets(args, tokenizer, model = model)

    
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
         os.makedirs(eval_output_dir)
    # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    
    # # eval_dataset = eval_dataset[10000:]
        
    # eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False)
    
    overall_loss = 0
    num_case = 0
    ii = 0
    
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.train_batch_size)
    
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        ii += 1
        if ii < len(eval_dataloader):
            with torch.no_grad():
                num_case += 1
                inputs_bc = {
                            "mention_token_ids" : batch[0],
                            "mention_token_masks" : batch[1],
                            "target": batch[3]
                            }
                
                hinge_LOSS, eval_logit = model.forward_1(**inputs_bc)
                
                preds = eval_logit.detach().cpu().numpy()
                preds = preds.tolist()[0][0]
                lable = batch[3].detach().cpu().numpy()
                lable = lable.tolist()[0]
                hinge_LOSS = hinge_LOSS.detach().cpu().numpy()
                
                overall_loss += hinge_LOSS
                
                if num_case % 10 == 0:
                    print(overall_loss/num_case)
        else:
            break
    return overall_loss/num_case




def load_and_creat_BC_datasets(args, tokenizer, model=None):

    mode = 'train'
    logger.info("Creating BC features from dataset file at %s", args.data_dir)
    features = get_BC_examples_new(args.data_dir,args.max_seq_length,tokenizer, args)

    all_bc_token = torch.tensor([f.mention_token_ids for f in features], dtype=torch.long)
    all_bc_mask = torch.tensor([f.mention_token_masks for f in features], dtype=torch.long)
    all_sequence_tages = torch.tensor([f.sequence_tags for f in features], dtype=torch.long)
    all_result = torch.tensor([f.result for f in features], dtype=torch.float)
    all_protname =[f.mention_textname for f in features]
    dataset = TensorDataset(all_bc_token,
                            all_bc_mask,
                            all_sequence_tages,
                            all_result
                            )

    return dataset, (all_bc_token, all_bc_mask), all_sequence_tages, all_result, all_protname





def load_and_creat_BC_datasets_dev(args, tokenizer, model=None):

    mode = 'train'
    logger.info("Creating BC features from dataset file at %s", args.data_dir)
    features = get_BC_examples_new_dev(args.data_dir,args.max_seq_length,tokenizer, args)

    all_bc_token = torch.tensor([f.mention_token_ids for f in features], dtype=torch.long)
    all_bc_mask = torch.tensor([f.mention_token_masks for f in features], dtype=torch.long)
    all_sequence_tages = torch.tensor([f.sequence_tags for f in features], dtype=torch.long)
    all_result = torch.tensor([f.result for f in features], dtype=torch.float)
    all_protname =[f.mention_textname for f in features]
    dataset = TensorDataset(all_bc_token,
                            all_bc_mask,
                            all_sequence_tages,
                            all_result
                            )

    return dataset, (all_bc_token, all_bc_mask), all_sequence_tages, all_result, all_protname


def load_and_creat_BC_datasets_Test(args, tokenizer, model=None):

    mode = 'test'
    logger.info("Creating BC features from dataset file at %s", args.data_dir)
    features = get_BC_examples_new_test(args.data_dir,args.max_seq_length,tokenizer, args)

    all_bc_token = torch.tensor([f.mention_token_ids for f in features], dtype=torch.long)
    all_bc_mask = torch.tensor([f.mention_token_masks for f in features], dtype=torch.long)
    all_sequence_tages = torch.tensor([f.sequence_tags for f in features], dtype=torch.long)
    all_result = torch.tensor([f.result for f in features], dtype=torch.float)
    all_protname =[f.mention_textname for f in features]
    dataset = TensorDataset(all_bc_token,
                            all_bc_mask,
                            all_sequence_tages,
                            all_result
                            )

    return dataset, (all_bc_token, all_bc_mask), all_sequence_tages, all_result, all_protname


def load_and_creat_GE(args, tokenizer, model=None):
    
    mode = 'test'
    
    
    logger.info("Creating BC features from dataset file at %s", args.data_dir)
    features, senten_enti_candi_number = get_GE_examples(args.data_dir,args.max_seq_length,tokenizer, args)

    all_bc_token = torch.tensor([f.mention_token_ids for f in features], dtype=torch.long)
    all_bc_mask = torch.tensor([f.mention_token_masks for f in features], dtype=torch.long)
    all_sequence_tages = torch.tensor([f.sequence_tags for f in features], dtype=torch.long)
    all_result = torch.tensor([f.result for f in features], dtype=torch.float)
    all_protname =[f.mention_textname for f in features]
    dataset = TensorDataset(all_bc_token,
                            all_bc_mask,
                            all_sequence_tages,
                            all_result
                            )
    
    return dataset, (all_bc_token, all_bc_mask), all_sequence_tages, all_protname, senten_enti_candi_number


def train_accuracy(args, tokenizer , model, prefix=""):
    
    
    logger.info("Creating BC features one by one for accuracy from dataset file at %s", args.data_dir)
    features = get_BC_examples_new(args.data_dir,args.max_seq_length,tokenizer, args)

    max_seq_length = args.max_seq_length
    
    entity_path = './data/BC4GE_data_PosiNegaCandi_train25.json'
    a_file = open(entity_path, "r")
    Gene_data_PosiNega = json.loads(a_file.read())
    a_file.close()
    
    
    accuracy = 0
    num_case = 0
    
    for case_id, val in Gene_data_PosiNega.items():
        
        if num_case == 5:
            break
        
        features = []
        
        Genedata = val[0]
        Gene_trueGo = val[1]
        Gene_posi = val[2]
        Gene_nega = val[3]
        
        
        
        tokenized_text_, mention_start_markers, mention_end_markers, sequence_tags \
        = get_mentions_tokens(Genedata,tokenizer)
        
        doc_tokens = tokenizer.convert_tokens_to_ids(tokenized_text_)
        seq_tag_ids = convert_tags_to_ids(sequence_tags)
        # bc with positive candi 
        for go_id in Gene_posi:
            candi_token, candi_seq = get_candi_tokens(Gene_posi[go_id],tokenizer)
            candi_seq = convert_tags_to_ids(candi_seq)
            candi_token = tokenizer.convert_tokens_to_ids(candi_token)
            
            token_bc_candi = doc_tokens + candi_token
            sequence_tags_bc_candi = seq_tag_ids + candi_seq
            result = 1.0
            # store into Features
            if len(token_bc_candi) > max_seq_length:
                print(len(token_bc_candi))
                
                token_bc_candi = token_bc_candi[:max_seq_length]
                token_bc_mask = [1] * max_seq_length
                sequence_tags_bc_candi = sequence_tags_bc_candi[:max_seq_length]
            else:
                mention_len = len(token_bc_candi)
                pad_len = max_seq_length - mention_len
                token_bc_candi += [tokenizer.pad_token_id] * pad_len
                token_bc_mask = [1] * mention_len + [0] * pad_len
                sequence_tags_bc_candi += [-100]*pad_len 
                
                
            features.append(
                InputFeatures1(
                    mention_token_ids = token_bc_candi, 
                    mention_token_masks = token_bc_mask,
                    sequence_tags = sequence_tags_bc_candi, 
                    result = result,
                    mention_textname = Genedata['gene_name']
                )
                )
            
            
        # bc with negative candi
        for go_id in Gene_nega:
            candi_token, candi_seq = get_candi_tokens(Gene_nega[go_id],tokenizer)
            candi_seq = convert_tags_to_ids(candi_seq)
            candi_token = tokenizer.convert_tokens_to_ids(candi_token)
            
            token_bc_candi = doc_tokens + candi_token
            sequence_tags_bc_candi = seq_tag_ids + candi_seq
            result = 0.0
            # store into Features
            if len(token_bc_candi) > max_seq_length:
                print(len(token_bc_candi))
                sequence_tags_bc_candi = sequence_tags_bc_candi[:max_seq_length]
                token_bc_candi = token_bc_candi[:max_seq_length]
                token_bc_mask = [1] * max_seq_length
                
            else:
                mention_len = len(token_bc_candi)
                pad_len = max_seq_length - mention_len
                token_bc_candi += [tokenizer.pad_token_id] * pad_len
                token_bc_mask = [1] * mention_len + [0] * pad_len
                sequence_tags_bc_candi += [-100]*pad_len
                
                
            features.append(
                InputFeatures1(
                    mention_token_ids = token_bc_candi, 
                    mention_token_masks = token_bc_mask,
                    sequence_tags = sequence_tags_bc_candi, 
                    result = result,
                    mention_textname = Genedata['gene_name']
                )
                )


        bc_token_each = torch.tensor([f.mention_token_ids for f in features], dtype=torch.long)
        bc_mask_each = torch.tensor([f.mention_token_masks for f in features], dtype=torch.long)
        sequence_tages_each = torch.tensor([f.sequence_tags for f in features], dtype=torch.long)
        result_each = torch.tensor([f.result for f in features], dtype=torch.float)
        protname_each =[f.mention_textname for f in features]
        dataset_each = TensorDataset(bc_token_each,
                                bc_mask_each,
                                sequence_tages_each,
                                result_each
                                )
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        
        dataloader_each = DataLoader(dataset_each, batch_size=args.train_batch_size,
                                        shuffle=False)
        
        score_each = []
        lable_each = []
        for batch in enumerate(dataloader_each):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch[1])
            
            with torch.no_grad():
                
                inputs_bc = {
                            "mention_token_ids" : batch[0],
                            "mention_token_masks" : batch[1],
                            "target": batch[3]
                            }
                
                hinge_LOSS, eval_logit = model.forward_1(**inputs_bc)
                
                preds = eval_logit.detach().numpy()
                preds = preds.tolist()[0][0]
                lable = batch[3].detach().numpy()
                lable = lable.tolist()[0]
                hinge_LOSS = hinge_LOSS.detach().numpy()

                score_each.append(preds)
                lable_each.append(lable)

        max_sc = max(score_each)
        indx_max = score_each.index(max_sc)
        max_lable = lable_each[indx_max]

        if max_lable == 1:
            accuracy += 1
        num_case += 1
        print(accuracy, num_case)
    

    return accuracy, num_case, accuracy/num_case


def load_and_cache_examples(args, tokenizer, model=None):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    mode = 'train' if args.do_train else 'test'
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}".format(
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop()),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        all_entities = np.load(os.path.join(args.data_dir, 'all_entities.npy'))
        all_entity_token_ids = np.load(os.path.join(args.data_dir, 'all_entity_token_ids.npy'))
        all_entity_token_masks = np.load(os.path.join(args.data_dir, 'all_entity_token_masks.npy'))
        all_document_ids = np.load(os.path.join(args.data_dir, 'all_document_ids.npy'))
        all_label_candidate_ids = np.load(os.path.join(args.data_dir, 'all_label_candidate_ids.npy'))
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples, docs, entities = get_examples(args.data_dir, mode)
        features, (all_entities, all_entity_token_ids, all_entity_token_masks), (all_document_ids, all_label_candidate_ids) = convert_examples_to_features(
            examples,
            docs,
            entities,
            args.max_seq_length,
            tokenizer,
            args,
            model,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
            np.save(os.path.join(args.data_dir, 'all_entities.npy'),
                        np.array(all_entities))
            np.save(os.path.join(args.data_dir, 'all_entity_token_ids.npy'),
                    np.array(all_entity_token_ids))
            np.save(os.path.join(args.data_dir, 'all_entity_token_masks.npy'),
                    np.array(all_entity_token_masks))
            np.save(os.path.join(args.data_dir, 'all_document_ids.npy'),
                    np.array(all_document_ids))
            np.save(os.path.join(args.data_dir, 'all_label_candidate_ids.npy'),
                    np.array(all_label_candidate_ids))

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_mention_token_ids = torch.tensor([f.mention_token_ids for f in features], dtype=torch.long)
    all_mention_token_masks = torch.tensor([f.mention_token_masks for f in features], dtype=torch.long)
    all_candidate_token_ids_1 = torch.tensor([f.candidate_token_ids_1 if f.candidate_token_ids_1 is not None else [0] for f in features], dtype=torch.long)
    all_candidate_token_masks_1 = torch.tensor([f.candidate_token_masks_1 if f.candidate_token_masks_1 is not None else [0] for f in features], dtype=torch.long)
    all_candidate_token_ids_2 = torch.tensor([f.candidate_token_ids_2 if f.candidate_token_ids_2 is not None else [0] for f in features], dtype=torch.long)
    all_candidate_token_masks_2 = torch.tensor([f.candidate_token_masks_2 if f.candidate_token_masks_2 is not None else [0] for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_mention_start_indices = torch.tensor([f.mention_start_indices for f in features], dtype=torch.long)
    all_mention_end_indices = torch.tensor([f.mention_end_indices for f in features], dtype=torch.long)
    all_num_mentions = torch.tensor([f.num_mentions for f in features], dtype=torch.long)
    all_seq_tag_ids = torch.tensor([f.seq_tag_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_mention_token_ids,
                            all_mention_token_masks,
                            all_candidate_token_ids_1,
                            all_candidate_token_masks_1,
                            all_candidate_token_ids_2,
                            all_candidate_token_masks_2,
                            all_labels,
                            all_mention_start_indices,
                            all_mention_end_indices,
                            all_num_mentions,
                            all_seq_tag_ids,
                            )
    return dataset, (all_entities, all_entity_token_ids, all_entity_token_masks), (all_document_ids, all_label_candidate_ids)


def create_datas(args, tokenizer, model=None):
    dataset = 1
    return dataset    




def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="",
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--resume_path",
        default=None,
        type=str,
        required=False,
        help="Path to the checkpoint from where the training should resume"
    )
    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_mention_length",
        default=20,
        type=int,
        help="Maximum length of a mention span"
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument("--do_test",action="store_true", help="Whether to run testing.")
    
    
    
    
    parser.add_argument(
        "--do_lower_case", action="store_true", default=False, help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--use_random_candidates", action="store_true", help="Use random negative candidates during training"
    )
    parser.add_argument(
        "--use_tfidf_candidates", action="store_true", help="Use random negative candidates during training"
    )
    parser.add_argument(
        "--use_hard_negatives",  action="store_true", help="Use hard negative candidates during training"
    )
    parser.add_argument(
        "--use_hard_and_random_negatives", action="store_true", help="Use hard negative candidates during training"
    )
    parser.add_argument(
        "--include_positive", action="store_true", help="Includes the positive candidate during inference"
    )
    parser.add_argument(
        "--use_all_candidates", action="store_true", help="Use all entities as candidates"
    )
    parser.add_argument(
        "--num_candidates", type=int, default=5, help="Number of candidates to consider per mention"
    )
    parser.add_argument(
        "--num_max_mentions", type=int, default=8, help="Maximum number of mentions in a document"
    )
    parser.add_argument(
        "--ner", type=bool, default=False, help="Model will perform only BIO tagging"
    )
    parser.add_argument(
        "--alternate_batch", type=bool, default=False, help="Model will perform either BIO tagging or entity linking per batch during training"
    )
    parser.add_argument(
        "--ner_and_ned", type=bool, default=True, help="Model will perform both BIO tagging and entity linking per batch during training"
    )
    parser.add_argument(
        "--gamma", type=float, default=0, help="Threshold for mention candidate prunning"
    )
    parser.add_argument(
        "--lambda_1", type=float, default=1, help="Weight of the random candidate loss"
    )
    parser.add_argument(
        "--lambda_2", type=float, default=0, help="Weight of the hard negative candidate loss"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    
    # args.data_dir = './data/Ge11/dev/GE11_dev_candidate_25.pkl'
    
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        if args.no_cuda:
            args.n_gpu = 0
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer 
    ########################################### important
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    pretrained_bert = PreDualEncoder.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # Add new special tokens '[Ms]' and '[Me]' to tag mention
    new_tokens = ['[Ms]', '[Me]']
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    pretrained_bert.resize_token_embeddings(len(tokenizer))

    model = DualEncoderBert(config, pretrained_bert)


    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

###################### start train_1/ eval_1
    if args.do_train:
        
        global_step, tr_loss, loss_train_tt, loss_eval_tt  = train_1(args, model, tokenizer)
    
        print(global_step, tr_loss)

        a_file = open("BC4GE_data_train_eva_scibert.json", "w")
        jj = json.dumps({'train':loss_train_tt, 'eval':loss_eval_tt})
        a_file.write(jj)
        a_file.close()
        
        
    
    results = {}
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        num_chek = 0
        for checkpoint in checkpoints:
            num_chek += 1
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model-1000000.bin')))
            model.to(args.device)
            result, mean_loss = evaluate_1(args, model, tokenizer, prefix=prefix)
            # result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            
            # a_file = open("BC4GE_data_evaluation.json", "w")
            # json = json.dumps(result)
            # a_file.write(json)
            # a_file.close()
            
            results[num_chek] = [result, mean_loss]
            print(mean_loss)
    
        a_file = open("BC4GE_data_testsiteN.json", "w")
        jj = json.dumps(result)
        a_file.write(jj)
        a_file.close()
        
    if args.do_test:
        print(123)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        num_chek = 0
        for checkpoint in checkpoints:
            num_chek += 1
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model-1000000.bin')))
            model.to(args.device)

            result = test_ge11(args, model, tokenizer, prefix=prefix)
            
            
            
        
        
    
    
        
        # np_bc = last_vect_bc.detach().numpy()

        # bc_norm = np_bc/np.sqrt(np.sum(np_bc**2))

        # inputs_match = {
        #         "mention_token_ids" : batch[2],
        #         "mention_token_masks" : batch[3],
        #         }
        # last_vect_match = model.forward_1(**inputs_match)

        # np_match = last_vect_match.detach().numpy()

        # match_norm = np_match/np.sqrt(np.sum(np_match**2))
        
        # SCORES[step-1] = np.inner(bc_norm,match_norm)
        
        # print(last_vect_match.size(), '=========', step)
    # results = {}




    # x = np.arange(50)
    # y = SCORES
    # my_xticks = Prot_names
    # plt.xticks(x, my_xticks)
    # plt.plot(x, y)
    # plt.show()
    # plt.show(block=False)
    # #input('press <ENTER> to continue')

    # f = open("BC4GE_Score_data_nameVsdefinition.pkl","wb")
    # # write the python object (dict) to pickle file
    # pickle.dump(SCORES,f)
    # # close file
    # f.close()


    # f = open("BC4GE_Score_name.pkl","wb")
    # # write the python object (dict) to pickle file
    # pickle.dump(Prot_names,f)
    # # close file
    # f.close()




    # a_file = open("BC4GE_Score_data.pkl", "rb")
    # DIC_GENENAMES = pickle.load(a_file)
    # a_file.close()

    # np.inner(last_vect_bc.detach().numpy(),last_vect_bc.detach().numpy())
###############don't need#########
    # # Training
    # if args.do_train:
    #     if args.resume_path is not None:
    #         # Load a trained model and vocabulary from a saved checkpoint to resume training
    #         model.load_state_dict(torch.load(os.path.join(args.resume_path, 'pytorch_model-1000000.bin')))
    #         tokenizer = tokenizer_class.from_pretrained(args.resume_path)
    #         model.to(args.device)
    #         logger.info("INFO: Checkpoint loaded successfully. Training will resume from %s", args.resume_path)
    #     global_step, tr_loss = train(args, model, tokenizer)
    #     logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    # if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     # Create output directory if needed
    #     if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #         os.makedirs(args.output_dir)

    #     logger.info("Saving model checkpoint to %s", args.output_dir)
    #     # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    #     # They can then be reloaded using `from_pretrained()`
    #     model_to_save = (
    #         model.module if hasattr(model, "module") else model
    #     )  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(args.output_dir)
    #     tokenizer.save_pretrained(args.output_dir)

    #     # Good practice: save your training arguments together with the trained model
    #     torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    #     # Load a trained model and vocabulary that you have fine-tuned
    #     model.load_state_dict(torch.load(os.path.join(args.output_dir, 'pytorch_model-1000000.bin')))
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir)
    #     model.to(args.device)

    # # Evaluation
    # results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        # checkpoints = [args.output_dir]
        # if args.eval_all_checkpoints:
        #     checkpoints = list(
        #         os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
        #     )
        #     logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        # logger.info("Evaluate the following checkpoints: %s", checkpoints)
        # for checkpoint in checkpoints:
        #     global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        #     prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        #     model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model-1000000.bin')))
        #     model.to(args.device)
        #     result = evaluate(args, model, tokenizer, prefix=prefix)
        #     result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
        #     results.update(result)
###############don't need ---end---#########
    return None


if __name__ == "__main__":
    main()

# script:
# train 
#  python run_e2e_span.py --data_dir data/BC5CDR/processed_data --model_type bert --model_name_or_path ./scibert_scivocab_uncased --output_dir output_base --num_train_epochs 20 --use_random_candidates --do_train --no_cuda 

# eval
# python run_e2e_span.py --data_dir data/BC5CDR/processed_data --model_type bert --model_name_or_path ./biobert_v1.1_pubmed --output_dir output209n --use_random_candidates --do_eval --no_cuda 

# test for ge11
# python run_e2e_span.py --data_dir ./data/Ge11/dev/GE11_dev_candidate_25.pkl --model_type bert --model_name_or_path ./biobert_v1.1_pubmed --output_dir output206 --use_random_candidates --do_test

# test for ge11
# python run_e2e_span.py --data_dir ./data/Ge11/train/GE11_train_candidate_25.pkl --model_type bert --model_name_or_path ./biobert_v1.1_pubmed --output_dir output206 --use_random_candidates --do_test






