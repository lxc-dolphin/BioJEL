import os
import json
import random
import math
import pdb
import logging
logger = logging.getLogger(__name__)
import faiss
import torch

import pickle


def get_examples(data_dir, mode):
    if 'NCBI' in data_dir:
        entity_path = './data/NCBI_Disease/raw_data/entities.txt'
    elif 'BC5CDR' in data_dir:
        entity_path = './data/BC5CDR/raw_data/entities.txt'
    elif 'st21pv' in data_dir:
        entity_path = './data/MM_st21pv_CUI/raw_data/entities.txt'
    elif 'aida' in data_dir:
        entity_path = './data/aida-yago2-dataset/raw_data/entities.txt'
    elif 'dummy' in data_dir:
        entity_path = './data/dummy_data/raw_data/entities.txt'
    else:
        entity_path = './data/MM_full_CUI/raw_data/entities.txt'
    entities = {}
    with open(entity_path, encoding='utf-8') as f:
        for line in f:
            if 'BC5CDR' in data_dir:
                e, text = line.strip().split('\t')
            else:
                e, _, text = line.strip().split('\t')
            entities[e] = text

    file_path = os.path.join(data_dir, mode, 'documents/documents.json')
    docs = {}
    with open(file_path, encoding='utf-8') as f:
        print("documents dataset is loading......")
        for line in f:
            fields = json.loads(line.strip())
            docs[fields["document_id"]] = {"text": fields["text"]}
        print("documents dataset is done :)")

    # doc_ids = list(docs.keys())
    file_path = os.path.join(data_dir, mode, 'mentions/mentions.json')
    ments = {}
    with open(file_path, encoding='utf-8') as f:
        print("mentions {} dataset is loading......".format(mode))
        # doc_idx = 0
        for line in f:
            # doc_id = doc_ids[doc_idx]
            doc_mentions = json.loads(line.strip())
            if len(doc_mentions) > 0:
                doc_id = doc_mentions[0]["content_document_id"]
                ments[doc_id] = json.loads(line)
            # for line in f:
            #     fields = json.loads(line)
            #     ments[fields["mention_id"]] = {k: v for k, v in fields.items() if k != "mention_id"}
        print("mentions {} dataset is done :)".format(mode))
    return ments, docs, entities



def get_BC_examples(data_dir,max_seq_length,
            tokenizer,
            args):
   
    entity_path = './data/BC4GE_Matched_100data.json'
    
    entities = {}
    doc_bc = {}
    doc_match = {}

    a_file = open(entity_path, "r")
    MatchedGenes = json.loads(a_file.read())
    a_file.close()

    random_text = "A randomized comparison of labetalol and nitroprusside for induced hypotension. In a randomized study, labetalol-induced hypotension and nitroprusside-induced hypotension were compared in 20 patients (10 in each group) scheduled for major orthopedic procedures. Each patient was subjected to an identical anesthetic protocol and similar drug-induced reductions in mean arterial blood pressure"

    for prot, goid_BC_matched in MatchedGenes.items():
        BCid = goid_BC_matched[0]
        Matchedid = goid_BC_matched[1]

        entities[prot] = {}
        # ["def"]["text"]
        for bc_id, item in BCid.items():
            if prot not in doc_bc:
                doc_bc[prot] = item["name"]
            else: 
                doc_bc[prot] += item["name"]

        for match_id, item in Matchedid.items():
            if prot not in doc_match:
                doc_match[prot] = item["def"]["text"]
            else: 
                doc_match[prot] += item["def"]["text"]

    # convert docs to features --> get_marked_mentions: docs to doc_tokens
    features = []
    context_bc = doc_bc
    context_match = doc_match
    

    for prot in entities:
        token_context_bc_ = [tokenizer.cls_token]
        token_context_match_ = [tokenizer.cls_token]
        
        token_context_bc_ += tokenizer.tokenize(context_bc[prot])
        token_context_match_ += tokenizer.tokenize(context_match[prot])

        token_context_bc_ += [tokenizer.sep_token]
        token_context_match_ += [tokenizer.sep_token]

        token_bc = tokenizer.convert_tokens_to_ids(token_context_bc_)
        token_match = tokenizer.convert_tokens_to_ids(token_context_match_)

        if len(token_bc) > max_seq_length:
            print(len(token_bc))
            token_bc = token_bc[:max_seq_length]
            token_bc_mask = [1] * max_seq_length
        else:
            mention_len = len(token_bc)
            pad_len = max_seq_length - mention_len
            token_bc += [tokenizer.pad_token_id] * pad_len
            token_bc_mask = [1] * mention_len + [0] * pad_len

        if len(token_match) > max_seq_length:
            print(len(token_match))
            token_match = token_match[:max_seq_length]
            token_match_mask = [1] * max_seq_length
        else:
            mention_len = len(token_match)
            pad_len = max_seq_length - mention_len
            token_match += [tokenizer.pad_token_id] * pad_len
            token_match_mask = [1] * mention_len + [0] * pad_len

        features.append(
            InputFeatures1(
                mention_token_ids = token_bc, 
                mention_token_masks = token_bc_mask,
                candidate_token_ids = token_match, 
                candidate_token_masks = token_match_mask,
                mention_textname = prot
            )
        )

    return entities, doc_bc, doc_match, features


def get_mentions_tokens(Genedata,tokenizer):
    
    start_indx = Genedata['start']
    end_indx = Genedata['end']
    context_text = Genedata['text']
    mention_name = context_text[start_indx: end_indx]        
    
    tokenized_text = [tokenizer.cls_token]
    sequence_tags = []
    mention_start_markers = []
    mention_end_markers = []
    
    # tokenize the text before the mention 
    prefix  = context_text[0:start_indx]
    prefix_tokens = tokenizer.tokenize(prefix)        
    tokenized_text += prefix_tokens
    # The sequence tag for prefix tokens is 'O' , 'DNT' --> 'Do Not Tag'
    for j, token in enumerate(prefix_tokens):
        sequence_tags.append('O' if not token.startswith('##') else 'DNT')
    # Add mention start marker to the tokenized text
    mention_start_markers.append(len(tokenized_text))
    # Tokenize the mention and add it to the tokenized text
    mention_tokens = tokenizer.tokenize(mention_name)
    tokenized_text += mention_tokens
    # Sequence tags for mention tokens -- first token B, other tokens I
    for j, token in enumerate(mention_tokens):
        if j == 0:
            sequence_tags.append('B')
        else:
            sequence_tags.append('I' if not token.startswith('##') else 'DNT')
    # Add mention end marker to the tokenized text
    mention_end_markers.append(len(tokenized_text) - 1)
    
    # text after the mention
    suffix = context_text[end_indx:]
    if len(suffix)>0:
        suffix_tokens = tokenizer.tokenize(suffix)
        tokenized_text += suffix_tokens
        # The sequence tag for suffix tokens is 'O'
        for j, token in enumerate(suffix_tokens):
            sequence_tags.append('O' if not token.startswith('##') else 'DNT')
    tokenized_text += [tokenizer.sep_token]
    
    return tokenized_text, mention_start_markers, mention_end_markers, sequence_tags




def get_candi_tokens(CandiData, tokenizer):
    
    candi_text = CandiData['def']
    tokenize_text = tokenizer.tokenize(candi_text)
    sequence_tags = []
    for j, token in enumerate(tokenize_text):
            sequence_tags.append('O' if not token.startswith('##') else 'DNT')
    
    return tokenize_text, sequence_tags



def get_BC_examples_new(data_dir,max_seq_length,
            tokenizer,
            args):
   
    entity_path = './data/BC4GE_data_PosiNegaCandi_train25.json'
        
    a_file = open(entity_path, "r")
    Gene_data_PosiNega = json.loads(a_file.read())
    a_file.close()

    features = []
    
    for case_id, val in Gene_data_PosiNega.items():
        
        
        
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
            
            
            
        # tag_to_id_map = {'O': 0, 'B': 1, 'I': 2, 'DNT': -100}
        # def convert_tags_to_ids(seq_tags):
        #     seq_tag_ids = [-100]  # corresponds to the [CLS] token
        #     for t in seq_tags:
        #         seq_tag_ids.append(tag_to_id_map[t])
        #     seq_tag_ids.append(-100)  # corresponds to the [SEP] token
        #     return seq_tag_ids
    return features


def get_BC_examples_new_dev(data_dir,max_seq_length,
            tokenizer,
            args):
   
    entity_path = './data/BC4GE_data_PosiNegaCandi_dev25n.json'
        
    a_file = open(entity_path, "r")
    Gene_data_PosiNega = json.loads(a_file.read())
    a_file.close()

    features = []
    
    for case_id, val in Gene_data_PosiNega.items():
        
        
        
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
            
            
            
        # tag_to_id_map = {'O': 0, 'B': 1, 'I': 2, 'DNT': -100}
        # def convert_tags_to_ids(seq_tags):
        #     seq_tag_ids = [-100]  # corresponds to the [CLS] token
        #     for t in seq_tags:
        #         seq_tag_ids.append(tag_to_id_map[t])
        #     seq_tag_ids.append(-100)  # corresponds to the [SEP] token
        #     return seq_tag_ids
    return features


def get_BC_examples_new_test(data_dir,max_seq_length,
            tokenizer,
            args):
   
    entity_path = './data/BC4GE_data_PosiNegaCandi_test25n.json'
        
    a_file = open(entity_path, "r")
    Gene_data_PosiNega = json.loads(a_file.read())
    a_file.close()

    features = []
    
    for case_id, val in Gene_data_PosiNega.items():
        
        
        
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
            
            
            
        # tag_to_id_map = {'O': 0, 'B': 1, 'I': 2, 'DNT': -100}
        # def convert_tags_to_ids(seq_tags):
        #     seq_tag_ids = [-100]  # corresponds to the [CLS] token
        #     for t in seq_tags:
        #         seq_tag_ids.append(tag_to_id_map[t])
        #     seq_tag_ids.append(-100)  # corresponds to the [SEP] token
        #     return seq_tag_ids
    return features


def get_GE_mentions_tokens(sentence,posi_start,posi_end,tokenizer):
    
    start_indx = int(posi_start)
    end_indx = int(posi_end)
    context_text = sentence
    mention_name = context_text[start_indx: end_indx]        
    
    tokenized_text = [tokenizer.cls_token]
    sequence_tags = []
    mention_start_markers = []
    mention_end_markers = []
    
    # tokenize the text before the mention 
    prefix  = context_text[0:start_indx]
    prefix_tokens = tokenizer.tokenize(prefix)        
    tokenized_text += prefix_tokens
    # The sequence tag for prefix tokens is 'O' , 'DNT' --> 'Do Not Tag'
    for j, token in enumerate(prefix_tokens):
        sequence_tags.append('O' if not token.startswith('##') else 'DNT')
    # Add mention start marker to the tokenized text
    mention_start_markers.append(len(tokenized_text))
    # Tokenize the mention and add it to the tokenized text
    mention_tokens = tokenizer.tokenize(mention_name)
    tokenized_text += mention_tokens
    # Sequence tags for mention tokens -- first token B, other tokens I
    for j, token in enumerate(mention_tokens):
        if j == 0:
            sequence_tags.append('B')
        else:
            sequence_tags.append('I' if not token.startswith('##') else 'DNT')
    # Add mention end marker to the tokenized text
    mention_end_markers.append(len(tokenized_text) - 1)
    
    # text after the mention
    suffix = context_text[end_indx:]
    if len(suffix)>0:
        suffix_tokens = tokenizer.tokenize(suffix)
        tokenized_text += suffix_tokens
        # The sequence tag for suffix tokens is 'O'
        for j, token in enumerate(suffix_tokens):
            sequence_tags.append('O' if not token.startswith('##') else 'DNT')
    tokenized_text += [tokenizer.sep_token]
    
    return tokenized_text, mention_start_markers, mention_end_markers, sequence_tags





def get_GE_examples(data_dir,max_seq_length,
            tokenizer,
            args):
    
    entity_path = data_dir
    f = open(entity_path,"rb")
    sentences_all = pickle.load(f)
    f.close()   
    
        
    # a_file = open(entity_path, "r")
    # Gene_data_PosiNega = json.loads(a_file.read())
    # a_file.close()

    features = []
    sentence_entity_candi_idx = []
    for val in sentences_all: # for each sentence
        
        
        
        sentence = val['sentence']
        entity_candi = val['entity_candi']
        terms = val['terms']
        
        entity_candi_numb = []
                
        for entity_name in entity_candi.keys(): # for each entity
            
            candidates = entity_candi[entity_name]
             
            entity_candi_numb.append(len(candidates)) # the candidate number for each entity
            
            for term_each in terms:
                if term_each[4] == entity_name:
                    start_posi = term_each[2]
                    end_posi = term_each[3]
            
            
            tokenized_text_, mention_start_markers, mention_end_markers, sequence_tags \
            = get_GE_mentions_tokens(sentence,start_posi, end_posi, tokenizer)
        
            doc_tokens = tokenizer.convert_tokens_to_ids(tokenized_text_)
            seq_tag_ids = convert_tags_to_ids(sequence_tags)
            
            for go_id ,each_candi in candidates.items():
                
                candi_token, candi_seq = get_candi_tokens(each_candi,tokenizer)
                candi_seq = convert_tags_to_ids(candi_seq)
                candi_token = tokenizer.convert_tokens_to_ids(candi_token)
            
                token_bc_candi = doc_tokens + candi_token
                sequence_tags_bc_candi = seq_tag_ids + candi_seq

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
                    
                    pad_len_seq = max_seq_length - len(sequence_tags_bc_candi)
                    sequence_tags_bc_candi += [-100]*pad_len_seq
                    
                    assert len(token_bc_candi) == len(sequence_tags_bc_candi)
                
                features.append(
                InputFeatures1(
                    mention_token_ids = token_bc_candi, 
                    mention_token_masks = token_bc_mask,
                    sequence_tags = sequence_tags_bc_candi,
                    result = 1.0, 
                    mention_textname = entity_name
                )
                )

            
             
        sentence_entity_candi_idx.append(entity_candi_numb)
            
    return features, sentence_entity_candi_idx


def get_window(prefix, mention, suffix, max_size):
    if len(mention) >= max_size:
        window = mention[:max_size]
        return window, 0, len(window) - 1

    leftover = max_size - len(mention)
    leftover_half = int(math.ceil(leftover / 2))

    if len(prefix) >= leftover_half:
        prefix_len = leftover_half if len(suffix) >= leftover_half else \
                     leftover - len(suffix)
    else:
        prefix_len = len(prefix)

    prefix = prefix[-prefix_len:]  # Truncate head of prefix
    window = prefix + ['[Ms]'] + mention + ['[Me]'] + suffix
    window = window[:max_size]  # Truncate tail of suffix

    mention_start_index = len(prefix)
    mention_end_index = len(prefix) + len(mention) - 1

    return window, mention_start_index, mention_end_index


def get_mention_window(mention_id, mentions, docs,  max_seq_length, tokenizer):
    max_len_context = max_seq_length - 2 # number of characters
    # Get "enough" context from space-tokenized text.
    content_document_id = mentions[mention_id]['content_document_id']
    context_text = docs[content_document_id]['text']
    start_index = mentions[mention_id]['start_index']
    end_index = mentions[mention_id]['end_index']
    prefix = context_text[max(0, start_index - max_len_context): start_index]
    suffix = context_text[end_index: end_index + max_len_context]
    extracted_mention = context_text[start_index: end_index]

    assert extracted_mention == mentions[mention_id]['text']

    # Get window under new tokenization.
    return get_window(tokenizer.tokenize(prefix),
                      tokenizer.tokenize(extracted_mention),
                      tokenizer.tokenize(suffix),
                      max_len_context)
    
    


def get_marked_mentions(document_id, mentions, docs,  max_seq_length, tokenizer, args):
    # print("Num mention in this doc =", len(mentions[document_id]))
    for m in mentions[document_id]:
        assert m['content_document_id'] == document_id

    context_text = docs[document_id]['text'].lower() if args.do_lower_case else docs[document_id]['text']
    tokenized_text = [tokenizer.cls_token]
    mention_start_markers = []
    mention_end_markers = []
    sequence_tags = []

    # print(len(context_text))
    # print(len(mentions[document_id]))
    prev_end_index = 0
    for m in mentions[document_id]:
        start_index = m['start_index']
        end_index = m['end_index']
        # print(start_index, end_index)
        if start_index >= len(context_text):
            continue
        extracted_mention = context_text[start_index: end_index]

        # Text between the end of last mention and the beginning of current mention
        prefix = context_text[prev_end_index: start_index]
        # Tokenize prefix and add it to the tokenized text
        prefix_tokens = tokenizer.tokenize(prefix)
        tokenized_text += prefix_tokens
        # The sequence tag for prefix tokens is 'O' , 'DNT' --> 'Do Not Tag'
        for j, token in enumerate(prefix_tokens):
            sequence_tags.append('O' if not token.startswith('##') else 'DNT')
        # Add mention start marker to the tokenized text
        mention_start_markers.append(len(tokenized_text))
        # tokenized_text += ['[Ms]']
        # Tokenize the mention and add it to the tokenized text
        mention_tokens = tokenizer.tokenize(extracted_mention)
        tokenized_text += mention_tokens
        # Sequence tags for mention tokens -- first token B, other tokens I
        for j, token in enumerate(mention_tokens):
            if j == 0:
                sequence_tags.append('B')
            else:
                sequence_tags.append('I' if not token.startswith('##') else 'DNT')

        # Add mention end marker to the tokenized text
        mention_end_markers.append(len(tokenized_text) - 1)
        # tokenized_text += ['[Me]']
        # Update prev_end_index
        prev_end_index = end_index

    suffix = context_text[prev_end_index:]
    if len(suffix) > 0:
        suffix_tokens = tokenizer.tokenize(suffix)
        tokenized_text += suffix_tokens
        # The sequence tag for suffix tokens is 'O'
        for j, token in enumerate(suffix_tokens):
            sequence_tags.append('O' if not token.startswith('##') else 'DNT')
    tokenized_text += [tokenizer.sep_token]

    return tokenized_text, mention_start_markers, mention_end_markers, sequence_tags


def get_entity_window(entity_text, max_entity_len, tokenizer):
    entity_tokens = tokenizer.tokenize(entity_text)
    if len(entity_tokens) > max_entity_len:
        entity_tokens = entity_tokens[:max_entity_len]
    return entity_tokens

class InputFeatures1(object):
    def __init__(self, mention_token_ids, mention_token_masks
                , sequence_tags, result,
                mention_textname
                 ):
        self.mention_token_ids = mention_token_ids
        self.mention_token_masks = mention_token_masks
        self.sequence_tags = sequence_tags
        self.result = result
        self.mention_textname = mention_textname


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, mention_token_ids, mention_token_masks,
                 candidate_token_ids_1, candidate_token_masks_1,
                 candidate_token_ids_2, candidate_token_masks_2,
                 label_ids, mention_start_indices, mention_end_indices,
                 num_mentions, seq_tag_ids):
        self.mention_token_ids = mention_token_ids
        self.mention_token_masks = mention_token_masks
        self.candidate_token_ids_1 = candidate_token_ids_1
        self.candidate_token_masks_1 = candidate_token_masks_1
        self.candidate_token_ids_2 = candidate_token_ids_2
        self.candidate_token_masks_2 = candidate_token_masks_2
        self.label_ids = label_ids
        self.mention_start_indices = mention_start_indices
        self.mention_end_indices = mention_end_indices,
        self.num_mentions = num_mentions
        self.seq_tag_ids = seq_tag_ids

tag_to_id_map = {'O': 0, 'B': 1, 'I': 2, 'DNT': -100}
def convert_tags_to_ids(seq_tags):
    seq_tag_ids = [-100]  # corresponds to the [CLS] token
    for t in seq_tags:
        seq_tag_ids.append(tag_to_id_map[t])
    seq_tag_ids.append(-100)  # corresponds to the [SEP] token
    return seq_tag_ids

def convert_examples_to_features(
    mentions,
    docs,
    entities,
    max_seq_length,
    tokenizer,
    args,
    model=None,
):

    # All entities
    all_entities = list(entities.keys())
    all_entity_token_ids = []
    all_entity_token_masks = []

    for c_idx, c in enumerate(all_entities):
        entity_text = entities[c].lower() if args.do_lower_case else entities[c]
        max_entity_len = max_seq_length // 4  # Number of tokens
        entity_window = get_entity_window(entity_text, max_entity_len, tokenizer)
        # [CLS] candidate text [SEP]
        candidate_tokens = [tokenizer.cls_token] + entity_window + [tokenizer.sep_token]
        candidate_tokens = tokenizer.convert_tokens_to_ids(candidate_tokens)
        if len(candidate_tokens) > max_seq_length:
            candidate_tokens = candidate_tokens[:max_seq_length]
            candidate_masks = [1] * max_seq_length
        else:
            candidate_len = len(candidate_tokens)
            pad_len = max_seq_length - candidate_len
            candidate_tokens += [tokenizer.pad_token_id] * pad_len
            candidate_masks = [1] * candidate_len + [0] * pad_len

        assert len(candidate_tokens) == max_seq_length
        assert len(candidate_masks) == max_seq_length

        all_entity_token_ids.append(candidate_tokens)
        all_entity_token_masks.append(candidate_masks)

    if args.use_hard_negatives or args.use_hard_and_random_negatives:
        if model is None:
            raise ValueError("`model` parameter cannot be None")
        logger.info("INFO: Building index of the candidate embeddings ...")
        # Gather all candidate embeddings for hard negative mining
        all_candidate_embeddings = []
        with torch.no_grad():
            # Forward pass through the candidate encoder of the dual encoder
            for i, (entity_tokens, entity_tokens_masks) in enumerate(
                    zip(all_entity_token_ids, all_entity_token_masks)):
                candidate_token_ids = torch.LongTensor([entity_tokens]).to(args.device)
                candidate_token_masks = torch.LongTensor([entity_tokens_masks]).to(args.device)
                if hasattr(model, "module"):
                    candidate_outputs = model.module.bert_candidate.bert(
                        input_ids=candidate_token_ids,
                        attention_mask=candidate_token_masks,
                    )
                else:
                    candidate_outputs = model.bert_candidate.bert(
                        input_ids=candidate_token_ids,
                        attention_mask=candidate_token_masks,
                    )
                candidate_embedding = candidate_outputs[1]
                all_candidate_embeddings.append(candidate_embedding)

        all_candidate_embeddings = torch.cat(all_candidate_embeddings, dim=0)

        # Indexing for faster search (using FAISS)
        # d = all_candidate_embeddings.size(1)
        # all_candidate_index = faiss.IndexFlatL2(d)  # build the index, d=size of vectors
        # here we assume `all_candidate_embeddings` contains a n-by-d numpy matrix of type float32
        # all_candidate_embeddings = all_candidate_embeddings.cpu().detach().numpy()
        # all_candidate_index.add(all_candidate_embeddings)

    if args.use_hard_and_random_negatives:
        # Get the existing hard negatives per mention
        if os.path.exists(os.path.join(args.data_dir, 'mention_hard_negatives.json')):
            with open(os.path.join(args.data_dir, 'mention_hard_negatives.json')) as f_hn:
                mention_hard_negatives = json.load(f_hn)
        else:
            mention_hard_negatives = {}

    # Process the mentions
    features = []
    position_of_positive = {}
    num_longer_docs = 0
    all_document_ids = []
    all_label_candidate_ids = []
    for (ex_index, document_id) in enumerate(mentions.keys()):
        # pdb.set_trace()
        # print(document_id)
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(mentions))

        # mention_window, mention_start_index, mention_end_index = get_mention_window(mention_id,
        #                                                                     mentions,
        #                                                                     docs,
        #                                                                     max_seq_length,
        #                                                                     tokenizer)

        doc_tokens_, mention_start_markers, mention_end_markers, seq_tags = get_marked_mentions(document_id,
                                                                            mentions,
                                                                            docs,
                                                                            max_seq_length,
                                                                            tokenizer,
                                                                            args)

        # print(mention_start_markers, mention_end_markers)
        doc_tokens = tokenizer.convert_tokens_to_ids(doc_tokens_)
        seq_tag_ids = convert_tags_to_ids(seq_tags)

        assert len(doc_tokens) == len(seq_tag_ids)


        if len(doc_tokens) > max_seq_length:
            print(len(doc_tokens))
            doc_tokens = doc_tokens[:max_seq_length]
            seq_tag_ids = seq_tag_ids[:max_seq_length]
            doc_tokens_mask = [1] * max_seq_length
            num_longer_docs += 1
            continue
        else:
            mention_len = len(doc_tokens)
            pad_len = max_seq_length - mention_len
            doc_tokens += [tokenizer.pad_token_id] * pad_len
            doc_tokens_mask = [1] * mention_len + [0] * pad_len
            seq_tag_ids += [-100] * pad_len

        assert len(doc_tokens) == max_seq_length
        assert len(doc_tokens_mask) == max_seq_length
        assert len(seq_tag_ids) == max_seq_length

        # Build list of candidates
        label_candidate_ids = []
        for m in mentions[document_id]:
            label_candidate_ids.append(m['label_candidate_id'])
            all_document_ids.append(document_id)
            all_label_candidate_ids.append(m['label_candidate_id'])

        candidates = []
        candidates_2 = None
        if args.do_train:
            if args.use_random_candidates:  # Random negatives
                for m_idx, m in enumerate(mentions[document_id]):
                    m_candidates = []
                    m_candidates.append(label_candidate_ids[m_idx])  # positive candidate
                    candidate_pool = set(entities.keys()) - set([label_candidate_ids[m_idx]])
                    negative_candidates = random.sample(candidate_pool, args.num_candidates - 1)
                    m_candidates += negative_candidates
                    candidates.append(m_candidates)

            elif args.use_tfidf_candidates:  # TF-IDF negatives
                for m_idx, m in enumerate(mentions[document_id]):
                    m_candidates = []
                    m_candidates.append(label_candidate_ids[m_idx])  # positive candidate
                    for c in m["tfidf_candidates"]:
                        if c != label_candidate_ids[m_idx] and len(m_candidates) < args.num_candidates:
                            m_candidates.append(c)
                    candidates.append(m_candidates)

            elif args.use_hard_and_random_negatives:
                # First get the random negatives
                for m_idx, m in enumerate(mentions[document_id]):
                    m_candidates = []
                    m_candidates.append(label_candidate_ids[m_idx])  # positive candidate
                    candidate_pool = set(entities.keys()) - set([label_candidate_ids[m_idx]])
                    negative_candidates = random.sample(candidate_pool, args.num_candidates - 1)
                    m_candidates += negative_candidates
                    candidates.append(m_candidates)

                # Then get the hard negative
                if model is None:
                    raise ValueError("`model` parameter cannot be None")
                # Hard negative candidate mining
                # print("Performing hard negative candidate mining ...")
                # Get mention embeddings
                input_token_ids = torch.LongTensor([doc_tokens]).to(args.device)
                input_token_masks = torch.LongTensor([doc_tokens_mask]).to(args.device)
                # Forward pass through the mention encoder of the dual encoder
                with torch.no_grad():
                    if hasattr(model, "module"):
                        mention_outputs = model.module.bert_mention.bert(
                            input_ids=input_token_ids,
                            attention_mask=input_token_masks,
                        )
                    else:
                        mention_outputs = model.bert_mention.bert(
                            input_ids=input_token_ids,
                            attention_mask=input_token_masks,
                        )
                last_hidden_states = mention_outputs[0]  # B X L X H
                # Pool the mention representations
                # mention_start_indices = torch.LongTensor([mention_start_markers]).to(args.device)
                # mention_end_indices = torch.LongTensor([mention_end_markers]).to(args.device)
                #
                if hasattr(model, "module"):
                    hidden_size = model.module.hidden_size
                else:
                    hidden_size = model.hidden_size
                #
                # mention_start_indices = mention_start_indices.unsqueeze(-1).expand(-1, -1, hidden_size)
                # mention_end_indices = mention_end_indices.unsqueeze(-1).expand(-1, -1, hidden_size)
                # mention_start_embd = last_hidden_states.gather(1, mention_start_indices)
                # mention_end_embd = last_hidden_states.gather(1, mention_end_indices)
                # if hasattr(model, "module"):
                #     mention_embeddings = model.module.mlp(torch.cat([mention_start_embd, mention_end_embd], dim=2))
                # else:
                #     mention_embeddings = model.mlp(torch.cat([mention_start_embd, mention_end_embd], dim=2))
                # mention_embeddings = mention_embeddings.reshape(-1, 1, hidden_size) # M X 1 X H

                mention_embeddings = []
                # print(mention_start_markers, mention_end_markers)
                for i, (s_idx, e_idx) in enumerate(zip(mention_start_markers, mention_end_markers)):
                    m_embd = torch.mean(last_hidden_states[:, s_idx:e_idx+1, :], dim=1)
                    mention_embeddings.append(m_embd)
                mention_embeddings = torch.cat(mention_embeddings, dim=0).unsqueeze(1)

                # Perform similarity search
                num_m = mention_embeddings.size(0)  #
                all_candidate_embeddings_ = all_candidate_embeddings.unsqueeze(0).expand(num_m, -1, hidden_size) # M X C_all X H

                # distance, candidate_indices = all_candidate_index.search(mention_embedding, args.num_candidates)
                # candidate_indices = candidate_indices[0]  # original size 1 X 10 -> 10
                # print(mention_embeddings)
                similarity_scores = torch.bmm(mention_embeddings,
                                              all_candidate_embeddings_.transpose(1, 2))  # M X 1 X C_all
                similarity_scores = similarity_scores.squeeze(1)  # M X C_all
                # print(similarity_scores)
                distance, candidate_indices = torch.topk(similarity_scores, k=args.num_candidates)

                candidate_indices = candidate_indices.cpu().detach().numpy().tolist()
                # print(candidate_indices)

                # print(len(mentions[document_id]))
                for m_idx, m in enumerate(mentions[document_id]):
                    mention_id = m["mention_id"]
                    # Update the list of hard negatives for this `mention_id`
                    if mention_id not in mention_hard_negatives:
                        mention_hard_negatives[mention_id] = []
                    # print(m_idx)
                    for i, c_idx in enumerate(candidate_indices[m_idx]):
                        c = all_entities[c_idx]
                        if c == m["label_candidate_id"]:  # Positive candidate position
                            if i not in position_of_positive:
                                position_of_positive[i] = 1
                            else:
                                position_of_positive[i] += 1
                            break
                        else:
                            # Append new hard negatives
                            if c not in mention_hard_negatives[mention_id]:
                                mention_hard_negatives[mention_id].append(c)

                candidates_2 = []
                # candidates_2.append(label_candidate_id)  # positive candidate
                # Append hard negative candidates
                for m_idx, m in enumerate(mentions[document_id]):
                    mention_id = m["mention_id"]
                    if len(mention_hard_negatives[mention_id]) < args.num_candidates:  # args.num_candidates - 1
                        m_hard_candidates = mention_hard_negatives[mention_id]
                    else:
                        candidate_pool = mention_hard_negatives[mention_id]
                        m_hard_candidates = random.sample(candidate_pool, args.num_candidates)  # args.num_candidates - 1
                    candidates_2.append(m_hard_candidates)

        elif args.do_eval:
            for m_idx, m in enumerate(mentions[document_id]):
                m_candidates = []

                if args.include_positive:
                    m_candidates.append(label_candidate_ids[m_idx])  # positive candidate
                    for c in m["tfidf_candidates"]:
                        if c != label_candidate_ids[m_idx] and len(m_candidates) < args.num_candidates:
                            m_candidates.append(c)
                elif args.use_tfidf_candidates:
                    for c in m["tfidf_candidates"]:
                        m_candidates.append(c)
                elif args.use_all_candidates:
                    m_candidates = all_entities

                candidates.append(m_candidates)

        # Number of mentions in the documents
        num_mentions = len(mentions[document_id])

        if args.use_all_candidates:
            # If all candidates are considered during inference,
            # then place dummy candidate tokens and candidate masks
            candidate_token_ids_1 = None
            candidate_token_masks_1 = None
            candidate_token_ids_2 = None
            candidate_token_masks_2 = None
        else:
            candidate_token_ids_1 = [[tokenizer.pad_token_id] * max_entity_len] * (args.num_max_mentions * args.num_candidates)
            candidate_token_masks_1 = [[0]*max_entity_len] * (args.num_max_mentions * args.num_candidates)
            candidate_token_ids_2 = None
            candidate_token_masks_2 = None

            c_idx = 0
            for m_idx, m_candidates in enumerate(candidates):
                if m_idx >= args.num_max_mentions:
                    logger.warning("More than {} mentions in doc, mentions after {} are ignored".format(
                            args.num_max_mentions, args.num_max_mentions))
                    break
                for c in m_candidates:
                    if c in entities:
                        entity_text = entities[c].lower() if args.do_lower_case else entities[c]
                        max_entity_len = max_seq_length // 4  # Number of tokens
                        entity_window = get_entity_window(entity_text, max_entity_len, tokenizer)
                        # [CLS] candidate text [SEP]
                        candidate_tokens = [tokenizer.cls_token] + entity_window + [tokenizer.sep_token]
                        candidate_tokens = tokenizer.convert_tokens_to_ids(candidate_tokens)
                    else:
                        candidate_tokens = [0]*max_entity_len
                    if len(candidate_tokens) > max_entity_len:
                        candidate_tokens = candidate_tokens[:max_entity_len]
                        candidate_masks = [1] * max_entity_len
                    else:
                        candidate_len = len(candidate_tokens)
                        pad_len = max_entity_len - candidate_len
                        candidate_tokens += [tokenizer.pad_token_id] * pad_len
                        candidate_masks = [1] * candidate_len + [0] * pad_len

                    assert len(candidate_tokens) == max_entity_len
                    assert len(candidate_masks) == max_entity_len
                    candidate_token_ids_1[c_idx] = candidate_tokens
                    candidate_token_masks_1[c_idx] = candidate_masks
                    c_idx += 1

            # This second set of candidates is required for Gillick et al. hard negative training
            if candidates_2 is not None:
                candidate_token_ids_2 = [[tokenizer.pad_token_id] * max_entity_len] * (
                        args.num_max_mentions * args.num_candidates)
                candidate_token_masks_2 = [[0] * max_entity_len] * (args.num_max_mentions * args.num_candidates)

                for m_idx, m_hard_candidates in enumerate(candidates_2):
                    if m_idx >= args.num_max_mentions:
                        logger.warning("More than {} mentions in doc, mentions after {} are ignored".format(
                                args.num_max_mentions, args.num_max_mentions))
                        break
                    c_idx = m_idx * args.num_candidates
                    for c in m_hard_candidates:
                        entity_text = entities[c].lower() if args.do_lower_case else entities[c]
                        max_entity_len = max_seq_length // 4  # Number of tokens
                        entity_window = get_entity_window(entity_text, max_entity_len, tokenizer)
                        # [CLS] candidate text [SEP]
                        candidate_tokens = [tokenizer.cls_token] + entity_window + [tokenizer.sep_token]
                        candidate_tokens = tokenizer.convert_tokens_to_ids(candidate_tokens)
                        if len(candidate_tokens) > max_entity_len:
                            candidate_tokens = candidate_tokens[:max_entity_len]
                            candidate_masks = [1] * max_entity_len
                        else:
                            candidate_len = len(candidate_tokens)
                            pad_len = max_entity_len - candidate_len
                            candidate_tokens += [tokenizer.pad_token_id] * pad_len
                            candidate_masks = [1] * candidate_len + [0] * pad_len

                        assert len(candidate_tokens) == max_entity_len
                        assert len(candidate_masks) == max_entity_len

                        candidate_token_ids_2[c_idx] = candidate_tokens
                        candidate_token_masks_2[c_idx] = candidate_masks
                        c_idx += 1

        # Target candidate
        label_ids = [-1] * args.num_max_mentions
        for m_idx, m_candidates in enumerate(candidates):
            if m_idx >= args.num_max_mentions:
                logger.warning("More than {} mentions in doc, mentions after {} are ignored".format(
                    args.num_max_mentions, args.num_max_mentions))
                break
            if label_candidate_ids[m_idx] in m_candidates:
                label_ids[m_idx] = m_candidates.index(label_candidate_ids[m_idx])
            else:
                label_ids[m_idx] = -100 # when target candidate not in candidate set

        # Pad the mention start and end indices
        mention_start_indices = [0] * args.num_max_mentions
        mention_end_indices = [0] * args.num_max_mentions
        if num_mentions <= args.num_max_mentions:
            mention_start_indices[:num_mentions] = mention_start_markers
            mention_end_indices[:num_mentions] = mention_end_markers
        else:
            mention_start_indices = mention_start_markers[:args.num_max_mentions]
            mention_end_indices = mention_end_markers[:args.num_max_mentions]
        # if ex_index < 3:
        #     logger.info("*** Example ***")
        #     logger.info("mention_token_ids: %s", " ".join([str(x) for x in mention_tokens]))
        #     logger.info("mention_token_masks: %s", " ".join([str(x) for x in mention_tokens_mask]))
        #     if candidate_token_ids_1 is not None:
        #         logger.info("candidate_token_ids_1: %s", " ".join([str(x) for x in candidate_token_ids_1]))
        #         logger.info("candidate_token_masks_1: %s", " ".join([str(x) for x in candidate_token_masks_1]))
        #     if candidate_token_ids_2 is not None:
        #         logger.info("candidate_token_ids_2: %s", " ".join([str(x) for x in candidate_token_ids_2]))
        #         logger.info("candidate_token_masks_2: %s", " ".join([str(x) for x in candidate_token_masks_2]))
        #     logger.info("label_ids: %s", " ".join([str(x) for x in label_id]))

        features.append(
            InputFeatures(mention_token_ids=doc_tokens,
                          mention_token_masks=doc_tokens_mask,
                          candidate_token_ids_1=candidate_token_ids_1,
                          candidate_token_masks_1=candidate_token_masks_1,
                          candidate_token_ids_2=candidate_token_ids_2,
                          candidate_token_masks_2=candidate_token_masks_2,
                          label_ids=label_ids,
                          mention_start_indices=mention_start_indices,
                          mention_end_indices=mention_end_indices,
                          num_mentions=num_mentions,
                          seq_tag_ids=seq_tag_ids,
                          )
        )


        # if ex_index == 4:
        #     break

    logger.info("*** Position of the positive candidates ***")
    print(position_of_positive)
    print(num_longer_docs)

    # Save the hard negatives
    if args.use_hard_and_random_negatives:
        with open(os.path.join(args.data_dir, 'mention_hard_negatives.json'), 'w+') as f_hn:
            json.dump(mention_hard_negatives, f_hn)
        f_hn.close()

    return features, (all_entities, all_entity_token_ids, all_entity_token_masks), (all_document_ids, all_label_candidate_ids)

# data_dir = './data/NCBI_Disease/collective_el_data_2'
# mode = "test"
#
# examples, docs, entities = get_examples(data_dir, mode)
# print(entities)
# for doc_id in examples:
#     print(len(examples[doc_id]))
#     print(docs[doc_id])
#
#     print(">>>>>>>>")
# print(len(entities))

