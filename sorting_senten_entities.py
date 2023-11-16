# sort each sentence with the relative mentions or entities
import pickle

from sqlalchemy import all_


path1 = "GE11_train_candidate_25_preds.pkl"
f = open(path1,"rb")
all_preds = pickle.load(f)
f.close()

path2 = "GE11_train_candidate_25_preds_candi_number.pkl"
f = open(path2,"rb")
sen_enti_candi_number = pickle.load(f)
f.close()

count_pred_idx = 0

sentence_candi_idx = [] # store the candidate idxs with highest score 
sentence_candi_sc = []

for sentence_candi in sen_enti_candi_number: # [24,25]
    # for each sentence store the candidate idxs with hightest scores
    _sent_candi_idx = []
    _sent_candi_sc = []
    for candi_numb in sentence_candi:  # 24
        _candi_preds = [] # store the candidate scores for each candidates
        for i in range(candi_numb):
            
            _candi_preds.append(all_preds[count_pred_idx])
            count_pred_idx += 1
            
        if _candi_preds != []:               
            _highest_sc = max(_candi_preds)
            _indx_pred = _candi_preds.index(_highest_sc)
            
            _sent_candi_idx.append(_indx_pred)
            _sent_candi_sc.append(_highest_sc)
        
    sentence_candi_idx.append(_sent_candi_idx)
    sentence_candi_sc.append(_sent_candi_sc)


path3 = "./data/Ge11/train/GE11_train_candidate_25.pkl"
f = open(path3,"rb")
ge11_dev_candi_data = pickle.load(f)
f.close()

for _idx, each_case in enumerate(ge11_dev_candi_data):

    posi_candi_idx = sentence_candi_idx[_idx]
    posi_candi_sc = sentence_candi_sc[_idx]
    
    entity_candi = each_case['entity_candi'] # dict {'entitiy1':{candis}, ...}
    
    each_case['posi_candi'] = []
    
    ith_entity = 0
    for entity_name in entity_candi.keys(): # entity name
        
        if entity_candi[entity_name] != {}:        
            _all_candi = list(entity_candi[entity_name].items())
            
            posi_candi = list(_all_candi[posi_candi_idx[ith_entity]])
            posi_candi_sc_each = posi_candi_sc[ith_entity]
            
            posi_candi.append(posi_candi_sc_each)
            posi_candi.append(entity_name)
            
            
            ith_entity += 1    

            each_case['posi_candi'].append(posi_candi)
            
    
f = open("GE11_train_candidate_25_positive.pkl","wb")
# jj = json.dumps(sentences_all)
pickle.dump(ge11_dev_candi_data,f)
# f.write(jj)
f.close()

    





