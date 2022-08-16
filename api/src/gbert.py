from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import fasttext
import string
from collections import Counter
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from collections import defaultdict
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import string
import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
DEVICE = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert=BertModel.from_pretrained('bert-base-uncased')
bert.eval()
maskedlm = BertForMaskedLM.from_pretrained('bert-base-uncased')
maskedlm.eval()
maskedlm.to(DEVICE)

train_idx_features_labels = np.genfromtxt("src/node_graph_reconstruct_model_directed.txt", dtype=np.dtype(str))
gbert_embs = {}
for i in range(train_idx_features_labels.shape[0]):
    gbert_embs[train_idx_features_labels[i, 0]] = np.array(train_idx_features_labels[i,1:], np.dtype(float))

BATCH_SIZE = 64
HIDDEN_STATES = [1024, 512]
LEARNING_RATE = 1e-4
NUMBER_NEGS = 1
EPOCHS = 500


class ProjectionLayer(nn.Module):
    def __init__(self, emb_size=300, hidden_sizes=[1024, 512], target_size=300, device='cuda'):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(emb_size,hidden_sizes[0]), 
            nn.ELU(), 
            nn.Linear(hidden_sizes[0],hidden_sizes[1]), 
            nn.ELU(), 
            nn.Linear(hidden_sizes[1],target_size)
        ).to(device).train()
        
    def forward(self, x):
        return self.layers(x)


model = ProjectionLayer(emb_size=32, hidden_sizes=HIDDEN_STATES, target_size=768, device=DEVICE).to(DEVICE)
model.load_state_dict(torch.load("src/projection_model_gb-b.pt", map_location=torch.device('cpu')))
model.eval()

stop_words = set(stopwords.words('english'))
allowed = set(string.ascii_lowercase + "#" + string.ascii_uppercase)

def is_alpha(test_str):
    return set(test_str) <= allowed

def filter_results(res):
    res_filter = []

    for r in res:
        if isinstance(r[0], list):
            r_un = list(dict.fromkeys(r[0]))
            r_un_clean = []
            for tok in r_un:
                if is_alpha(tok.lower()) and not (tok.lower() in stop_words):
                    r_un_clean.append(tok)

            if len(r_un_clean)>0:
                if len(r_un_clean) == 1 and (r_un_clean[0] not in [el[0] for el in res_filter]):
                    res_filter.append((r_un_clean[0], r[1]))

                elif len(r_un_clean) > 1 and (r_un_clean not in [el[0] for el in res_filter]):
                    res_filter.append((r_un_clean, r[1]))
                else:
                    continue
        else:
            if is_alpha(r[0].lower()) and not (r[0].lower() in stop_words) and (r[0].lower() not in [t[0] for t in res_filter]):
                res_filter.append((r[0], r[1]))
                
    res_filter.sort(key=lambda x: x[1], reverse=True)      
    return res_filter


def get_pred(inputs, subst_type='no', pred_child_emb=None, num=50):
    outputs = maskedlm.bert(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    hidden_states = outputs.last_hidden_state
    masked_position = [1]
    
    if subst_type == 'sub':
        hidden_states[:, masked_position, :] = torch.tensor(pred_child_emb).to(DEVICE)
    elif subst_type == 'mix':
        vec = torch.mean(torch.stack([hidden_states[:, masked_position, :].squeeze(), torch.tensor(pred_child_emb).to(DEVICE)]), dim=0)
        hidden_states[:, masked_position, :] = vec

    prediction_scores = maskedlm.cls(hidden_states)
    prediction_scores = prediction_scores.detach().squeeze(0)[1]
    probs = torch.softmax(prediction_scores, dim=-1)
    topk_prob, topk_indices = torch.topk(probs, 100)
    topk_prob = [t.item() for t in topk_prob]
    topk_tokens = tokenizer.convert_ids_to_tokens(topk_indices.cpu().numpy())
    candidates = list(zip(topk_tokens, topk_prob))
    candidates = [i for i in candidates if all(not j.isdigit() and not j in string.punctuation for j in i[0])]
    candidates = candidates[:num]
    return candidates


def predict_node_from_bert(word, a, cur_index):
    restop = a.get(word, generate_candidates(word))
    a[word] = restop
    res = a[word][cur_index]
    return res


def generate_candidates(word):
    pred_child_emb = model(torch.tensor(gbert_embs[word], dtype=torch.float32).to(DEVICE))
    vowels = "aeiouy"

    par = word.split(".")[0].replace("_", " ")
    if par[0] in vowels:
        art="an"
    else:
        art='a'
    text = f'[MASK] is {art} {par}'

    inputs = tokenizer(text, return_tensors='pt').to(DEVICE)
    mix = get_pred(inputs, subst_type='mix', pred_child_emb=pred_child_emb, num=100)
    candidates = filter_results([i for i in mix if i[0] not in set(wn.all_lemma_names()) and all(not j.isdigit() and not j in string.punctuation for j in i[0])])

    return [i[0] for i in candidates[:5]]
