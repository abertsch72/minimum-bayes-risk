import os
import torch
from torch import nn
import pickle
import numpy as np

import sys
sys.path.append("./")
sys.path.append("./src/")
from src.recom_search.model.exec_setup import tokenizer, model, dataset, args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("results.pkl", 'rb') as f:
    all_data = pickle.load(f)

results_dir = "/data/alexx/lattice-search/output/data/sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_zip_0.75_0.0_0.9"
result_files = os.listdir(results_dir)

def read_result(dir, file):
    with open(os.path.join(dir, file), 'rb') as f:
        x = pickle.load(f)
    return x

all_losses = []
batch_shapes = []

id2idx = {ex['id']: i for i, ex in enumerate(dataset)}
loss_fct = nn.CrossEntropyLoss(reduction='none')

model = model.to(device)
for data, result_file in zip(all_data, result_files):
    output = read_result(results_dir, result_file)
    ex = dataset[id2idx[output.doc_id]]
    document = ex['document']
    sents = document.split('\n')
    inp = "\n".join(sents[:10])[:5000]
    input_ids = tokenizer(inp, return_tensors="pt", padding=True, 
        truncation=True, max_length=1024).input_ids.to(device)
    input_ids = input_ids.expand(len(data['texts']), -1)
    hypo_ids = tokenizer(data['texts'], return_tensors='pt', padding=True,
        truncation=True, max_length=128).input_ids.to(device)
    pad_locs = (hypo_ids == 1)[:, 1:] # pad token id is 1
    decoder_input_ids = torch.where(pad_locs, 1, hypo_ids[:, :-1])
    labels = torch.where(pad_locs, -100, hypo_ids[:, 1:])

    with torch.no_grad():
        model_output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    logits = model_output.logits

    loss = loss_fct(logits.view(-1, model.config.vocab_size), labels.view(-1))
    loss = loss.view(logits.shape[0], logits.shape[1])
    
    all_losses.append(loss.mean(dim=-1))
    batch_shapes.append([logits.shape[0], logits.shape[1]])

batch_shapes_arr = np.array(batch_shapes)
all_losses = torch.cat(all_losses, dim=0).detach().cpu().numpy()
with open("all_losses.npy", 'wb+') as f:
    np.save(f, all_losses)
with open("batch_shapes.npy", 'wb+') as f:
    np.save(f, batch_shapes_arr)
