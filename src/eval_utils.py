from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from datasets import load_metric, load_dataset
import evaluate
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from os.path import join
import tqdm

def compute_ppl(model):

    max_length = model.config.n_positions
    stride = 512

    lls = []
    for i in tqdm.tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:,:-stride] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * stride

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / i)
    return ppl