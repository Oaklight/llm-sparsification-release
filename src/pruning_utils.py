import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import re



def prune_gpt2_layers(ratio):

    for i in range(48):
        list= ['h.{}.attn.c_attn'.format(i),
                'h.{}.attn.c_proj'.format(i),
                #'h.{}.ln_1'.format(i), leave layer norms unpruned
                #'h.{}.ln_2'.format(i),
                'h.{}.mlp.c_fc'.format(i),
                'h.{}.mlp.c_proj'.format(i)]
        for name, module in model.named_modules():
            if name in list:
                prune.l1_unstructured(module, name='weight', amount=ratio)
                prune.remove(module, 'weight')
                prune.l1_unstructured(module, name='bias', amount=ratio)
                prune.remove(module, 'bias')

        

def check_gpt_layer_sparsity(layer_num):
    print(
    "Sparsity in h.{}.attn.c_attn.weight: {:.2f}%".format(10,
        100. * float(torch.sum(model.h[layer_num].attn.c_attn.weight == 0))
        / float(model.h[layer_num].attn.c_attn.weight.nelement())
    )
)