import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import re



def prune_gpt2_layers(model, ratio):

    for i in range(48):
        list= ['transformer.h.{}.attn.c_attn'.format(i),
                'transformer.h.{}.attn.c_proj'.format(i),
                #'h.{}.ln_1'.format(i), leave layer norms unpruned
                #'h.{}.ln_2'.format(i),
                'transformer.h.{}.mlp.c_fc'.format(i),
                'transformer.h.{}.mlp.c_proj'.format(i)]
        for name, module in model.named_modules():
            if name in list:
                prune.l1_unstructured(module, name='weight', amount=ratio)
                prune.remove(module, 'weight')
                prune.l1_unstructured(module, name='bias', amount=ratio)
                prune.remove(module, 'bias')



def prune_M2M100_layers(model, ratio):

    for i in range(23):
        mod_list= ['model.encoder.layers.{}.self_attn.k_proj'.format(i),
                'model.encoder.layers.{}.self_attn.v_proj'.format(i),
                'model.encoder.layers.{}.self_attn.q_proj'.format(i),
                'model.encoder.layers.{}.self_attn.out_proj'.format(i),
                'model.encoder.layers.{}.fc1'.format(i),
                'model.encoder.layers.{}.fc2'.format(i),
                'model.decoder.layers.{}.self_attn.k_proj'.format(i),
                'model.decoder.layers.{}.self_attn.v_proj'.format(i),
                'model.decoder.layers.{}.self_attn.q_proj'.format(i),
                'model.decoder.layers.{}.self_attn.out_proj'.format(i),
                'model.decoder.layers.{}.fc1'.format(i),
                'model.decoder.layers.{}.fc2'.format(i)]
        for name, module in model.named_modules():
            if name in mod_list:
                prune.l1_unstructured(module, name='weight', amount=ratio)
                prune.remove(module, 'weight')
                prune.l1_unstructured(module, name='bias', amount=ratio)
                prune.remove(module, 'bias')




def check_gpt_layer_sparsity(model, layer_num):
    print(
    "Sparsity in h.{}.attn.c_attn.weight: {:.2f}%".format(10,
        100. * float(torch.sum(model.transformer.h[layer_num].attn.c_attn.weight == 0))
        / float(model.transformer.h[layer_num].attn.c_attn.weight.nelement())
    )
)


