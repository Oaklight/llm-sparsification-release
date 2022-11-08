# Utility functions for exploring sparsity of HuggingFace models
import torch
import numpy as np
import matplotlib.pyplot as plt
import re

def weight_size_distro(arr):
    """
    This function takes an array of weights and returns
    the distribution of their magnitudes and also plots
    them as a histogram
    """
    rv = np.histogram(np.abs(arr).flatten(),
                      bins=[0,.000001, .00001, .0001,.001,.01,.1,1, 10])
    counts, bins = rv
    pcts = counts / sum(counts)
    
    return counts, pcts


def plot_weight_distro(labels, values):
    """
    plotting function for plotting weight magnitude distributions
    """
    plt.bar(labels, values)
    plt.xticks(rotation = 45)
    for i in range(len(labels)):
        plt.text(i,values[i],round(values[i],4))
        

def collect_params(weight_dict, param_names):
    """
    Function for combining a specified list of parameter matrices
    into a single 1D vector
    """
    
    flat_params = np.array([None])
    for name in param_names:
        print(name)
        if weight_dict[name].numel() > 1:# filter out masked bias -10000
            if flat_params.any() == None:
                flat_params = weight_dict[name].flatten()
            else:
                flat_params = torch.cat((flat_params, weight_dict[name].flatten()))
                
    return flat_params


def gpt_layer_analysis(layer_num, weight_dict):
    """
    Function that takes the number of a gpt-XL
    layer and returns it's weight distribution
    and plots them
    """

    # search params keys for each layers params
    num = str(layer_num)
    search_str = "h." + num + "\.\S*"
    reg1 = re.compile(search_str)
    selects = reg1.findall('  '.join(weight_dict.keys()))
    
    layer = collect_params(weight_dict, selects)
    cnts, pcts = weight_size_distro(layer)
    plot_weight_distro(bins_labs, pcts1)
    
    return cnts, pcts