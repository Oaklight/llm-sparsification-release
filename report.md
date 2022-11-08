## Model Selection

For my 3 models, I've chosen:

 - Encoder-only: BERT Large
 - Decoder-only: GPT2 XL
 - Encoder-Decoder: T5 V1.1 XL

Unfortunately, I was unable to find Billion-parameter models that were available in the Encoder-only regime. However, BERT Large is 349M, which is within an order of magnitude at least. 

## Sparsity Structure
To assess sparsity structure, I have decided to take a (log scale) histogram and CDF chart of the absolute values of the weights for each model, and then a second CDF separated by layer for further analysis.

### Histograms
For our models, we have the following distribution of weight magnitudes:

With GPT2-XL, we see a relatively normal distribution on the log scale, with many parameters very low - the large majority of the weights falling beneath $10^{-1}$ in value, about 30% beneath $10^{-2}$, and very few greater than 0.1.

For T5 V1.1-XL, we see a slightly higher-magnitude set of weights - while many are still under $10^{-2}$, far greater are between that and $10^{-1}$, and there are a much larger amount greater than $0.1$, even approaching integer 1. This would suggest slightly less potential inherent sparsity, as more weights have semi-significant absolute values.


For BERT-Large, ...TIODODODODODODOD

 ### CDFs
And the following Cumulative Distribution Function:

The CDFs tell pretty much the same story as the histogram - for GPT2-XL we see that nearly 100% of weight magnitudes are represented when we hit $10^{-1}$, while it's only about 50% for T5 V1.1 XL. For BERT, TODODODODOD

### CDFs by Layer
And the following Cumulative Distribution Function, split by layer of the model:

Splitting by layer, we see that for GPT2-XL the weight distributions are nearly all the same between layers, except for h layer 0, which seems to have a somewhat greater proportion less than $10^{-3}$ (meaning the leftmost bracket is higher). 

We also see that T5 V1.1 XL is far more diverse, with some layers containing as much as 3x more magnitudes under $10^{-3}$ as others. Those layers seem to have a solidly left-ward distribution as well, often hitting nearly 100% about 0.7 orders of magnitude earlier. 

For BERT, TOSDHJSDAOIASDJASDHNJASD

## Sparsification
To sparsify the models, I used the [PyTorch Global Prune](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.global_unstructured.html) function to select the weights with the lowest N% of magnitudes (L1 norm) and prune them to 0, and then used the [PyTorch Prune Remove](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.remove.html#torch.nn.utils.prune.remove) function to apply the prunes, and then used the huggingface function to save the model. I applied this in one pass instead of iteratively pruning and re-training

## Benchmarks

## Size 
## Runtime

## Challenges of Sparsification
When sparsifying a model, one of the main goals is to reduce model size while not reducing model performance by nearly as much. For a smaller model such as a Multi-Layer Perceptron or even all the way up to a Convolutional Neural Net, this is relatively straightforward - as we know exactly how they work (in practice and in theory). For Large Language Models, though, while we know how they work in theory, in practice we are still not sure how weights correspond to behavior. We do not have a direct mapping of "area of network" to "ability" like we do the brain or CNNs, and so we cannot say that just because an area of the network has low magnitude weights that it is not critical to some aspect of its performance. Sparsification becomes more of an art than a science, in that if we get lucky in the weights we pick we may not sacrifice performance, but we cannot know this deterministically. 

Additionally, a challenge (that I personally faced) is support for the sparsification. When saving the sparsified models, we see that the PyTorch tool does not in fact reduce the model size at all, but just zeros out those weights in place, meaning no amount of representation is changed. Additionally, while we did see some speed up in the sparsified models, it was not nearly enough to correlate with the exponentially less amount of active parameters, and implies that PyTorch computes these sparsified models naively and without the expected speedup. So, even if we do find a great method of sparsifying these networks without losing much ability, as of now we will gain neither space savings nor significant time savings. 

