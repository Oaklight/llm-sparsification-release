## Model Selection

For my 3 models, I've chosen:

 - Encoder-only: BERT Large
 - Decoder-only: GPT2 XL
 - Encoder-Decoder: BART Large

Unfortunately, I was unable to find Billion-parameter models that were available in the Encoder-only and (computationally tractable) Encoder-Decoder regime. However, BERT Large is 349M, which is within an order of magnitude at least. 

## Sparsity Structure
To assess sparsity structure, I have decided to take a (log scale) histogram and CDF chart of the absolute values of the weights for each model, and then a second CDF separated by layer for further analysis.

### Histograms
For our models, we have the following distribution of weight magnitudes:

![image](https://user-images.githubusercontent.com/5566842/200707807-0601a7e1-42c1-436e-90cb-c512f2b3c883.png)

With GPT2-XL, we see a relatively normal distribution on the log scale, with many parameters very low - the large majority of the weights falling beneath $10^{-1}$ in value, about 30% beneath $10^{-2}$, and very few greater than 0.1.


![image](https://user-images.githubusercontent.com/5566842/200709225-a22f6b98-cbe8-4c2a-9736-7023e0ba5fb3.png)

For BART-Large, we see a slightly higher-magnitude set of weights - while many are still under $10^{-2}$, far greater are between that and $10^{-1}$, and there are a much larger amount greater than $0.1$, even approaching integer 1. This would suggest slightly less potential inherent sparsity, as more weights have semi-significant absolute values.

![image](https://user-images.githubusercontent.com/5566842/200709610-fc221b3d-6b82-449b-88fe-503e82b4c737.png)

For BERT-Large, we also see a sparsity structure focused within the $10^{-2}, 10^{-1}$ range, with only about 30% of weight magnitudes less than $10^{-2}$ and about 99% below $10^{-1}$. This suggests about 30% of the weights are easily sparsifiable, while a lot of the rest are relatively similar in magnitude. Very few weights are greater than $10^{-1}$ as well.

 ### CDFs
And the following Cumulative Distribution Function:

![image](https://user-images.githubusercontent.com/5566842/200709289-8db43471-750f-4658-98bb-7acfdd7f1cc2.png)

![image](https://user-images.githubusercontent.com/5566842/200709298-153cd7ec-d1d1-431b-a7f2-c87fcb38c58b.png)

![image](https://user-images.githubusercontent.com/5566842/200709615-299ca829-81c9-46e3-a19a-51db7dc3b107.png)

The CDFs tell pretty much the same story as the histogram - for GPT2-XL and BERT we see that nearly 100% of weight magnitudes are represented when we hit $10^{-1}$, while it's only about 50% for T5 V1.1 XL.

### CDFs by Layer
And the following Cumulative Distribution Function, split by layer of the model:

![image](https://user-images.githubusercontent.com/5566842/200709705-10409f15-e1b6-42bf-bb36-bc3021c6a1e8.png)

Splitting by layer, we see that for GPT2-XL the weight distributions are nearly all the same between layers, except for h layer 0, which seems to have a somewhat greater proportion less than $10^{-3}$ (meaning the leftmost bracket is higher). 

![image](https://user-images.githubusercontent.com/5566842/200709725-2e911100-b5a8-42de-99f3-16b1b194cff3.png)

We also see that BART has nearly identical sparsity structures, with no layers standing out particularly as ripe for sparsification.
![image](https://user-images.githubusercontent.com/5566842/200713250-8532b9d5-193a-47cc-9fb3-ed81d8f42866.png)


For BERT, we see nearly identical sparsity structures between the layers, with none particularly standing out.

## Sparsification
To sparsify the models, I used the [PyTorch Global Prune](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.global_unstructured.html) function to select the weights with the lowest N% of magnitudes (L1 norm) and prune them to 0, and then used the [PyTorch Prune Remove](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.remove.html#torch.nn.utils.prune.remove) function to apply the prunes, and then used the huggingface function to save the model. I applied this in one pass instead of iteratively pruning and re-training

## Benchmarks
For the two benchmarks, I chose Perplexity (as judged when fine-tuned on the Causal Language Modeling task) and XNLI accuracy, a mix of generation and classification tasks. 

### Perplexity
![image](https://user-images.githubusercontent.com/5566842/200710091-ec959d66-da99-4549-8d71-bf50dadc829d.png)

For GPT2, we see that the perplexity does not increase significantly when 10% sparsified, increase slightly after 50%, and then precipitously after 90% and above.

![image](https://user-images.githubusercontent.com/5566842/200710188-57c0dc59-5481-4073-b9e2-3f3b61e6e34b.png)

BERT is similar, although perplexity increases less quickly for 10% and 50%, but ends up roughly similar when sparsified 90% or more.

![image](https://user-images.githubusercontent.com/5566842/200710324-175cedc4-d7a9-459a-b40c-3e2f8c325124.png)

BART has a somewhat unbelievable plot for Perplexity, decreasing as the model became more sparsified and approaching 1 when fully sparsified. These are obviously bad results and should not be lent serious thought. However, of note, these statistics were generated with the exact same experimental setup as those of GPT2 and BERT, and statistics like these consistently occurred when generated, so I am not sure at this point what could have been done better.

### XNLI
For the second task, we perform Text Classification on pairs of sentences. Of note is that there are 3 classes, implying a 33.3% guessing baseline rate.

![image](https://user-images.githubusercontent.com/5566842/200712072-a65be070-c868-4a2e-bc1a-24705d8a4390.png)

For GPT, XNLI accuracy is roughly maintained through 10% sparsity, but as soon as it gets to 50% sparsified it reverts to guessing level, and stays there.

![image](https://user-images.githubusercontent.com/5566842/200712173-6eccab3a-04bf-47e9-8358-fb01fa659cb1.png)

For BART, XNLI accuracy drops immediately when sparsified. It never gets above guessing again.

![image](https://user-images.githubusercontent.com/5566842/200711809-9ff50052-57a4-47ac-b13d-ad113e5dec4f.png)

With BERT, XNLI accuracy stays ok and even increases (incredibly dubiously, a larger Eval set should be considered) through 10% sparsity. As soon as it hits 50% sparsity, though, it regresses to guessing levels and stays constant past that.


## Size 
![image](https://user-images.githubusercontent.com/5566842/200712254-1525f9e3-c792-4e70-b30e-90d3b4f253bd.png)

![image](https://user-images.githubusercontent.com/5566842/200712275-598bed6b-235c-4298-9c0f-ef8ed570e020.png)

![image](https://user-images.githubusercontent.com/5566842/200712280-ea8770a5-244f-4ccb-bb82-f8555a55e86b.png)

As seen above, none of the models shrink in size at all when sparsified. This is because the PyTorch pruning implementation does not eliminate part of networks, but rather just zeros out weights that are viewed as "pruned". So, it still requires the same amount of space to store. 
## Runtime

### CLM Runtime
![image](https://user-images.githubusercontent.com/5566842/200712529-ef909973-346f-4770-8eb2-687ab039c7dd.png)

![image](https://user-images.githubusercontent.com/5566842/200712550-263a5d3b-e4bd-408b-b82f-403b8b468dd5.png)

![image](https://user-images.githubusercontent.com/5566842/200712563-f820acd0-d376-47b1-8a7e-1cad6849a6d1.png)

### XNLI Runtime
![image](https://user-images.githubusercontent.com/5566842/200712702-5aae21b7-b18b-4440-bccb-7114662b2639.png)



![image](https://user-images.githubusercontent.com/5566842/200712747-51f35f34-56f7-42ab-aff3-e4331ccf9c73.png)

![image](https://user-images.githubusercontent.com/5566842/200712763-8ef4bdf9-d0a6-4edd-9fa8-72f858030c3d.png)


As discussed in Size, the PyTorch pruning simply sets parameters to 0 that are "pruned". So, there are no real performance gains with a commodity chip that doesn't skip 0-mults, as we are still just doing those multiplications. Of "note" are the XNLI runtimes for BERT and BART, which seem to indicate some ridiculous property of the 99% sparsified model. These were computed while the Polaris cluster Debug Queue (which this was running on) was particularly loaded, and there was peculaiar speed changes going on. So, I'd be a bit doubtful of those, especially since the prior argument still applies to them. 
## Challenges of Sparsification
When sparsifying a model, one of the main goals is to reduce model size while not reducing model performance by nearly as much. For a smaller model such as a Multi-Layer Perceptron or even all the way up to a Convolutional Neural Net, this is relatively straightforward - as we know exactly how they work (in practice and in theory). For Large Language Models, though, while we know how they work in theory, in practice we are still not sure how weights correspond to behavior. We do not have a direct mapping of "area of network" to "ability" like we do the brain or CNNs, and so we cannot say that just because an area of the network has low magnitude weights that it is not critical to some aspect of its performance. Sparsification becomes more of an art than a science, in that if we get lucky in the weights we pick we may not sacrifice performance, but we cannot know this deterministically. 

Additionally, a challenge (that I personally faced) is support for the sparsification. When saving the sparsified models, we see that the PyTorch tool does not in fact reduce the model size at all, but just zeros out those weights in place, meaning no amount of representation is changed. Additionally, while we did see some speed up in the sparsified models, it was not nearly enough to correlate with the exponentially less amount of active parameters, and implies that PyTorch computes these sparsified models naively and without the expected speedup. So, even if we do find a great method of sparsifying these networks without losing much ability, as of now we will gain neither space savings nor significant time savings. 

