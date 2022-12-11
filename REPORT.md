# LLM Sparsification homework 

## Models

My model choices are 
* E-o: BERT
* D-o: GPT-2
* E-D: Pegasus

These don't all have over 1B parameters, but I chose to go with models that were 
1. classic
2. had few parameters so they were easy to work with 
(Note: I originally went with BART for E-D but ran into issues and swapped it out)

## Weight Distributions

First note that the size of these models is variable

| model | n_params | 
| ----- | ---------|
| bert  | 109,482,240 |
| gpt-2 | 124,439,808 |
| pegasus | 570,797,056| 

There may be triples of models that are closer in size so that size does not confound with architecture in our analyses, 
but at least these are in the same order of magnitude. 

Now, to take a look at the parameter histograms, we see that they are all very peaky around 0, so, out of these hundreds of millions of parameters, a lot of parameters that are doing more-or-less nothing.  

![image](https://user-images.githubusercontent.com/25695528/206887850-e7945023-16d8-4ab2-960f-49e846828eec.png)

We can take a step towards quantifying how much nothing they are doing by plotting how the number of them that have a magnitude less than some cutoff changes as that cutoff grows:  

![image](https://user-images.githubusercontent.com/25695528/206889820-d7e28124-0d35-46c7-aee1-f24465248196.png)


This gives us the sense that the relative sparsity (where here I mean "fraction of parameters that are close to zero") decreases as we move from BERT to GPT2 to Pegasus. This corresponds with the sparsity decreasing as the number of parameters increases, but I'm not going to claim thats a general trend (I doubt it is), its just what we see here. 

To get another view of this, we can just consider the fraction of parameters that have a magnitude below some fixed cutoff (essentially just a vertical slice of the plots above). I arbitrarily chose the cutoff to be 0.1, which is maybe a little too big for a parameter to be mostly useless, but I think its an easy place to start, as the difference across the models is quite striking at this point in the curve

| model | frac params < 0.1 | 
| ----- | ---------|
| bert  | 0.98 |
| gpt-2 | 0.59 |
| pegasus | 0.23 | 

### By Layer

If we take a look at these distributions broken down by layer, we see that the earlier layers tend to have more parameters with lower magnitudes (note that I broke the E-D model down into its E and D submodules).

![image](https://user-images.githubusercontent.com/25695528/206888151-56b08bfd-87f8-4a0a-8edb-d56bb4a8c72b.png)

## Sparsification results

I used `pytorch.prune` for this part. For simplicity, I just did global pruning. 

Because the pytorch prune tutorial looks pretty simple, I assumed things would go smoothly. They didn't, and I ran into a lot of trouble. 

The first difficulty is that I ran into issues pruning the BART model, so I replaced it with Pegasus. Then trying to prune pegasus repeatedly crashed my compute node. So, I decided to just prune the E-o and D-o models. 

I wrote the pruned models to disk and used the examples code from the to run the benchmark [huggingface pytorch examples] (https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling)

In the interest of time, I was only able to run against one benchmar (wikitext).

### Pretrained models only

First I tried to see if I could get away without fine-tuning these models. The lackluster results below show that, no, I cant. (Especially the BERT model appears to have extremely bad performance on this task without finetuning). 
![image](https://user-images.githubusercontent.com/25695528/206889552-1ee2a804-74ed-45c3-8a8f-b70dc9f5c2b8.png)
![image](https://user-images.githubusercontent.com/25695528/206889589-530e8c88-cf35-40e0-8845-2cd47bc9c5fe.png)
Interestingly, in this case, GPT2 actually gets slower under pruning. This is kindof bizarre, since more of the weights are masked out. This may be related to the lack of fine-tuning, but I am not sure

### Fine-tuned models 

I encountered another difficulty, in that the fine-tuning and benchmarking evaluation code from  [huggingface pytorch examples] (https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling) repeatedly crashed when fine-tuning `bert`, so I was only able to fine-tune GPT2. Here are the results

![image](https://user-images.githubusercontent.com/25695528/206889795-77bcc9c1-074a-4f6f-a6f9-f8d8a85834ac.png)
![image](https://user-images.githubusercontent.com/25695528/206889797-4b282606-a819-475a-ad37-2fdc4e4b8e0a.png)

I am actually quite pleased with these results. This is the first time (!) I have successfully fine tuned a transformer and gotten clean results. Although there are a lot of training wheels still on here, that is something. 

I note that there is a very obvious sweetspot in terms of speed and accuracy at 50% pruning for this model and task. Of course, we expect this to vary with model and task, and pruning method. 

## Model Sizes
I had some difficulty computing the model sizes for this work. It seems like pytorch prune works by computing a mask on the parameters, rather than by actually deleting them. So, when I computed the sizes of the models, they were all the same. They are also all the same on disk.
Their sizes on disk are:
|model| size|
|----|----|
|bert | 418 M|
| gpt2 | 487 M  |

## Notes on difficulties in pruning transformers. 

Most of the difficulties I encountered above were just related to me being new at working with transformers. However, I the result I saw on model sizes gives me pause. It appears that pytorch prune stores pruned model parameters as just masked dense matrices. This gives us _no_ advantage on space taken up by the model, and I am slightly confused about why the speed improved. If we wish to deploy transformers in edge settings, it seems like advances need to be made in storing the sparsified models on disc and representing them as sparse datasstructures in memory. I am reminded of Rick's lecture on how modern hardware
