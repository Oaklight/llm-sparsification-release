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

![image](https://user-images.githubusercontent.com/25695528/206887899-f7b1d56c-f60e-4d8f-89c0-c6e2fc99ac85.png)

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



