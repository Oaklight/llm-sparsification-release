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



