## 1 
### Sparsification

### Pruning
Zero specific weights.

### Quantization
How can we better encode the data of the weights? Can we use lower precision encodings, which require less memory and should therefore allow cheaper inference on the same device, to run the llm?

### Distillation
Can we use a smaller model which has similar performance but is cheaper to run than the larger model?

[Here](https://intellabs.github.io/distiller/knowledge_distillation.html)

### MoEfication
Mixture of Experts -- e.g. create a series of smaller models each with expert knowledge, but only infer on a subset of the experts, thereby reducing inference costs.

## 2
Eo
Do
E-D

[Reddit](https://www.reddit.com/r/MLQuestions/comments/l1eiuo/when_would_we_use_a_transformer_encoder_only/)

```
BERT (or bidirectional encoder): when you work at sentence level, like sentence classification, e.g. given text x predict class y, or perform extractive tasks such as extractive question answering - or when you want to work with contextual word embeddings

GPT (or unidirectional decoder): when you want to perform language modeling or in general open-ended text generation, e.g. given text x predict the next words of x such as x_5, x_6, x_7, etc.

encoder-decoder: when you want to generate some text different with respect to the input, such as machine translation or abstractive summarization, e.g. given text x predict words y_1, y_2,y_3, etc.
```