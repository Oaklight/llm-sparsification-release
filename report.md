# llm-sparsification-cvf


## Report

1. Understand and distinguish these concepts: 
   - Sparsification: The process of sparsifying a model's weights (i.e. zeroing out weights )
   - Pruning: The selection of weights to sparsify
   - Quantization: Reducing the size of a model by converting data to a lower precision representation
   - Distillation: The process of transferring knowledge from a large model to a smaller one
   - MoEfication: Process of converting a model into a MoE version by only using some subset of the parameters based on the particular input.

2. Choose your models. *Pick 3 models, 1 from each category.* Each pick should be of more than 1B parameters before pruning.
   - Encoder-only: DeBERTaxxl
   - Decoder-only: GPT2xxl
   - Encoder-Decoder: M2M100


3. Devise approaches to assess sparsity structure in your choice of models and answer these questiosn:
   - what fraction of parameters >> 0? overall? by layer?
   - how does this vary by layer?

See the notebooks: gptxl_exploration.ipynb, DeBERTa_exploration.ipynb, and M2M_exploration.ipynb

4. Produce sparsified versions of your models at 10%, 50%, 90%, 95%, 99%, by either coding your methods or using existing tools provided below
   Explain the nature of your methods, regardless of whether you code it yourselves.

See the notebooks: M2M_pruning_and_eval.ipynb and gpt2xl_pruning_notebook.ipynb

5. Find 2 common benchmarks used by your models, by reviewing their publications. \
   Set them up and obtain baseline results of original models. \
   Compare performance of your sparsified versions with the baselines.

See the notebooks: M2M_pruning_and_eval.ipynb and gpt2xl_pruning_notebook.ipynb


6. Compare size of models and runtime for sparsified models.

See the notebooks: M2M_pruning_and_eval.ipynb and gpt2xl_pruning_notebook.ipynb


7. Explain the challenges of sparsification on LLMs.

My results demonstrate that at a certain threshold performance of sparsified models deteriorates rapidly. Thus a major challenge of sparsification of LLMs is finding a good balance between compression and performance. Furthermore LLM performance for a single network might deteriorate at different rates for different evaluation tasks, which makes finding the optimal level of sparsification more difficult as you either need to find a task-dependent level, or find a level that maintains a relatively high level of performance on a wide range of tasks.
