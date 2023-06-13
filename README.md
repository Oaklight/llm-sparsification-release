# llm-sparsification-cvf


## Report


### Models assessed:
   - Encoder-only: DeBERTaxxl (1,564,549,632 parameters)
   - Decoder-only: GPT2xxl (1,557,611,200 parameters)
   - Encoder-Decoder: M2M100 (1,239,470,080)


### Research questions:
   - what fraction of parameters >> 0? overall? by layer?
   - how does this vary by layer?

See the notebooks: gptxl_exploration.ipynb, DeBERTa_exploration.ipynb, and M2M_exploration.ipynb

### Sparsified evaluation
   Performance and runtime evaluation of sparsified versions of the models at 10%, 50%, 90%, 95%, 99% on common benchmarks

See the notebooks: M2M_pruning_and_eval.ipynb and gpt2xl_pruning_notebook.ipynb

### Conclusion

My results demonstrate that at a certain threshold performance of sparsified models deteriorates rapidly. Thus a major challenge of sparsification of LLMs is finding a good balance between compression and performance. Furthermore LLM performance for a single network might deteriorate at different rates for different evaluation tasks, which makes finding the optimal level of sparsification more difficult as you either need to find a task-dependent level, or find a level that maintains a relatively high level of performance on a wide range of tasks.
