
for model in `ls ./models | grep bert`
	do python run_clm.py --model_name_or_path ./models/$model --tokenizer_name "bert-base-uncased" --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --per_device_eval_batch_size 8 --do_eval --output_dir results-$model/
	done
