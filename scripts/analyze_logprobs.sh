python analyze_logprobs.py \
--model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
--train_files data/train/processed/dolly/dolly_data.jsonl \
--max_seq_length 2048 \
--batch_size 32
