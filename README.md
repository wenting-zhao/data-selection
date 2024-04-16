# data-selectioon

## setup
```
uv venv # unless already created
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install flash-attn --no-build-isolation
```

## doing stuff
1. get training data
```
bash scripts/download_training_data.sh
```

2. compute log p(validation example | training example)
```
bash scripts/compute_logprobs.sh
```

3. look at logprobs and examples
```
bash scripts/analyze_logprobs.sh
```

4. selecting training examples
```
bash scripts/select_train.sh
```

5. lora fine-tune on training examples
```
bash scripts/finetune_lora.sh
```
