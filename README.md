# data-selectioon

## setup
```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
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
