# Try the QYSPD pipeline on custom dataset
1. Prepare files according to `inference/config.yaml`.
2. Download Qwen2.5-7B-Instruct model from Hugging Face public repo https://huggingface.co/Qwen/Qwen2.5-7B-Instruct; update the `model_path` field in the yaml file accordingly.
3. Run inference code from `inference/run_exp.py`.

# Data used
1. The four comprehensive history documents are obtained from http://www.xueheng.net/.
2. The complete catalogue of classical genres is provided in `data/catalogue.csv`.
3. Fanqie dependency graphs for discussed characters are in `data/plots/`; they are rendered with Graphviz.

# Params
1. SamplingParams
    - batch_size: 32
    - temperature: 0.7
    - top_p: 0.9
    - repetition_penalty: 1.2
    - max_tokens: 512
2. Environment
    - Ubuntu 22.04
    - Python 3.10
    - CUDA 11.8
    - vllm 0.6.3