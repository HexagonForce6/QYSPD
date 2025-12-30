# Try the QYSPD pipeline on custom dataset
1. Prepare files according to `inference/config.yaml`.
2. Download Qwen2.5-7B-Instruct model from https://huggingface.co/Qwen/Qwen2.5-7B-Instruct.
3. Run inference code from `inference/run_exp.py`.

# Data used
1. The four comprehensive history files are from http://www.xueheng.net/.
2. The complete catalogue for classical genres is provided in `data/catalogue.csv`.
3. Fanqie dependency graphs for discussed characters are in `data/plots/`; they are generated using Graphviz.