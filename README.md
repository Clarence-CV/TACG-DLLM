TACG
====
From Confidence to Commitment: Trajectory-Aware Commit Gating for Diffusion Language Model Decoding



A lightweight research codebase for fast inference experiments on masked diffusion language models, with support for **LLaDA 8B Instruct** and **Dream-v0-Instruct-7B**.

This repository is organized around one practical goal: experiments should be runnable directly from shell scripts in `scripts/`, while keeping model-specific evaluation logic inside `src/`.

## What is included

- LLaDA evaluation code
- Dream evaluation code
- runnable shell presets for `gsm8k`, `math`, `humaneval`, and `mbpp`
- task config files
- local benchmark data files used by the current setup

## Repository layout

```text
TACG/
├── data/
├── scripts/
├── src/
├── tasks/
├── requirements.txt
├── pip_install.sh
├── LICENSE
└── README.md
```

Key paths:

- `src/llada_evaluation.py`: main evaluation entry for LLaDA
- `src/dream_evaluation.py`: main evaluation entry for Dream
- `src/model/`: decoding implementations
- `scripts/`: runnable experiment presets
- `tasks/`: benchmark configs

## Environment

Recommended environment:

- Python 3.12
- CUDA-enabled PyTorch
- `transformers`
- `datasets`
- `numpy`
- `tqdm`
- `regex`

Example setup with conda:

```bash
conda create -n tacg python=3.12 -y
conda activate tacg
cd /your/path/to/TACG
bash pip_install.sh
```

The install script will:

- upgrade basic pip tooling
- install packages from `requirements.txt`
- install `huggingface_hub` CLI support
- print example commands for downloading local model checkpoints

If you already have a preferred CUDA / PyTorch stack, install that first and then run:

```bash
pip install -r requirements.txt
```

## Model checkpoints

Current scripts assume local checkpoints such as:

- `/your/checkpoints/LLaDa-8B`
- `/your/checkpoints/Dream-v0-Instruct-7B`

These can be overridden at runtime with environment variables.

Example:

```bash
MODEL_PATH=/path/to/model SAVE_DIR=./results_local bash scripts/run_llada_gsm8k.sh
```

Example manual downloads:

```bash
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir /your/checkpoints/LLaDa-8B
huggingface-cli download Dream-org/Dream-v0-Instruct-7B --local-dir /your/checkpoints/Dream-v0-Instruct-7B
```

## How to run experiments

All primary experiments are intended to run from shell scripts in `scripts/`.

Common runtime overrides:

- `GPU_ID`
- `CUDA_VISIBLE_DEVICES`
- `MODEL_PATH`
- `SAVE_DIR`

Example launch:

```bash
cd /your/path/to/TACG
GPU_ID=0 SAVE_DIR=./results_local bash scripts/run_llada_math.sh
```

For shared machines, the following pattern is usually more stable:

```bash
cd /your/path/to/TACG && \
TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=1 \
OMP_THREAD_LIMIT=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=0 \
SAVE_DIR=./results_local \
bash scripts/run_llada_math.sh
```

## LLaDA scripts

### GSM8K

```bash
bash scripts/run_llada_gsm8k.sh
bash scripts/run_llada_gsm8k_klass.sh
```

### MATH

```bash
bash scripts/run_llada_math.sh
bash scripts/run_llada_math_klass.sh
```

### HumanEval

```bash
bash scripts/run_llada_humaneval.sh
bash scripts/run_llada_humaneval_klass.sh
```

### MBPP

```bash
bash scripts/run_llada_mbpp.sh
bash scripts/run_llada_mbpp_klass.sh
```

There are also lightweight `llada_*.sh` scripts in the same directory, but the `run_llada_*` scripts are the main reproducible presets.

## Dream scripts

### GSM8K

```bash
bash scripts/dream_gsm8k.sh
bash scripts/dream_klass_gsm8k.sh
```

### MATH

```bash
bash scripts/dream_math.sh
bash scripts/dream_klass_math.sh
```

### HumanEval

```bash
bash scripts/dream_humaneval.sh
bash scripts/dream_klass_humaneval.sh
```

### MBPP

```bash
bash scripts/dream_mbpp.sh
bash scripts/dream_klass_mbpp.sh
```

## Outputs

Results are written under the chosen `SAVE_DIR`.

Typical outputs include:

- `all_results.json`
- stepwise decoding traces when `--save_steps` is enabled
- benchmark-specific sample files for code-generation tasks

 

## License

See `LICENSE`.
