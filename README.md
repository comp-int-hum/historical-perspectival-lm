# Perspectival Language Models

_A brief description of what your project does and why it’s useful._

## Table of Contents

- [Installation](#installation)
  - [Automatic Setup](#automatic-setup)
  - [Manual Setup](#manual-setup)
- [Usage](#usage)
  - [Training On New Data](#training-on-new-data-configuration)
    - [Pretraining](#pretraining)
    - [Finetuning](#finetuning)
  - [Recreating Experiments](#recreating-experiments-configuration)
    - [Data Preparation](#data-preparation)
    - [Training](#training)
    - [Evaluation](#evaluation)
  - [Start Run](#start-run)
- [Contact](#contact)

---

## Installation

First, clone the repository:

```bash
git clone https://github.com/comp-int-hum/historical-perspectival-lm.git
```

### Automatic Setup

There is an automatic setup script you can run:

```bash
./setup.sh
```

This script installs all necessary dependencies, performs any required environment setup, and prepares your project to run.

### Manual Setup

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Install the Evaluation Harness**:

   1. Clone the evaluation harness code:

      ```bash
      git clone -b historical-minimal-pairs https://github.com/sabrinaxinli/evaluation-pipeline-2024.git
      ```

   2. Install its dependencies:

      ```bash
      cd evaluation-pipeline-2024
      pip install -e .
      pip install minicons
      pip install --upgrade accelerate
      cd ..
      ```

---

## Usage

### Training On New Data (configuration)

1. **Prepare Data**  
   To train models on your own data, place the text files into the `custom_data/` directory. For each category, include the following text files:

   - `data.train`
   - `data.dev`
   - `data.test`

   An example can be seen in `custom_data/song_lyrics`, including the file `preprocessing.ipynb` for an example of how the data was preprocessed.

2. **Customize Data Loading**  
   Update the `custom.py` file to specify where the data is located:
   ```python
   # Data settings
   DATA = "LOAD_CUSTOM_DATA"  # set data loading method
   CUSTOM_DATA_DIRECTORY = "custom_data/song_lyrics"  # your custom data directory
   ```

### Pretraining

If you want to run pretraining (according to the BabyLlama2 training recipe), set:

```python
# RUN settings
RUN_PRETRAINING = True
```

in `custom.py`.

To change the model size and other training parameters, you can modify the configuration files in the `1_training/config` directory and then reference them in `1_training/custom_pretraining.py`:

```python
# training configs
TRAINER_CONFIG_1 = "1_training/config/llama-smoll-345M.yaml"
TRAINER_CONFIG_2 = "1_training/config/llama-smoll-345M.yaml"
STUDENT_CONFIG   = "1_training/config/llama-smoll-345M.yaml"
```

### Finetuning

To fine-tune a model using DoRa, set:

```python
# RUN settings
RUN_FINETUNING = True
```

in `custom.py`.

You also need to specify the base model path and configuration. For instance:

```python
DORA_LLAMA_CONFIG = "1_training/config/dora-llama8B.yaml"
MODEL_PATH        = "your_model_path"
```

If you use a model other than `llama3-8B`, you may need to update the configuration to target the correct modules.

### Recreating Experiments (configuration)

The pipeline is designed so that new models can be easily trained. Follow the steps below to recreate the results from the paper.

#### Data Preparation

To run the data preparation pipeline, set:

```python
DATA = "DATA_PREPARATION"
```

in `custom.py`.

This pipeline needs a local copy of the gutenberg corpus, which needs to be set in the `0_data_preparation/custom_data_preparation.py` file:
```python
GUTENBERG_PATH = "your_local_gutenberg_respository"
```

A quantized Llama3 70B model was used to identify work dates (requiring two NVIDIA 3090s and ~1 day of processing). The results were stored in `0_data_preparation/data/gb_authors_dates_1950.jsonl`, and by default, this file is not recomputed.

To force a complete recomputation, set:
```python
# Model and prompt settings
USE_DATES_FILE = False
```
in `0_data_preparation/custom_data_preparation.py`.


Alternatively, the whole data preparation step can be skipped by directly loading the papers training data from the custom_data/historical_data directory:

```python
PROJECT_NAME = "historical"
DATA = "LOAD_CUSTOM_DATA"
CUSTOM_DATA_DIRECTORY = "custom_data/historical_data"
```


#### Training

To train both the pretrained and finetuned models as in the paper, both run settings must be set to 'True' in the customs.py file:

```python
RUN_PRETRAINING = True
RUN_FINETUNING = True
```

For finetuning on Llama3 8B, a local path to the model should be provided in '1_training/custom_finetuning.py':
```python
MODEL_PATH        = "your_model_path"
```
The paper used this Llama model: [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

#### Evaluation
The paper evaluated over BLiMP and the cloze task. Set these in the custom.py file:
```python
RUN_EVALUATION = True
EVALUATION_TASKS_LIST = ["blimp", "cloze_task_topk"]
```

### Start run:
Once your data is prepared, you can start the training run locally by navigating to the `perspectival_language_models` directory and running:

```bash
scons -Q
```

To run via Slurm, adjust Slurm variables in `custom.py`:

```python
STEAMROLLER_ENGINE = 'slurm'
GPU_COUNT = 1
MEMORY = "64GB"
GPU_ACCOUNT = "gpu_account_name"
CPU_ACCOUNT = "cpu_account_name"
GPU_QUEUE = "gpu_queue"
CPU_QUEUE = "cpu_queue"
```

Then start the run with:

```bash
scons -Q STEAMROLLER_ENGINE=slurm
```


---

## Contact

For any questions or additional information, please reach out:

- **Email**: elisabeth.fittschen@gmail.com
- **GitHub Issues**: [Open an issue](https://github.com/comp-int-hum/historical-perspectival-lm/issues)

---
