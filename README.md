# Perspectival Language Models

Perspectival Language Models is the official codebase accompanying the paper “Pretraining Language Models for Diachronic Linguistic Change Discovery”. As part of this work, we offer a straightforward way to transfer the training pipeline to other text corpora for those interested in adapting our methods.

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


## Installation

First, clone the repository:

```bash
git clone https://github.com/comp-int-hum/historical-perspectival-lm.git
```

### Automatic Setup
For an installation of the required dependencies and evaluation harness please run the [setup.sh](setup.sh) script
```bash
./setup.sh
```

### Manual Setup
Alternatively this is the description for a manual setup:
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


## Usage

### Training On New Data (configuration)

#### Prepare Data
   To train models on your own data, place the text files into the [custom_data/](perspectival_language_models/custom_data/) directory. For each category, include:
   
   - `data.train`
   - `data.dev`
   - `data.test`

   For example, see [custom_data/song_lyrics](perspectival_language_models/custom_data/song_lyrics), which also contains a preprocessing file ([preprocessing.ipynb](perspectival_language_models/custom_data/song_lyrics/preprocessing.ipynb)).

   Update 
   [custom.py](perspectival_language_models/custom.py) 
   to specify where the data is located:
   ```python
   # Data settings
   DATA = "LOAD_CUSTOM_DATA"  # set data loading method
   CUSTOM_DATA_DIRECTORY = "custom_data/song_lyrics"  # your custom data directory
   ```

#### Pretraining

To run pretraining (according to the BabyLlama2 training recipe), set:

```python
# RUN settings
RUN_PRETRAINING = True
```

in 
[custom.py](perspectival_language_models/custom.py).

To change the model size or other training parameters, modify the relevant configuration files in `1_training/config` and reference them in 
[custom_pretraining.py](perspectival_language_models/1_training/custom_pretraining.py), for example:

```python
# training configs
TRAINER_CONFIG_1 = "1_training/config/llama-smoll-345M.yaml"
TRAINER_CONFIG_2 = "1_training/config/llama-smoll-345M.yaml"
STUDENT_CONFIG   = "1_training/config/llama-smoll-345M.yaml"
```

#### Finetuning

To fine-tune a model using DoRa, set:

```python
# RUN settings
RUN_FINETUNING = True
```

in 
[custom.py](perspectival_language_models/custom.py).

You also need to specify the base model path and configuration, for example:

```python
DORA_LLAMA_CONFIG = "1_training/config/dora-llama8B.yaml"
MODEL_PATH        = "your_model_path"
```

If you use a model other than `llama3-8B`, adjust the configuration to target the correct modules.

### Recreating Experiments (configuration)

Follow these steps to recreate the paper’s experiments.

#### Data Preparation

To run the data preparation pipeline, set:

```python
DATA = "DATA_PREPARATION"
```

in 
[custom.py](perspectival_language_models/custom.py).

You will need a local copy of the Gutenberg corpus; configure its path in 
[custom_data_preparation.py](perspectival_language_models/0_data_preparation/custom_data_preparation.py):

```python
GUTENBERG_PATH = "your_local_gutenberg_respository"
```

A quantized Llama3 70B model was used to identify work dates. The results were stored in [gb_authors_dates_1950.jsonl](perspectival_language_models/0_data_preparation/data/gb_authors_dates_1950.jsonl). This file is not recomputed by default. To force a complete recomputation, set:

```python
# Model and prompt settings
USE_DATES_FILE = False
```

in 
[custom_data_preparation.py](perspectival_language_models/0_data_preparation/custom_data_preparation.py).

Alternatively, skip data preparation by directly loading the paper’s training data from `custom_data/historical_data`:

```python
PROJECT_NAME = "historical"
DATA = "LOAD_CUSTOM_DATA"
CUSTOM_DATA_DIRECTORY = "custom_data/historical_data"
```

#### Training

To train both the pretrained and finetuned models as in the paper, enable both in 
[custom.py](perspectival_language_models/custom.py):

```python
RUN_PRETRAINING = True
RUN_FINETUNING = True
```

For finetuning on Llama3 8B, specify the local path in 
[custom_finetuning.py](perspectival_language_models/1_training/custom_finetuning.py):

```python
MODEL_PATH = "your_model_path"
```

The paper used this Llama model: [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

#### Evaluation

To replicate the paper’s evaluation on BLiMP and the cloze task, set:

```python
RUN_EVALUATION = True
EVALUATION_TASKS_LIST = ["blimp", "cloze_task_topk"]
```

in 
[custom.py](perspectival_language_models/custom.py).

### Start Run

Once your data is prepared, you can start the training run locally by navigating to the `perspectival_language_models` directory and running:

```bash
scons -Q
```

To run via Slurm, adjust Slurm variables in 
[custom.py](perspectival_language_models/custom.py):

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

## Contact

For questions or additional information, feel free to reach out:

- **Email**: elisabeth.fittschen@studium.uni-hamburg.de  
- **GitHub Issues**: [Open an issue](https://github.com/comp-int-hum/historical-perspectival-lm/issues)

