import os
from custom import *

# WandB settings
WANDB_PROJECT = "perspectival_language_models_finetuning"

WORK_DIR = f"1_training/work_finetuning/{PROJECT_NAME}"
ORIGINAL_WORK_DIR = WORK_DIR

# training configs
DORA_LLAMA_CONFIG = "1_training/config/dora-llama8B.yaml"
MODEL_PATH = "${DATA_ROOT}/models/llama3/Meta-Llama-3-8B-converted" # TO CHANGE
