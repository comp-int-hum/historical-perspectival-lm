import os
from perspectival_language_models.custom import *

# WandB settings
WANDB_PROJECT = "perspectival_language_models_finetuning"


# training configs
DORA_LLAMA_CONFIG = "config/dora-llama8B.yaml"
MODEL_PATH = "${DATA_ROOT}/models/llama3/Meta-Llama-3-8B-converted" # TO CHANGE
