import os
from custom import *

# WandB settings
WANDB_PROJECT = "perspectival_language_models_pretraining"

WORK_DIR = f"1_training/work/pretraining/{PROJECT_NAME}"
ORIGINAL_WORK_DIR = WORK_DIR

# training configs
TRAINER_CONFIG_1 = "1_training/config/llama-smoll-345M.yaml"
TRAINER_CONFIG_2 = "1_training/config/llama-smoll-345M.yaml"
STUDENT_CONFIG = "1_training/config/llama-smoll-345M.yaml"
